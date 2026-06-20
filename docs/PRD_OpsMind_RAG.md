# OpsMind-RAG 需求文档 (PRD)

**版本**: v1.0  
**日期**: 2026-06-19  
**作者**: AI-assisted Design  
**状态**: 草案 → 开发基线

---

## 1. 项目概述

### 1.1 项目背景
企业运维场景中，故障排查知识分散在 Confluence、Jira、Slack、GitHub 等多源异构系统中。传统搜索无法处理语义关联，静态 RAG 又缺乏迭代推理能力。本系统旨在构建一个 **Agentic RAG 平台**，支持混合检索、多 Agent 协作推理、工具调用与人机协作，解决运维知识检索与根因分析的实际痛点。

### 1.2 项目定位
- **Demo 周期**: 7 天可跑通核心链路
- **生产目标**: 架构可直接扩展为生产级平台，支持多租户、多数据源、动态工具扩展
- **技术深度**: 每个技术组件都使用高级功能，覆盖编排、通信、中断恢复、可观测性等企业级特性

### 1.3 目标用户
- **初级 SRE**: 通过对话快速获取排查手册，降低学习曲线
- **资深运维**: 利用 Agent 迭代推理加速根因定位，自动关联历史案例
- **团队管理者**: 通过可观测性数据了解知识库使用情况和常见故障模式

---

## 2. 术语表

| 术语 | 定义 |
|------|------|
| **Agent** | 具有特定职责的 AI 实体，通过 LLM 驱动，可调用工具、与其他 Agent 通信 |
| **AgentRuntime** | 多 Agent 编排引擎，负责任务分解、调度、状态管理、中断恢复 |
| **Hybrid Search** | 稠密向量检索 + 稀疏关键词检索的融合策略 |
| **RRF** | Reciprocal Rank Fusion，倒数排名融合算法 |
| **Cross-Encoder** | 双塔式重排序模型，对 (query, doc) 对做精细相关性打分 |
| **Chunking** | 将长文档切分为语义完整的片段，用于向量索引 |
| **Contextual Embeddings** | 在索引时为每个 chunk 附加上下文前缀，解决孤立 chunk 语义丢失问题 |
| **Interrupt** | Agent 工作流中的人工介入点，系统暂停等待用户输入后恢复 |
| **Checkpoint** | 工作流状态的持久化快照，支持崩溃恢复和断点续传 |
| **ToolRegistry** | 工具注册中心，统一管理工具的注册、发现、权限、限流、熔断 |
| **Message Bus** | 基于 Redis Streams 的 Agent 间异步通信基础设施 |
| **Trace** | 通过 OpenTelemetry 实现的全链路追踪，覆盖从用户查询到最终响应的完整路径 |

---

## 3. 功能需求

### 3.1 核心功能矩阵

| 功能模块 | 功能点 | 优先级 | 验收标准 |
|---------|--------|--------|---------|
| **数据接入** | 支持 Markdown、JSON、PDF 格式文档导入 | P0 | 能解析 EnterpriseRAG-Bench 全部文档，提取标题、正文、元数据 |
| | 可插拔 Connector 架构 | P0 | 新增数据源只需实现 `BaseConnector` 接口，无需改动核心代码 |
| | 文档预处理（去重、去噪、元数据提取） | P1 | 重复文档自动识别，草稿/离题内容标记为低置信度 |
| | CDC 变更监听（接口预留） | P2 | Connector 提供 `watch()` 方法，生产级可接入 Kafka/Webhook |
| **索引层** | 稠密向量索引（BGE-M3） | P0 | Milvus 中 dense vector 维度 1024，支持 HNSW 索引 |
| | 稀疏向量索引（BM25 等效） | P0 | Milvus sparse vector 支持关键词精确匹配 |
| | 混合检索（Dense + Sparse + RRF 融合） | P0 | 单次查询同时走两条路径，RRF 融合后返回 Top-K |
| | 智能 Chunking（Markdown 结构感知 + 重叠窗口） | P0 | 按标题层级切分，相邻 chunk 64 token 重叠 |
| | 上下文增强（Contextual Embeddings） | P1 | 每个 chunk 索引前由 LLM 生成上下文前缀 |
| | 增量索引与版本控制 | P1 | 文档更新时只重建受影响 chunk，支持知识库版本回滚 |
| **检索 Agent** | 多路召回（Dense + Sparse） | P0 | 单次查询并行执行两种检索 |
| | RRF 融合排序 | P0 | 融合公式正确实现，不依赖框架黑盒 |
| | Cross-Encoder 重排序 | P0 | 粗排 Top-50 → 精排 Top-5，使用 BGE-Reranker-Large |
| | 查询扩展（Query Expansion） | P1 | LLM 生成 3-5 个查询变体，提升召回率 |
| | 溯源系统（Citation） | P0 | 每个 chunk 带精确来源定位（文档名、行号、段落） |
| | 上下文窗口动态管理 | P1 | 根据 LLM context limit 自动截断，优先保留高相关性内容 |
| **推理 Agent** | 单轮推理（检索 → 分析 → 回答） | P0 | 基础问答链路完整 |
| | 迭代推理（Multi-hop） | P0 | 证据不足时自动提取 knowledge gaps，发起第二轮检索，最多 3 轮 |
| | 置信度评估 | P0 | 基于证据覆盖度、来源一致性、信息新鲜度计算 0-1 分数 |
| | 人机协作中断（Human-in-the-loop） | P0 | 置信度 < 0.7 时暂停，返回中断原因和选项 |
| | 中断恢复（Resume） | P0 | 携带 session_id 和人工输入，从 checkpoint 继续执行 |
| | 假设生成与验证 | P1 | 先生成多个假设，再收集证据逐一验证或排除 |
| **执行 Agent** | 工具注册与发现 | P0 | `ToolRegistry` 支持动态注册/注销工具 |
| | 工具调用（权限校验 + 限流 + 超时） | P0 | 每次调用前检查用户权限，超默认 10s 自动失败 |
| | 错误重试（指数退避，最多 3 次） | P0 | 工具失败时自动重试，记录重试次数 |
| | 熔断降级 | P0 | 连续失败 5 次后熔断，30s 后半开探测，备用方案自动切换 |
| | 并行执行（无依赖工具） | P1 | 独立工具调用并行执行，有依赖的串行执行 |
| | 执行审计日志 | P1 | 每次工具调用记录 trace_id、参数、结果、耗时 |
| **AgentRuntime** | 任务分解（Task Decomposition） | P0 | 将用户查询拆分为可并行/串行的子任务 |
| | 状态机编排（LangGraph） | P0 | 使用 LangGraph 定义 Agent 工作流，支持条件边和循环 |
| | 状态持久化（Checkpoint） | P0 | 每个节点执行后自动保存到 Redis，支持崩溃恢复 |
| | 跨 Agent 通信（Message Bus） | P0 | Agent 间通过 Redis Streams 异步通信 |
| | 广播、点对点、发布订阅三种模式 | P1 | 支持不同通信模式适配不同协作场景 |
| | 消息幂等性 | P1 | 同一消息不重复处理 |
| | 死信队列（DLQ） | P2 | 处理失败的消息进入 DLQ，人工排查 |
| | 背压控制 | P2 | 消费者处理不过来时生产者自动降速 |
| | Session 隔离 | P0 | 不同用户 session 状态完全隔离 |
| **前端** | 对话式交互界面 | P0 | 类似 Claude 的左右分栏，左侧对话，右侧溯源 |
| | 实时流式输出（SSE） | P0 | LLM 生成内容实时推送，无需等待完整响应 |
| | 溯源高亮与跳转 | P0 | 引用标记可点击，跳转到源文档精确位置 |
| | Agent 执行过程可视化 | P1 | 显示当前活跃 Agent、已执行步骤、耗时 |
| | 人工介入面板 | P0 | 中断时显示原因、选项（继续/修改/转人工） |
| | 对话历史与恢复 | P1 | 支持回到历史对话某一步重新执行 |
| **可观测性** | 全链路追踪（OpenTelemetry） | P0 | 每个请求生成 trace，覆盖所有 Agent 和工具调用 |
| | 结构化日志（JSON） | P0 | 所有日志包含 trace_id、span_id、agent_id、event_type |
| | Prometheus 指标暴露 | P0 | 检索延迟、Agent 迭代次数、工具成功率、中断频率 |
| | 性能剖析（LangGraph StreamingEvents） | P1 | 实时暴露每个节点的执行时间和输入输出大小 |
| | 告警规则（接口预留） | P2 | 指标异常时触发告警，生产级可接入 PagerDuty |

### 3.2 非功能需求

| 维度 | 要求 | 说明 |
|------|------|------|
| **性能** | 单次查询端到端延迟 < 5s（本地环境） | 含检索、重排、推理、工具调用 |
| | 支持并发 10 用户 | Demo 级，生产级通过水平扩展解决 |
| **可用性** | 单节点部署即可运行 | Docker Compose 一键启动 |
| | 崩溃后能从 checkpoint 恢复 | 依赖 Redis 持久化 |
| **安全性** | 工具调用必须权限校验 | 每个工具绑定允许的用户角色 |
| | API 请求必须认证 | JWT Token 或 API Key |
| | 敏感操作必须审计 | 工具调用、中断恢复等记录完整日志 |
| **可扩展性** | 新增数据源零代码侵入 | 实现 Connector 接口即可 |
| | 新增工具零代码侵入 | 实现 BaseTool 接口，注册到 Registry |
| | 换 LLM 只需改配置 | 支持 OpenAI、Claude、Ollama 等 |
| | 换向量数据库只需改配置 | 接口抽象，支持 Milvus、Qdrant、Weaviate |
| **可维护性** | 代码覆盖率 > 70% | 核心逻辑必须有单元测试 |
| | 接口文档自动生成 | FastAPI 原生 OpenAPI + Swagger UI |
| | 部署文档完整 | README 包含架构说明、部署步骤、常见问题 |

---

## 4. 数据需求

### 4.1 数据源

**Demo 数据源**: EnterpriseRAG-Bench 数据集
- 包含 Q&A 对和对应的源文档（Markdown、JSON 格式）
- 覆盖企业内部常见文档类型：Slack 对话、Jira 工单、Confluence 页面、Google Drive 文件
- 包含半结构化噪音：离题讨论、草稿、重复页面

**生产级数据源（接口预留）**:
- Confluence（通过 REST API + Webhook）
- Jira（通过 REST API）
- Slack（通过 Event API）
- GitHub（通过 GraphQL API）
- 企业内部 Wiki（通过自定义 Connector）

### 4.2 数据模型

#### Document（原始文档）
```python
class Document(BaseModel):
    doc_id: str                    # 全局唯一 ID（UUID）
    source: str                    # 来源标识（如 "confluence://space/page"）
    source_type: str               # 来源类型（markdown, json, pdf, confluence...）
    title: str
    content: str                   # 原始内容
    metadata: dict                 # 扩展元数据（作者、创建时间、标签、团队等）
    created_at: datetime
    updated_at: datetime
    version: str                   # 文档版本号
    status: str                    # 状态：active, archived, draft
```

#### Chunk（索引片段）
```python
class Chunk(BaseModel):
    chunk_id: str                  # 全局唯一 ID
    doc_id: str                    # 关联的 Document
    content: str                   # 片段内容
    context_prefix: str | None     # 上下文增强前缀（Contextual Embeddings）
    embedding_dense: list[float]   # 稠密向量
    embedding_sparse: dict         # 稀疏向量（token_id: weight）
    start_line: int                # 在原文中的起始行号
    end_line: int                  # 在原文中的结束行号
    section_path: list[str]        # 文档结构路径（如 ["H1", "H2", "H3"]）
    metadata: dict                 # 继承自 Document 的元数据
    index_version: str             # 索引版本号（支持版本回滚）
```

#### Citation（溯源引用）
```python
class Citation(BaseModel):
    citation_id: str               # 引用编号（如 [1], [2]）
    chunk_id: str
    doc_id: str
    doc_title: str
    excerpt: str                   # 引用的原文片段
    source_url: str | None         # 可点击的源文档链接
    relevance_score: float         # 相关性分数
```

#### Task（Agent 任务）
```python
class Task(BaseModel):
    task_id: str
    session_id: str
    task_type: str                 # retrieve, reason, execute, composite
    description: str
    parameters: dict
    dependencies: list[str]        # 依赖的其他 task_id
    priority: int                  # 优先级（数值越小越优先）
    timeout: int                   # 超时时间（秒）
    max_retries: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    status: str                    # pending, running, completed, failed, interrupted
    result: dict | None
    error: str | None
```

---

## 5. 接口需求

### 5.1 用户接口（REST API）

#### POST /api/query
**描述**: 主查询接口，触发完整 RAG + Agent 推理链路  
**请求体**:
```json
{
  "query": "MySQL 主从延迟如何排查？",
  "session_id": "optional-existing-session",
  "context": {
    "user_id": "user-123",
    "team": "sre",
    "env": "production"
  },
  "options": {
    "max_iterations": 3,
    "enable_interrupt": true,
    "tools": ["query_metrics", "query_logs"]
  }
}
```
**响应（流式 SSE）**:
```json
// 事件类型：agent_start
{"event": "agent_start", "agent_id": "retrieve", "timestamp": "..."}

// 事件类型：retrieval_result
{"event": "retrieval_result", "chunks": [...], "citations": [...]}

// 事件类型：agent_start
{"event": "agent_start", "agent_id": "reason", "timestamp": "..."}

// 事件类型：reasoning_step
{"event": "reasoning_step", "step": 1, "hypothesis": "...", "confidence": 0.65}

// 事件类型：interrupted
{"event": "interrupted", "reason": "置信度不足，需要确认", "options": [...]}

// 或事件类型：final_answer
{"event": "final_answer", "answer": "...", "citations": [...], "tools_called": [...]}
```

#### POST /api/resume
**描述**: 从中断点恢复执行  
**请求体**:
```json
{
  "session_id": "sess-xxx",
  "human_input": "确认继续，重点关注网络延迟",
  "option": "continue"
}
```

#### POST /api/retrieve
**描述**: 仅检索，不推理（用于调试和独立使用）  
**请求体**:
```json
{
  "query": "MySQL 主从延迟",
  "top_k": 10,
  "filters": {
    "team": "sre",
    "doc_type": "runbook"
  }
}
```

#### GET /api/admin/sessions/{session_id}
**描述**: 查看 session 完整状态（用于调试）  
**响应**: 包含当前状态、历史消息、checkpoint 位置等

#### GET /api/admin/metrics
**描述**: Prometheus 指标暴露端点

### 5.2 Agent 间接口（Message Bus）

**Redis Streams 消息格式**:
```json
{
  "message_id": "msg-uuid",
  "correlation_id": "trace-id",
  "sender": "agent://reason",
  "recipient": "agent://retrieve",
  "message_type": "REQUEST|RESPONSE|BROADCAST|EVENT",
  "payload": {
    "task_id": "task-uuid",
    "action": "retrieve",
    "parameters": {...}
  },
  "timestamp": "2026-06-19T21:14:00Z",
  "ttl": 300
}
```

---

## 6. 约束与假设

### 6.1 技术约束
- **Python 3.11+**：利用 asyncio、typing 新特性
- **Milvus 2.4+**：需要 hybrid search 和 JSON 元数据支持
- **Redis 7.0+**：需要 Streams 的 Consumer Group 功能
- **LLM API**: 需要支持 function calling 的模型（Claude 3.5 Sonnet、GPT-4o、Qwen2.5 等）

### 6.2 资源约束（Demo 环境）
- **机器**: 单台 8C16G 云服务器或本地开发机
- **GPU**: 非必须，Embedding 和 Reranker 使用 CPU 量化版本（BGE-M3、BGE-Reranker 的 ONNX 版本）
- **存储**: 50GB SSD（Milvus 数据 + 文档原始文件）

### 6.3 假设
- EnterpriseRAG-Bench 数据集可公开下载且允许研究使用
- 目标用户具备基础运维知识，能理解技术术语
- Demo 阶段不接入真实企业系统，使用模拟数据
- 生产级部署时，网络、认证、Secrets 管理由基础设施团队提供

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Milvus 混合检索性能不达标 | 检索延迟 > 5s | 降级为纯向量检索，或换用 Qdrant/Weaviate |
| LLM API 限流/故障 | 系统不可用 | 实现模型路由，主模型故障时自动切换备用模型 |
| LangGraph checkpoint 丢失 | 中断后无法恢复 | Redis AOF 持久化 + 定期备份到磁盘 |
| EnterpriseRAG-Bench 数据格式复杂 | 解析失败 | 先处理干净子集，逐步增加复杂度 |
| 前端流式输出实现复杂 | Day 7 无法完成 | 降级为先返回完整结果，流式作为 P1 优化 |

---

## 8. 验收标准

### 8.1 功能验收
- [ ] 能成功索引 EnterpriseRAG-Bench 全部文档
- [ ] 输入 Bench 任意问题，系统返回带引用的答案
- [ ] 迭代推理至少完成 1 轮 self-correction
- [ ] 工具调用成功（模拟工具即可）
- [ ] 人工中断后能成功恢复
- [ ] 新增一个模拟工具，零代码侵入核心系统

### 8.2 性能验收
- [ ] 单次查询端到端延迟 < 5s（本地环境，不含 LLM 首 token 时间）
- [ ] 支持 5 个并发查询不崩溃

### 8.3 可观测性验收
- [ ] 能在日志中看到完整的 trace_id 贯穿所有组件
- [ ] Prometheus 指标能正常暴露和查询
- [ ] LangGraph 每个节点的执行时间可观测

### 8.4 扩展性验收
- [ ] 实现一个新的 `MockConnector`，注册后即可接入新数据源
- [ ] 修改配置切换 LLM 模型（如从 Claude 切换到 GPT）
- [ ] 修改配置切换向量数据库（接口预留，Milvus 实现优先）

---

## 9. 附录

### 9.1 参考资源
- [EnterpriseRAG-Bench](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Milvus Hybrid Search](https://milvus.io/docs/hybrid_search.md)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [OpenClaw RAG 架构分析](内部讨论)

### 9.2 变更日志

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-06-19 | 初始版本 | AI-assisted Design |
