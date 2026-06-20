# OpsMind-RAG 总体设计文档 (HLD)

**版本**: v1.0  
**日期**: 2026-06-19  
**对应 PRD**: PRD_OpsMind_RAG.md v1.0

---

## 1. 设计目标

1. **架构生产级**: 每个模块都有清晰接口抽象，支持横向扩展和组件替换
2. **技术深度覆盖**: 不满足于 API 调用，深入使用各组件高级功能
3. **可观测性内建**: 从 Day 1 就内置追踪、日志、指标
4. **7 天可跑通**: 功能聚焦但代码完整，避免过度设计同时保留扩展性

---

## 2. 系统架构

### 2.1 逻辑架构（四层）

```
┌─────────────────────────────────────────────────────────────┐
│  用户交互层 (React + SSE)                                    │
│  Chat · Citation · AgentTrace · InterruptDialog             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  API 网关层 (FastAPI)                                        │
│  REST + SSE · 认证 · 限流 · 异常处理                         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  编排与运行时层                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ AgentRuntime │  │ LangGraph    │  │ Message Bus  │      │
│  │ 任务分解/调度 │  │ 状态机编排   │  │ Redis Streams│      │
│  │ 中断恢复     │  │ 条件边/循环  │  │ P2P/广播/PubSub│    │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Agent 层（三个核心 Agent）                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ RetrieveAgent │  │ ReasonAgent  │  │ ExecuteAgent │      │
│  │ 混合检索     │  │ 迭代推理     │  │ 工具调用     │      │
│  │ RRF+重排序   │  │ 置信度评估   │  │ 熔断/限流    │      │
│  │ 查询扩展     │  │ 人机中断     │  │ 审计日志     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  数据与基础设施层                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Milvus   │  │  Redis   │  │  SQLite  │  │ 对象存储  │   │
│  │ 混合向量库 │  │ 状态/消息 │  │ 元数据   │  │ 原始文件  │   │
│  │ HNSW+BM25 │  │ Streams  │  │ 审计日志 │  │ 备份     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 部署架构

**Demo 阶段（Docker Compose）**:
```
单节点: Milvus(standalone) + Redis + FastAPI App + React(Nginx)
```

**生产目标（K8s）**:
```
Ingress → FastAPI Replica × N → Redis Cluster → Milvus Cluster(Q/D/I Node)
                                    ↓
                              S3/MinIO + PostgreSQL
```

---

## 3. 核心组件设计

### 3.1 AgentRuntime（运行时引擎）

**定位**: 对标 Hermes Agent Runtime，负责任务分解、调度、状态管理。

**核心流程**:
```
用户查询 → TaskDecomposer 分解为子任务
         → ExecutionPlanner 生成执行计划（并行/串行/条件）
         → LangGraph 编排各 Agent 执行
         → 每阶段结束 checkpoint 到 Redis
         → 异常/中断时从 checkpoint 恢复
```

**关键抽象**:
- `Task`: 原子任务单元，含类型、参数、依赖、超时
- `ExecutionPlan`: 多阶段执行计划，同阶段任务并行
- `Session`: 用户会话状态，含历史消息、checkpoint 链
- `Event`: 流式事件（agent_start, retrieval_result, interrupted, final_answer）

### 3.2 RetrieveAgent（混合检索）

**定位**: 实现 OpenClaw 级别的混合检索能力。

**状态机（LangGraph）**:
```
ParseQuery → ExpandQuery → [并行] DenseSearch + SparseSearch
                                    ↓
                              RRF Fusion → Rerank(Top50→Top5)
                                    ↓
                              BuildContext（动态截断 + 引用标记）
```

**关键技术**:
| 技术 | 实现 | 深度点 |
|------|------|--------|
| 混合检索 | Milvus Hybrid Search (dense + sparse) | 自己实现 RRF 融合，不依赖框架黑盒 |
| 重排序 | BGE-Reranker-Large (ONNX CPU) | 粗排 Top-50 再精排，批处理优化 |
| 查询扩展 | LLM 生成 3-5 个变体查询 | 提升召回率，控制噪音 |
| Chunking | Markdown 结构感知 + 64 token 重叠 | 保留语义完整性 |
| 上下文增强 | 索引前 LLM 生成上下文前缀 | 解决孤立 chunk 语义丢失 |

### 3.3 ReasonAgent（迭代推理）

**定位**: 不是"检索一次就回答"，而是迭代深挖。

**状态机（LangGraph）**:
```
接收上下文 → 生成假设 → 评估证据 → 计算置信度
     ↑                                    │
     └──── 置信度<0.7 ──→ 提取知识缺口 ──┘
              ↓
         中断（等待人工输入）
              ↓
         恢复后继续 或 生成最终答案
```

**关键技术**:
- **多跳检索**: 证据不足时自动提取新关键词，发起第二轮检索（最多 3 轮）
- **置信度评估**: 基于证据覆盖度、来源一致性、信息新鲜度计算
- **中断恢复**: LangGraph `interrupt` + `checkpoint`，支持任意节点暂停和恢复

### 3.4 ExecuteAgent（工具调用）

**定位**: 生产级工具执行层，对标 OpenCode/Claude Code 的工具能力。

**核心机制**:
```
ToolRegistry
├── 注册/发现: 动态注册工具，无需重启
├── 权限控制: RBAC，每个工具绑定允许角色
├── 限流: Token Bucket，防止过载
├── 熔断: 连续失败 5 次后熔断，30s 后半开探测
├── 重试: 指数退避，最多 3 次
└── 降级: 主工具失败时切换备用实现
```

### 3.5 Message Bus（Agent 通信）

**定位**: 基于 Redis Streams 的异步通信总线。

**通信模式**:
| 模式 | 场景 | 实现 |
|------|------|------|
| 点对点 | Agent A 请求 Agent B 执行特定任务 | `XADD` 到目标 Agent 的 Stream |
| 广播 | 状态变更通知所有监听者 | `PUBLISH` 频道 |
| 发布订阅 | 某类事件的多消费者处理 | Consumer Group 消费同一 Stream |

**生产特性**:
- 消息幂等: 每个消息带唯一 ID，消费者记录已处理 ID
- 死信队列: 处理失败的消息进入 DLQ Stream
- 背压控制: 消费者通过 `XREADGROUP COUNT` 控制消费速率

---

## 4. 数据流设计

### 4.1 主查询链路

```
用户输入查询
    │
    ▼
┌─────────────┐
│ API Gateway │──→ 认证/限流/日志
└──────┬──────┘
       │
       ▼
┌─────────────┐
│AgentRuntime │──→ 生成 ExecutionPlan
│ TaskDecomposer│
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│RetrieveAgent│────→│  Milvus     │
│ 混合检索    │     │ HybridSearch │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ ReasonAgent │────→│  LLM API    │
│ 迭代推理    │     │ (Claude/GPT)│
│ 置信度评估  │     └─────────────┘
└──────┬──────┘
       │
       ├── 置信度 < 0.7 ──→ 中断 ──→ 等待人工输入 ──→ 恢复
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ExecuteAgent │────→│ ToolRegistry │
│ 工具调用    │     │ (模拟/真实) │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│  结果聚合   │──→ 生成带引用的最终答案
│  溯源标记   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   前端 SSE  │──→ 流式输出（agent_start → retrieval → reasoning → final_answer）
│   事件流    │
└─────────────┘
```

### 4.2 索引链路

```
原始文档 → Connector 提取 → 预处理（去重/去噪）
                                    │
                                    ▼
                              智能 Chunking（结构感知 + 重叠）
                                    │
                                    ▼
                              上下文增强（LLM 生成前缀）
                                    │
                                    ▼
                              Embedding（Dense + Sparse）
                                    │
                                    ▼
                              Milvus 写入（含元数据）
```

---

## 5. 可观测性设计

### 5.1 三层可观测

| 层级 | 技术 | 覆盖内容 |
|------|------|---------|
| **追踪** | OpenTelemetry | 全链路 trace，从 HTTP 请求到每个 LangGraph 节点到工具调用 |
| **指标** | Prometheus | 检索延迟、Agent 迭代次数、工具成功率、中断频率、消息队列堆积 |
| **日志** | 结构化 JSON | 所有组件统一格式，含 trace_id、span_id、agent_id、event_type |

### 5.2 关键指标

```
opsmind_retrieval_latency_seconds{stage="dense|sparse|fusion|rerank"}
opsmind_agent_iterations_total{agent="reason"}
opsmind_tool_execution_total{tool="query_metrics",status="success|failure|timeout"}
opsmind_interrupt_total{reason="low_confidence|tool_failure"}
opsmind_message_bus_lag_seconds{stream="agent:reason"}
```

---

## 6. 扩展性设计

### 6.1 可替换组件（通过配置切换）

| 组件 | 当前实现 | 可替换为 |
|------|---------|---------|
| 向量数据库 | Milvus | Qdrant、Weaviate、Pinecone |
| Embedding 模型 | BGE-M3 | OpenAI text-embedding-3、E5 |
| 重排序模型 | BGE-Reranker | Cohere Rerank、Jina Reranker |
| LLM | Claude 3.5 Sonnet | GPT-4o、Qwen2.5、本地 Ollama |
| 消息总线 | Redis Streams | RabbitMQ、Kafka、NATS |
| 状态存储 | Redis | PostgreSQL、MongoDB |

### 6.2 新增数据源（零侵入）

```python
class MyConnector(BaseConnector):
    async def extract(self, source: str) -> AsyncIterator[Document]:
        # 实现提取逻辑
        ...

    async def watch(self, source: str, callback: Callable):
        # 实现变更监听（可选）
        ...

# 注册即可使用
runtime.register_connector("my_source", MyConnector())
```

### 6.3 新增工具（零侵入）

```python
class MyTool(BaseTool):
    name = "my_tool"
    permissions = ["sre", "admin"]

    async def execute(self, params: dict, context: ExecutionContext) -> ToolResult:
        # 实现工具逻辑
        ...

# 注册即可使用
tool_registry.register(MyTool())
```

---

## 7. 技术选型与深度覆盖

| 技术 | 选型 | 用到的高级功能 | 学习产出 |
|------|------|---------------|---------|
| **LangGraph** | 状态机编排 | 条件边、循环、interrupt/checkpoint、StreamingEvents、持久化到 Redis | 复杂状态机编排、中断恢复 |
| **Milvus** | 混合向量库 | Hybrid Search (dense+sparse)、HNSW 参数调优、JSON 元数据过滤、强一致性 | 向量数据库生产级实践 |
| **LlamaIndex** | RAG 框架 | 自定义 Retriever、多路召回融合、Node Postprocessor 链、Response Synthesizer | RAG 流水线深度定制 |
| **Redis** | 状态/消息 | Streams Consumer Group、消息幂等、死信队列、背压控制、AOF 持久化 | 事件驱动架构 |
| **OpenTelemetry** | 分布式追踪 | 手动 Span 创建、上下文传递、与 LangGraph 集成、Jaeger 导出 | 全链路可观测 |
| **FastAPI** | API 框架 | 依赖注入、后台任务、SSE 流式输出、WebSocket、异常处理 | 高性能 API 设计 |
| **Pydantic** | 数据校验 | Settings 管理、复杂模型验证、Discriminated Union、自定义验证器 | Python 工程化 |
| **React** | 前端 | Zustand 状态管理、SSE EventSource、Agent 执行可视化、可恢复对话 | 现代前端架构 |

---

## 8. 目录结构

```
opsmind-rag/
├── docker/
│   ├── docker-compose.yml          # Milvus + Redis + App + Frontend
│   └── Dockerfile
├── opsmind/                        # 主包
│   ├── core/                       # 核心框架
│   │   ├── agent_runtime.py        # AgentRuntime + TaskDecomposer
│   │   ├── message_bus.py          # Redis Streams 封装
│   │   ├── session_manager.py      # 状态持久化
│   │   └── execution_planner.py    # 执行计划生成
│   ├── agents/                     # Agent 实现
│   │   ├── retrieve_agent.py       # 检索 Agent（LangGraph 状态机）
│   │   ├── reason_agent.py         # 推理 Agent（LangGraph 状态机）
│   │   └── execute_agent.py        # 执行 Agent + ToolRegistry
│   ├── retrieval/                  # 检索层
│   │   ├── milvus_store.py         # Milvus 混合检索封装
│   │   ├── chunker.py              # 智能 Chunking
│   │   ├── reranker.py             # Cross-Encoder 重排序
│   │   └── embedder.py             # Embedding 模型路由
│   ├── connectors/                 # 数据接入
│   │   ├── base.py                 # BaseConnector 抽象
│   │   ├── markdown_connector.py
│   │   └── bench_connector.py      # EnterpriseRAG-Bench 专用
│   ├── tools/                      # 工具实现
│   │   ├── base.py                 # BaseTool 抽象
│   │   ├── query_metrics.py        # 模拟 Prometheus
│   │   ├── query_logs.py           # 模拟 ELK
│   │   └── notify.py               # 模拟通知
│   ├── models/                     # Pydantic 模型
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── citation.py
│   │   └── task.py
│   ├── observability/              # 可观测性
│   │   ├── tracer.py               # OpenTelemetry 配置
│   │   ├── metrics.py              # Prometheus 指标
│   │   └── logger.py               # 结构化日志
│   └── api/                        # FastAPI 接口
│       ├── main.py
│       ├── routes/
│       │   ├── query.py            # /api/query (SSE)
│       │   ├── retrieve.py         # /api/retrieve
│       │   └── admin.py            # /api/admin
│       └── dependencies.py
├── frontend/                       # React 前端
│   └── src/
│       ├── components/
│       │   ├── Chat.tsx
│       │   ├── CitationPanel.tsx
│       │   ├── AgentTrace.tsx
│       │   └── InterruptDialog.tsx
│       └── stores/
│           └── chatStore.ts
├── data/                           # EnterpriseRAG-Bench 数据
├── scripts/                        # 工具脚本
│   ├── ingest.py
│   ├── index.py
│   └── evaluate.py
├── tests/                          # 测试
│   ├── unit/
│   └── integration/
├── pyproject.toml                  # Poetry 依赖
├── Makefile
└── README.md
```

---

## 9. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 为什么自建 AgentRuntime 而非直接用 LangGraph？ | LangGraph 负责单 Agent 状态机，自定义 Runtime 负责跨 Agent 任务分解和通信 | 对标 Hermes，学习 Agent 基础设施 |
| 为什么用 Redis Streams 而非 RabbitMQ？ | Streams 原生支持 Consumer Group 和消息持久化，与 Session 存储共用 Redis | 减少运维复杂度，Demo 阶段足够 |
| 为什么用 Milvus 而非 Qdrant？ | Milvus 2.4+ 原生支持 hybrid search (dense+sparse)，无需外部融合 | 减少自建融合逻辑，但 RRF 仍自己实现以学习原理 |
| 为什么用 SSE 而非 WebSocket？ | SSE 更适合服务器单向推送，实现简单，自动重连 | 前端用 EventSource 即可，降低复杂度 |
| 为什么用 SQLite 而非 PostgreSQL？ | Demo 阶段零配置，生产级接口已抽象，切换只需改配置 | 快速启动，不影响架构设计 |

---

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Milvus 混合检索性能不达标 | 降级为纯向量检索；或换 Qdrant/Weaviate |
| LLM API 限流/故障 | 模型路由：Claude → GPT → 本地 Ollama 自动降级 |
| LangGraph checkpoint 丢失 | Redis AOF 持久化 + 定期磁盘备份 |
| 7 天无法完成全部功能 | 按 P0/P1 优先级交付，P1 作为优化项 |
| 前端流式输出实现复杂 | 先返回完整结果，流式作为 P1 优化 |

---

## 11. 附录

### 11.1 参考资源
- [EnterpriseRAG-Bench](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Milvus Hybrid Search](https://milvus.io/docs/hybrid_search.md)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [OpenClaw RAG 架构分析](内部讨论)

### 11.2 变更日志

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-06-19 | 初始版本 | AI-assisted Design |
