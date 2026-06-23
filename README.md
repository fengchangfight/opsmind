# OpsMind RAG

> **Agentic RAG Platform for Enterprise Operations** — 混合检索 · 多 Agent 协作 · 流式推理 · MCP 工具

<p align="center">
  <img src="docs/opsmind.png" alt="OpsMind Demo" width="800">
</p>

---

## 架构

```
用户查询 → FastAPI (SSE) → RetrieveAgent → Milvus Hybrid Search (dense+sparse+RRF)
                                                ↓
                              Reranker (Cross-Encoder BGE-Reranker → Top-5)
                                                ↓
                              ReasonAgent (LangGraph iter + MCP/native tools)
                                                ↓
                              DeepSeek-v4 → SSE 流式输出 + Citation 溯源
```

### 技术栈

| 层 | 技术 |
|---|------|
| **前端** | React 18 + TypeScript + Vite + Tailwind CSS |
| **API** | FastAPI + SSE 流式输出 + JWT 认证 |
| **RAG 框架** | LlamaIndex (SentenceSplitter, MilvusVectorStore, SentenceTransformerRerank, IngestionPipeline, VectorStoreIndex) |
| **向量库** | Milvus 2.4 Standalone (HNSW dense + Sparse BM25 + RRFRanker 融合) |
| **Embedding** | BGE-small-en-v1.5 dense (384d) + BM25 sparse (LlamaIndex FastEmbedEmbedding) |
| **LLM** | DeepSeek-v4-pro (OpenAI 兼容 API) |
| **Agent 编排** | LangGraph (迭代推理 + 置信度评估 + interrupt/resume) |
| **会话持久化** | SQLite (Repository 模式, SQLite/PostgreSQL 双后端) |
| **上下文管理** | TokenBudget + ConversationCompactor + CompactionTrigger (参考 Claude Code / Hermes / OpenCode) |
| **工具系统** | Native ToolRegistry (3 内置工具) + MCP 框架 (stdio/http/sse) |
| **数据** | EnterpriseRAG-Bench 数据集 + LlamaIndex IngestionPipeline 增量索引 |

### 核心能力

- **混合检索** — Dense (BGE-small 384d) + Sparse (BM25) → Milvus `hybrid_search()` → RRFRanker(k=60)
- **Cross-Encoder 精排** — BGE-Reranker 粗排 Top-N → 精排 Top-5 (LlamaIndex SentenceTransformerRerank)
- **Query Expansion** — LLM 生成 3-5 变体查询，多路召回去重
- **迭代推理** — LangGraph StateGraph: 置信度 < 0.7 → 自动提取知识缺口 → re-retrieve (最多 3 轮)
- **多轮对话** — SQLite 持久化对话 + ConversationCompactor (Head/Tail/Summary) + 跨 Session 恢复
- **MCP 工具** — stdio/http/sse 多传输, 自动工具发现, 与 Native ToolRegistry 统一执行
- **流式推理** — SSE 逐 token 推送 + tool_call/tool_result/reasoning_step 事件
- **溯源引用** — 每个回答附带精确来源标记 `[1] [2]`，可追溯文档名和片段
- **用户体系** — JWT 登录 + Session per-user 隔离 + 前端三列布局 (会话/对话/引用)
- **增量索引** — content hash cache → 跳过未修改文档 → 自动清理已删除文档
- **生产就绪** — Docker Compose 一键部署, Milvus 存算分离, Repository 模式双 DB 后端

---

## 快速开始

### 环境要求

- Python 3.10+ / Node.js 18+ / Docker 24+

### 1. 启动 Milvus

```bash
docker compose up -d
```

### 2. 安装依赖 & 摄入数据

```bash
pip install fastapi uvicorn pymilvus pydantic-settings openai fastembed httpx "llama-index>=0.11" sentence-transformers mcp aiosqlite

# 摄入 EnterpriseRAG-Bench 文档 (增量索引，二次运行秒过)
python scripts/ingest.py
```

### 3. 配置 API Key

```bash
set LLM_API_KEY=sk-your-deepseek-api-key
```

### 4. 启动服务

```bash
# 后端
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

# 前端（另开终端）
cd frontend && npm install && npm run dev
```

访问 `http://localhost:5173` — Demo 用户 `alice` / `bob`，密码 `opsmind123`。

### 5. 管理面板

- Attu (Milvus GUI): `http://localhost:8001`
- API 文档 (Swagger): `http://localhost:8000/api/docs`

---

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/login` | POST | 用户登录 (JWT) |
| `/api/query` | GET (SSE) | 流式问答 — 检索 → 推理 → 逐 token 推送 |
| `/api/retrieve` | POST | 纯检索接口（调试用） |
| `/api/sessions` | GET/DELETE | 会话列表与删除 |
| `/api/mcp/status` | GET | MCP server 连接状态 |
| `/health` | GET | 健康检查 |

### SSE 事件流

```
event: agent_start       → {"agent_id": "retrieve"}
event: retrieval_result  → {"num_results": 5, "latency_ms": 280}
event: reasoning_step    → {"step": 1, "confidence": 0.85}       (LangGraph 迭代)
event: tool_call         → {"tool_name": "mcp_demo_echo", ...}   (MCP/native 工具)
event: tool_result       → {"tool_name": "...", "result": "..."}
event: chunk             → {"content": "根据..."}
event: final_answer      → {"answer": "...", "citations": [...]}
```

---

## 示例问题

基于当前 Default 摄入 (5 docs/类) 的精确匹配：

| 问题 | 命中文档 |
|------|---------|
| `How does the retention policy evaluation work in the control plane?` | ADR-017: Retention Policy |
| `How to validate a new model version with shadow deployment?` | Cohort-driven Shadow Validation |
| `What is the probe pruning strategy for cost-efficient evals?` | Probe Pruning and Adaptive Sampling |
| `How does evaluator backpressure and priority fairness work?` | Evaluator Backpressure Playbook |
| `How are operational eval sweeps prioritized?` | Eval Sweep and Prioritization Framework |
| `How does runtime-driven batching and quantization work?` | cost-pilot: batching + quantization |
| `How does eager QK fusion improve inference speed?` | Eager-QK fuse + rotary normalization |

**触发工具**: `What is the current time?` (native) · `Calculate 15 * 23 + 7` (native) · `Use the echo tool to say hello` (MCP)

---

## 项目结构

```
opsmind-rag/
├── app/                             # 后端
│   ├── config.py                    # 配置 (Pydantic Settings)
│   ├── models/                      # Document, Chunk, Citation, SearchResult
│   ├── connectors/                  # 可插拔数据接入层 (BaseConnector)
│   ├── retrieval/                   # Embedder, VectorStore (MilvusVectorStore), Reranker, Chunker
│   ├── agents/                      # RetrieveAgent, ReasonAgent, ReasonGraph (LangGraph)
│   ├── api/                         # FastAPI + SSE + JWT Auth
│   ├── persistence/                 # Repository 模式 (SQLite + PostgreSQL)
│   ├── context/                     # TokenBudget, ConversationCompactor, ContextOrchestrator
│   ├── tools/                       # Native ToolRegistry + BaseTool (3 内置工具)
│   └── mcp/                         # MCP 框架 (McpManager, McpServerTask, ToolAdapter)
├── frontend/                        # React + Vite + Tailwind
├── scripts/                         # ingest.py (LlamaIndex IngestionPipeline), smoke_test.py, demo_mcp_server.py
├── docs/                            # PRD, HLD, 5×LLD 设计文档
├── docker-compose.yml               # Milvus Standalone + Attu GUI
├── start_demo.bat                   # Windows 一键启动
├── DEV_MANUAL.md                    # 开发手册
└── DEPLOYMENT.md                    # 生产部署文档
```

---

## License

MIT
