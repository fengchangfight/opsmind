# OpsMind RAG — 开发手册 (DEV MANUAL)

**版本**: v0.3
**日期**: 2026-06-23

---

## 1. 快速开始

### 1.1 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 后端开发语言 |
| Node.js | 18+ | 前端构建工具链 |
| Docker | 24+ | Milvus standalone (etcd+minio+milvus) |
| 磁盘 | 3GB+ | Milvus 向量数据 + Docker 镜像 + 模型缓存 |

### 1.2 安装依赖

```bash
# 后端依赖
pip install fastapi uvicorn pymilvus pydantic-settings openai fastembed httpx \
    "llama-index>=0.11" sentence-transformers "mcp>=1.24" aiosqlite langgraph

# 前端依赖
cd frontend && npm install
```

### 1.3 配置 API Key

```bash
# 环境变量方式（推荐）
set LLM_API_KEY=sk-your-deepseek-api-key

# 切换 LLM 提供商
# DeepSeek: LLM_BASE_URL=https://api.deepseek.com/v1  LLM_MODEL=deepseek-v4-pro
# OpenAI:  LLM_BASE_URL=https://api.openai.com/v1     LLM_MODEL=gpt-4o-mini
# Ollama:  LLM_BASE_URL=http://localhost:11434/v1     LLM_MODEL=qwen2.5
```

### 1.4 启动 Milvus

```bash
docker compose up -d
docker compose ps  # 确认 etcd, minio, milvus, attu 均 healthy
```

### 1.5 数据摄入

```bash
# 增量索引 — content hash cache 跳过未修改文档
python scripts/ingest.py

# 可选参数（.env 中设置）:
# DEMO_CATEGORIES_RAW=confluence,github
# DEMO_MAX_DOCS_PER_CATEGORY=50
```

数据路径: `sampledata/all_documents/` (EnterpriseRAG-Bench 数据集)

### 1.6 启动服务

```bash
# 后端 (只 watch 源码目录，避免 DB 变化触发重载)
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

# 前端
cd frontend && npm run dev

# 或一键启动 (Windows)
start_demo.bat
```

访问地址:
- 前端: `http://localhost:5173` (Demo 用户 `alice` / `bob`，密码 `opsmind123`)
- API 文档: `http://localhost:8000/api/docs`
- Attu GUI: `http://localhost:8001`

---

## 2. 项目结构

```
opsmind-rag/
├── app/                              # 后端主包
│   ├── config.py                     # 配置 (Pydantic Settings)
│   ├── models/
│   │   └── document.py               # Document, Chunk, Citation, SearchResult
│   ├── connectors/                   # 数据接入层
│   │   ├── base.py                   # BaseConnector 抽象接口
│   │   └── txt_connector.py          # .txt 文件解析器
│   ├── retrieval/                    # 检索层 (LlamaIndex 驱动)
│   │   ├── chunker.py                # SimpleChunker
│   │   ├── embedder.py               # FastEmbedEmbedding (LlamaIndex) + BM25 sparse
│   │   ├── vector_store.py           # MilvusVectorStore (LlamaIndex) + hybrid search
│   │   └── reranker.py               # SentenceTransformerRerank (LlamaIndex)
│   ├── agents/                       # Agent 层
│   │   ├── retrieve_agent.py         # Query Expansion + Hybrid Search + Reranker
│   │   ├── reason_agent.py           # LangGraph 迭代推理 + MCP/native tool loop
│   │   └── reason_graph.py           # LangGraph StateGraph (evaluate → confidence → loop)
│   ├── api/                          # FastAPI
│   │   ├── main.py                   # 入口 + 生命周期 (MCP 启动, 模型缓存)
│   │   ├── auth.py                   # JWT 认证中间件
│   │   ├── schemas.py                # 请求/响应 Schema
│   │   └── routes/                   # query, retrieve, resume, sessions, auth, mcp
│   ├── persistence/                  # Repository 模式 (SQLite + PostgreSQL)
│   ├── context/                      # 上下文管理
│   │   ├── token_budget.py           # Token 预算分配器
│   │   ├── conversation_compactor.py # Head/Tail/Summary 压缩 (参考 Claude Code/Hermes/OpenCode)
│   │   ├── compaction_trigger.py     # 多路径触发 (preflight/overflow/manual)
│   │   └── context_orchestrator.py   # 上下文编排器
│   ├── tools/                        # 原生工具系统
│   │   ├── base.py                   # BaseTool + PermissionChecker + CircuitBreaker
│   │   ├── registry.py               # ToolRegistry (注册/发现/执行)
│   │   ├── datetime_tool.py          # get_current_time
│   │   ├── calculator_tool.py        # 算术
│   │   └── random_tool.py            # 随机数
│   └── mcp/                          # MCP 框架
│       ├── config.py                 # McpServerConfig (stdio/http/sse)
│       ├── server_task.py            # McpServerTask (连接/发现/重试)
│       ├── tool_adapter.py           # MCP → OpenAI function calling
│       └── manager.py                # McpManager (生命周期/工具调用)
├── scripts/
│   ├── ingest.py                     # LlamaIndex IngestionPipeline (增量索引)
│   ├── smoke_test.py                 # 冒烟测试
│   └── demo_mcp_server.py            # Demo MCP Server (echo + sysinfo)
├── frontend/                         # React + Vite + Tailwind
├── data/                             # SQLite DB + LangGraph checkpoint + ingest cache
├── docs/                             # PRD, HLD, 5×LLD
├── docker-compose.yml                # Milvus Standalone + Attu
├── start_demo.bat
├── DEV_MANUAL.md
└── DEPLOYMENT.md
```

---

## 3. API 接口说明

### 3.1 `POST /api/login`

登录获取 JWT token。

```json
POST /api/login  {"username": "alice", "password": "opsmind123"}
→ {"token": "...", "user": {"user_id": "alice", "display_name": "Alice Wang", "role": "sre"}}
```

### 3.2 `GET /api/query` (SSE)

流式问答接口，返回 Server-Sent Events。

**参数**: `?query=<str>&top_k=<int>&session_id=<str>`

| 事件 | 说明 |
|------|------|
| `agent_start` | Agent 开始执行 |
| `retrieval_result` | 检索完成 |
| `reasoning_step` | LangGraph 迭代进度 (step, confidence) |
| `tool_call` | MCP/native 工具调用开始 |
| `tool_result` | 工具执行结果 |
| `chunk` | 流式 token |
| `final_answer` | 最终答案 + citations |
| `interrupted` | 置信度不足，需人工介入 |

### 3.3 `POST /api/retrieve`

纯检索接口（不调用 LLM）。

```json
POST /api/retrieve  {"query": "...", "top_k": 5}
→ {"results": [...], "citations": [...], "latency_ms": 280}
```

### 3.4 `GET /api/sessions`

会话列表（需认证）。`DELETE /api/sessions/{id}` 删除。

### 3.5 `GET /api/mcp/status`

MCP server 连接状态。

### 3.6 `GET /health`

健康检查：`{"status": "ok", "docs_indexed": 260}`

---

## 4. 开发指南

### 4.1 添加新 Native Tool

```python
# 1. 创建 app/tools/my_tool.py
from app.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {"type": "object", "properties": {...}}

    async def execute(self, arguments: dict) -> str:
        return "result"

# 2. 在 app/tools/registry.py 的 create_default_registry() 中注册
registry.register(MyTool())
```

### 4.2 添加新 MCP Server

```python
# 在 main.py 的 _load_demo_mcp_servers() 中添加:
manager.add_server(McpServerConfig(
    name="my-server",
    transport=StdioConfig(command="python", args=["my_mcp_server.py"]),
))
```

### 4.3 切换 Embedding 模型

```bash
# .env 或环境变量
EMBEDDING_DENSE_MODEL=BAAI/bge-small-en-v1.5   # 384d, 轻量
EMBEDDING_SPARSE_MODEL=Qdrant/bm25              # BM25 稀疏
```

### 4.4 切换 LLM

```bash
set LLM_BASE_URL=https://api.openai.com/v1
set LLM_MODEL=gpt-4o-mini
```

### 4.5 本地开发流程

```bash
# 1. 冒烟测试
python scripts/smoke_test.py

# 2. 增量摄入 (快)
python scripts/ingest.py

# 3. 启动后端
uvicorn app.api.main:app --reload --reload-dir app

# 4. 前端热更新
cd frontend && npm run dev

# 5. 测试 API
curl "http://localhost:8000/api/query?query=test&top_k=3"
curl -X POST http://localhost:8000/api/retrieve -H "Content-Type: application/json" -d '{"query":"test"}'
```

---

## 5. 常见问题

### Q: MCP server 未连接
检查启动日志是否有 `[MCP] demo: connected, 2 tools`。若没有，检查 `scripts/demo_mcp_server.py` 是否存在。

### Q: Reranker 报 connection timeout
模型已缓存在 `~/.cache/huggingface/hub/`，启动时设 `HF_HUB_OFFLINE=1` 禁止联网。

### Q: uvicorn 频繁重启 (watchfiles)
数据库写入触发了 reload。使用 `--reload-dir app` 只 watch 源码。

### Q: 摄入反复处理同一文档
增量缓存 `data/ingest_cache_li/` 会跳过未修改文档。清除缓存后全量重建：`rm -rf data/ingest_cache_li/`。

### Q: 前端 MCP 工具不触发
LangGraph 路径使用 `json_object` 格式禁用 function calling。检测到 tools 时自动走 tool-loop 路径。

---

## 6. 性能基准 (Demo 环境)

| 指标 | 数值 | 说明 |
|------|------|------|
| 摄入速度 | ~2-5 docs/秒 | FastEmbed CPU + BM25 sparse |
| 检索延迟 | ~16ms (260 docs) | Milvus HNSW + BGE-small |
| Hybrid 检索 | ~30ms | Dense + Sparse + RRF |
| 端到端延迟 | ~3-8s | 含 LLM 推理 |
| 内存占用 | ~2GB | Python + Milvus (etcd/minio/milvus) |
| 向量维度 | 384d dense + BM25 sparse | BGE-small + Qdrant/bm25 |

---

## 7. 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.3 | 2026-06-23 | LlamaIndex 全栈集成 (SentenceSplitter, MilvusVectorStore, SentenceTransformerRerank, IngestionPipeline); 混合检索 dense+sparse; LangGraph 迭代推理; MCP 工具框架; SQLite 持久化 + 压缩; 用户登录; 增量索引 |
| v0.2 | 2026-06-21 | Milvus standalone, 上下文管理, session 持久化 |
| v0.1 | 2026-06-20 | 初始 Demo: FastAPI + ChromaDB + React |
