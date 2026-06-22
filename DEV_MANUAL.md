# OpsMind RAG — 开发手册 (DEV MANUAL)

**版本**: v0.1  
**日期**: 2026-06-20

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
# 后端依赖 (在 opsmind-rag 根目录)
pip install fastapi uvicorn pymilvus pydantic-settings openai fastembed httpx

# 前端依赖
cd frontend
npm install
```

### 1.3 配置 API Key

```bash
# 方式一：环境变量（推荐，不落盘）
set LLM_API_KEY=sk-your-deepseek-key

# 方式二：.env 文件
cp .env.example .env
# 编辑 .env，填写 LLM_API_KEY

# 切换 LLM 提供商
# OpenAI:    LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
# DeepSeek:  LLM_BASE_URL=https://api.deepseek.com/v1  LLM_MODEL=deepseek-v4-pro
# Ollama:    LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5
```

### 1.4 启动 Milvus

```bash
# 启动 Milvus standalone（etcd + minio + milvus）
docker compose up -d

# 检查状态（约 30s 后可达 healthy）
docker compose ps
```

### 1.5 数据摄入

```bash
# 快速摄入（Demo 用，约 60-120 秒）
python scripts/ingest.py

# 可选参数（在 .env 或环境变量中设置）:
# DEMO_CATEGORIES_RAW=confluence,github   # 要索引的类别
# DEMO_MAX_DOCS_PER_CATEGORY=50          # 每类最多文档数
```

数据路径: `sampledata/all_documents/` (EnterpriseRAG-Bench 数据集)

### 1.6 启动服务

```bash
# 后端
uvicorn opmind.api.main:app --host 0.0.0.0 --port 8000 --reload

# 前端 (另开终端)
cd frontend && npm run dev

# 或一键启动 (Windows)
start_demo.bat
```

访问地址:
- 前端: `http://localhost:5173`
- API 文档 (Swagger): `http://localhost:8000/api/docs`
- 健康检查: `http://localhost:8000/health`

---

## 2. 项目结构

```
opsmind-rag/
├── opmind/                          # 后端主包
│   ├── config.py                    # 配置 (Pydantic Settings)
│   ├── models/                      # 数据模型
│   │   └── document.py              # Document, Chunk, Citation, SearchResult
│   ├── connectors/                  # 数据接入层
│   │   ├── base.py                  # BaseConnector 抽象接口
│   │   └── txt_connector.py         # .txt 文件解析器
│   ├── retrieval/                   # 检索层
│   │   ├── chunker.py               # 智能文档切分 (SimpleChunker)
│   │   ├── embedder.py              # 向量生成 (FastEmbed, BGE-small)
│   │   └── vector_store.py          # 向量存储 (Milvus via pymilvus)
│   ├── agents/                      # Agent 层
│   │   ├── retrieve_agent.py        # 检索 Agent
│   │   └── reason_agent.py          # 推理 Agent (DeepSeek/OpenAI)
│   └── api/                         # FastAPI 接口
│       ├── main.py                  # App 入口 + 生命周期
│       ├── schemas.py               # 请求/响应 Schema
│       └── routes/
│           ├── query.py             # GET /api/query (SSE 流式)
│           ├── retrieve.py          # POST /api/retrieve (纯检索)
│           └── resume.py            # POST /api/resume (中断恢复)
├── scripts/
│   ├── ingest.py                    # 文档摄入脚本
│   └── smoke_test.py                # 冒烟测试
├── frontend/                        # React 前端
│   └── src/
│       ├── components/Chat.tsx      # 主对话组件
│       ├── api/client.ts            # API 客户端 + SSE
│       └── types/index.ts           # TypeScript 类型
├── docker-compose.yml               # Milvus + 依赖容器
├── docs/                            # 设计文档
│   ├── PRD_OpsMind_RAG.md
│   ├── HLD_OpsMind_RAG.md
│   └── LLD_OpsMind_RAG_0*.md
├── .env.example                     # 配置模板
├── start_demo.bat                   # Windows 一键启动脚本
└── DEV_MANUAL.md                    # 本文件
```

---

## 3. API 接口说明

### 3.1 `GET /api/query` (SSE)

流式问答接口，返回 Server-Sent Events。

**请求**: `GET /api/query?query=<str>&top_k=<int>&category=<str>`

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| query | string | 必填 | 用户查询 |
| top_k | int | 5 | 返回文档数 |
| category | string | 可选 | 过滤文档类别 (confluence, github) |

**SSE 事件流**:

```
event: agent_start       → {"agent_id": "retrieve"}
event: retrieval_result  → {"num_results": 3, "latency_ms": 280}
event: agent_start       → {"agent_id": "reason"}
event: chunk             → {"content": "根据..."}
event: chunk             → {"content": "..."}
event: final_answer      → {"answer": "...", "citations": [...], "model": "deepseek-v4-pro"}
event: error             → {"code": "INTERNAL", "message": "..."}
```

### 3.2 `POST /api/retrieve`

纯检索接口（不调用 LLM）。

**请求体**:
```json
{"query": "MySQL replication lag", "top_k": 5, "filters": {"category": "confluence"}}
```

**响应**:
```json
{
  "query": "MySQL replication lag",
  "results": [
    {"chunk_id": "...", "content": "...", "doc_title": "...", "score": 0.92}
  ],
  "citations": [...],
  "latency_ms": 280
}
```

### 3.3 `POST /api/resume`

中断恢复接口（Demo 为简化实现，重新推理）。

**请求体**:
```json
{"session_id": "sess-xxx", "human_input": "请更关注网络因素", "option": "continue"}
```

### 3.4 `GET /health`

健康检查：`{"status": "ok", "docs_indexed": 254}`

---

## 4. 开发指南

### 4.1 添加新数据源

实现 `BaseConnector` 接口：

```python
from opmind.connectors.base import BaseConnector
from opmind.models import Document

class MyConnector(BaseConnector):
    connector_name = "my_source"
    supported_types = ["json"]

    async def extract(self, source: str) -> AsyncIterator[Document]:
        # 从数据源读取文档
        yield Document(
            doc_id="...",
            source=f"my_source://...",
            source_type="json",
            title="...",
            content="...",
            metadata={"category": "custom"},
        )
```

### 4.2 切换 Embedding 模型

```python
# config.py 或环境变量
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # FastEmbed 本地模型
EMBEDDING_MODEL=text-embedding-3-small  # OpenAI API (需改 embedder.py)
```

### 4.3 切换 LLM

环境变量即可：
```bash
# DeepSeek
set LLM_BASE_URL=https://api.deepseek.com/v1
set LLM_MODEL=deepseek-v4-pro

# OpenAI
set LLM_BASE_URL=https://api.openai.com/v1
set LLM_MODEL=gpt-4o-mini

# Ollama 本地
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_MODEL=qwen2.5
```

### 4.4 切换向量数据库

当前 Milvus standalone → 修改 `retrieval/vector_store.py` 的 `VectorStore` 类即可，接口签名不变。

### 4.5 本地开发流程

```bash
# 1. 修改代码后跑冒烟测试
python scripts/smoke_test.py

# 2. 若改了 embedding/chunking，需要重新摄入
python scripts/ingest.py

# 3. 启动后端开发模式
uvicorn opmind.api.main:app --reload

# 4. 前端热更新开发
cd frontend && npm run dev

# 5. 用 curl 测试
curl "http://localhost:8000/api/query?query=test&top_k=3"
curl -X POST http://localhost:8000/api/retrieve -H "Content-Type: application/json" -d '{"query":"test","top_k":3}'
```

### 4.6 代码规范

- Python: 类型注解必须，使用 `async/await`
- TypeScript: `strict: true`, `noUncheckedIndexedAccess`
- 文档字符串: Google/Numpy 风格
- 提交信息: `<type>: <description>` (feat, fix, docs, refactor)

---

## 5. 常见问题

### Q: 摄入时报 "CollectionExists" / 数据冲突
重新摄入前清空旧数据：`python -c "from opmind.retrieval.vector_store import VectorStore; VectorStore().clear()"`

### Q: Milvus 连接失败
确认 `docker compose ps` 容器都在运行。若 etcd 报错，尝试 `docker compose down && docker compose up -d`。

### Q: SSE 连接断开
前端 EventSource 支持自动重连。若持续断开，检查 CORS 配置和防火墙。

### Q: doc_title 为空
需确保 chunker 中 `metadata` 包含 `doc_title` 字段。重新摄入即可。

### Q: Milvus 数据膨胀
检查 Docker volumes: `docker system df -v | findstr milvus`。定期 `docker compose down -v` 重建（需重新摄入）。

---

## 6. 性能基准 (Demo 环境)

| 指标 | 数值 | 说明 |
|------|------|------|
| 摄入速度 | ~2-5 docs/秒 | FastEmbed CPU |
| 检索延迟 | ~16ms (254 docs) | Milvus + HNSW + BGE-small |
| 端到端延迟 | ~3-8s | 含 LLM 推理 |
| 内存占用 | ~2GB | Python + Milvus (etcd/minio/milvus 容器) |
| 向量维度 | 384 (BGE-small) | 默认模型 |

---

## 7. 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.2 | 2026-06-21 | 切换到 Milvus standalone (HNSW 索引, RRFRanker) |
