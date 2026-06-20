# OpsMind-RAG 详细设计 — 数据与基础设施层 (LLD-05)

**版本**: v1.0
**日期**: 2026-06-20
**对应 HLD**: HLD_OpsMind_RAG.md §2.1 数据与基础设施层, §3.1-3.5, §4.2 索引链路

---

## 1. 模块总览

| 子模块 | 职责 | 核心技术 |
|--------|------|---------|
| **MilvusStore** | 向量存储与混合检索 | Milvus 2.4+, HNSW, Sparse Vector |
| **Embedder** | 稠密/稀疏向量生成 | BGE-M3 (ONNX), 批量编码 |
| **Chunker** | 智能文档切分 | Markdown 结构感知, 64 token 重叠 |
| **Reranker** | Cross-Encoder 重排序 | BGE-Reranker-Large (ONNX) |
| **Connector** | 可插拔数据接入层 | BaseConnector 抽象, EnterpriseRAG-Bench 适配 |
| **RedisStore** | 状态/消息/缓存存储 | Redis Streams, AOF 持久化 |
| **SQLiteStore** | 元数据与审计日志存储 | SQLite + SQLAlchemy async |

---

## 2. 目录结构

```
opmind/
├── retrieval/
│   ├── __init__.py
│   ├── milvus_store.py        # Milvus 混合检索封装
│   ├── chunker.py             # 智能 Chunking
│   ├── reranker.py            # Cross-Encoder 重排序
│   └── embedder.py            # Embedding 模型路由
├── connectors/
│   ├── __init__.py
│   ├── base.py                # BaseConnector 抽象
│   ├── markdown_connector.py  # Markdown/JSON 解析
│   └── bench_connector.py     # EnterpriseRAG-Bench 专用
├── models/
│   ├── __init__.py
│   ├── document.py            # Document, Chunk, Citation
│   └── task.py                # Task, ExecutionPlan, Session
├── observability/
│   ├── __init__.py
│   ├── tracer.py              # OpenTelemetry 配置
│   ├── metrics.py             # Prometheus 指标
│   └── logger.py              # 结构化日志
└── tools/
    ├── __init__.py
    ├── base.py                # BaseTool 抽象
    ├── query_metrics.py       # 模拟 Prometheus 查询
    ├── query_logs.py          # 模拟 ELK 查询
    └── notify.py              # 模拟通知
```

---

## 3. MilvusStore（向量存储）

### 3.1 Collection Schema

```python
from pymilvus import (
    Collection, CollectionSchema, FieldSchema, DataType,
    connections, utility, AnnSearchRequest, WeightedRanker,
)

# Collection 名称
COLLECTION_NAME = "opsmind_chunks"

# Schema 设计
FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="context_prefix", dtype=DataType.VARCHAR, max_length=1024),

    # 稠密向量 (BGE-M3, dim=1024)
    FieldSchema(name="embedding_dense", dtype=DataType.FLOAT_VECTOR, dim=1024),

    # 稀疏向量 (BGE-M3 稀疏编码)
    FieldSchema(name="embedding_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),

    # 元数据（JSON 过滤）
    FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="section_path", dtype=DataType.JSON),  # ["H1", "H2", "H3"]
    FieldSchema(name="start_line", dtype=DataType.INT32),
    FieldSchema(name="end_line", dtype=DataType.INT32),
    FieldSchema(name="team", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="index_version", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="created_at", dtype=DataType.INT64),
]

SCHEMA = CollectionSchema(fields=FIELDS, description="OpsMind RAG Chunks")
```

### 3.2 索引配置

```python
# 稠密向量索引 (HNSW)
DENSE_INDEX_PARAMS = {
    "metric_type": "IP",          # Inner Product（需向量已归一化）
    "index_type": "HNSW",
    "params": {
        "M": 16,                  # 每个节点的最大连接数（高召回）
        "efConstruction": 200,    # 构建时的搜索宽度（高精度）
    },
}

# 稀疏向量索引 (SPARSE_INVERTED_INDEX)
SPARSE_INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type": "SPARSE_INVERTED_INDEX",
    "params": {
        "drop_ratio_build": 0.2,  # 构建时丢弃低频词比例
    },
}

# 标量索引（元数据过滤加速）
SCALAR_INDEX_PARAMS = [
    ("doc_id", {"index_type": "TRIE"}),
    ("team", {"index_type": "TRIE"}),
    ("doc_type", {"index_type": "TRIE"}),
]
```

### 3.3 MilvusStore 核心接口

```python
class MilvusStore:
    """Milvus 向量存储封装，支持混合检索"""

    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.collection: Collection | None = None

    # === 连接管理 ===
    async def connect(self):
        connections.connect("default", host=self.host, port=self.port)
        await self._ensure_collection()

    async def disconnect(self):
        connections.disconnect("default")

    async def _ensure_collection(self):
        """确保 collection 存在，不存在则创建"""
        if utility.has_collection(COLLECTION_NAME):
            self.collection = Collection(COLLECTION_NAME)
        else:
            self.collection = Collection(COLLECTION_NAME, schema=SCHEMA)
            await self._create_indexes()

    async def _create_indexes(self):
        """创建索引"""
        self.collection.create_index("embedding_dense", DENSE_INDEX_PARAMS)
        self.collection.create_index("embedding_sparse", SPARSE_INDEX_PARAMS)
        for field, params in SCALAR_INDEX_PARAMS:
            self.collection.create_index(field, params)
        self.collection.load()

    # === 写入 ===
    async def insert_chunks(self, chunks: list[Chunk]) -> list[int]:
        """批量插入 chunks"""
        data = [
            [c.chunk_id for c in chunks],
            [c.doc_id for c in chunks],
            [c.content for c in chunks],
            [c.context_prefix or "" for c in chunks],
            [c.embedding_dense for c in chunks],
            [c.embedding_sparse for c in chunks],
            [c.metadata.get("doc_title", "") for c in chunks],
            [c.metadata.get("source", "") for c in chunks],
            [c.metadata.get("source_type", "") for c in chunks],
            [c.section_path for c in chunks],
            [c.start_line for c in chunks],
            [c.end_line for c in chunks],
            [c.metadata.get("team", "") for c in chunks],
            [c.metadata.get("doc_type", "") for c in chunks],
            [c.index_version for c in chunks],
            [int(c.metadata.get("created_at", time.time())) for c in chunks],
        ]
        return self.collection.insert(data)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定文档的所有 chunks"""
        expr = f'doc_id == "{doc_id}"'
        return self.collection.delete(expr)

    # === 混合检索 ===
    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int = 10,
        filters: dict | None = None,
        rerank_strategy: str = "rrf",   # rrf | weighted
    ) -> list[SearchResult]:
        """
        Milvus Hybrid Search。
        同时执行稠密 + 稀疏检索，使用内部或外部融合。
        """
        # 构建过滤表达式
        expr = self._build_filter_expr(filters) if filters else None

        # 稠密检索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="embedding_dense",
            param={"metric_type": "IP", "params": {"ef": 128}},
            limit=top_k * 2,
            expr=expr,
        )

        # 稀疏检索请求
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="embedding_sparse",
            param={"metric_type": "IP"},
            limit=top_k * 2,
            expr=expr,
        )

        # 混合检索
        if rerank_strategy == "rrf":
            # 使用 Milvus 内置 RRF Ranker
            ranker = RRFRanker(k=60)
        else:
            # 使用加权 Mean 融合
            ranker = WeightedRanker(0.5, 0.5)

        results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=top_k,
            output_fields=[
                "chunk_id", "doc_id", "content", "doc_title",
                "source", "section_path", "start_line", "end_line",
            ],
        )

        return self._parse_results(results)

    # === 独立检索（不使用 Milvus 内置融合） ===
    async def dense_search(
        self,
        vectors: list[float],
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """纯稠密向量检索"""
        expr = self._build_filter_expr(filters) if filters else None
        results = self.collection.search(
            data=[vectors],
            anns_field="embedding_dense",
            param={"metric_type": "IP", "params": {"ef": 128}},
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "start_line", "end_line"],
        )
        return self._parse_results(results)

    async def sparse_search(
        self,
        vectors: dict[int, float],
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """纯稀疏向量检索"""
        expr = self._build_filter_expr(filters) if filters else None
        results = self.collection.search(
            data=[vectors],
            anns_field="embedding_sparse",
            param={"metric_type": "IP"},
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "start_line", "end_line"],
        )
        return self._parse_results(results)

    # === 辅助方法 ===
    @staticmethod
    def _build_filter_expr(filters: dict) -> str:
        """将 dict 过滤条件转换为 Milvus 表达式"""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            elif isinstance(value, list):
                vals = ", ".join(f'"{v}"' for v in value)
                conditions.append(f"{key} in [{vals}]")
        return " && ".join(conditions) if conditions else ""

    @staticmethod
    def _parse_results(results) -> list[SearchResult]:
        parsed = []
        for hits in results:
            for hit in hits:
                parsed.append(SearchResult(
                    chunk_id=hit.entity.get("chunk_id"),
                    doc_id=hit.entity.get("doc_id"),
                    content=hit.entity.get("content"),
                    doc_title=hit.entity.get("doc_title"),
                    score=hit.score,
                    start_line=hit.entity.get("start_line"),
                    end_line=hit.entity.get("end_line"),
                ))
        return parsed
```

### 3.4 混合检索策略对比

| 策略 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **Milvus Hybrid Search** | 内置 `hybrid_search()` + RRFRanker | 简单，服务端融合 | 黑盒，无法自定义 RRF 逻辑 |
| **自建 RRF Fusion** | 两次独立搜索 → 应用层 RRF | 完全可控，可加额外策略 | 两次网络往返，自建算法 |
| **混合模式（推荐）** | Demo 用自建 RRF，生产用 Milvus 内置 | 兼顾学习与性能 | 需要两套代码路径 |

---

## 4. Embedder（向量生成）

### 4.1 模型路由设计

```python
from enum import Enum
from abc import ABC, abstractmethod

class EmbedderBackend(Enum):
    BGE_M3_ONNX = "bge-m3-onnx"
    BGE_M3_TRANSFORMERS = "bge-m3-transformers"
    OPENAI = "openai"
    OLLAMA = "ollama"

class BaseEmbedder(ABC):
    @abstractmethod
    async def encode(self, texts: list[str]) -> list[list[float]]:
        """生成稠密向量, dim=1024"""
        ...

    @abstractmethod
    async def encode_sparse(
        self, texts: list[str]
    ) -> list[dict[int, float]]:
        """生成稀疏向量, {token_id: weight}"""
        ...


class EmbedderRouter:
    """
    多后端 Embedder 路由器。
    默认：BGE-M3 ONNX CPU。
    可切换：OpenAI / Ollama。
    """

    def __init__(self, config: Settings):
        self.backend = config.EMBEDDING_BACKEND
        self._embedder = self._build_embedder()

    def _build_embedder(self) -> BaseEmbedder:
        match self.backend:
            case EmbedderBackend.BGE_M3_ONNX:
                return BgeM3ONNXEmbedder(
                    model_path=settings.BGE_MODEL_PATH,
                    use_gpu=False,
                    batch_size=32,
                )
            case EmbedderBackend.BGE_M3_TRANSFORMERS:
                return BgeM3HFEmbedder(model_name="BAAI/bge-m3")
            case EmbedderBackend.OPENAI:
                return OpenAIEmbedder(
                    model="text-embedding-3-small",
                    api_key=settings.OPENAI_API_KEY,
                )
            case _:
                raise ValueError(f"Unsupported backend: {self.backend}")

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return await self._embedder.encode(texts)

    async def encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return await self._embedder.encode_sparse(texts)
```

### 4.2 BGE-M3 ONNX 实现

```python
class BgeM3ONNXEmbedder(BaseEmbedder):
    """
    使用 ONNX Runtime 运行 BGE-M3。
    支持稠密 + 稀疏双路输出。
    CPU 量化版本，适合无 GPU 部署。
    """

    def __init__(self, model_path: str, use_gpu: bool = False, batch_size: int = 32):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.replace(".onnx", ""))
        self.batch_size = batch_size

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """稠密向量: 取 [CLS] token 的 hidden state（归一化）"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="np", padding=True, truncation=True, max_length=512
            )
            outputs = self.session.run(["dense_embedding"], dict(inputs))
            # 归一化
            embeddings = outputs[0] / np.linalg.norm(outputs[0], axis=1, keepdims=True)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    async def encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """稀疏向量: BGE-M3 的 Lexical Weights 输出"""
        all_sparse = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="np", padding=True, max_length=512)
            outputs = self.session.run(["sparse_embedding"], dict(inputs))
            # 转换为 {token_id: weight} 字典
            for sparse in outputs[0]:
                token_weights = {
                    int(idx): float(weight)
                    for idx, weight in enumerate(sparse)
                    if weight > 0
                }
                all_sparse.append(token_weights)
        return all_sparse
```

---

## 5. Chunker（智能文档切分）

### 5.1 Markdown 结构感知切分

```python
from dataclasses import dataclass
import re

@dataclass
class ChunkBoundary:
    """Chunk 切分边界"""
    start_line: int
    end_line: int
    section_path: list[str]   # ["## 主从延迟", "### 排查步骤"]


class MarkdownStructureChunker:
    """
    Markdown 结构感知切分器。
    策略：
    1. 首先按 Markdown 标题层级（H1, H2, H3）切分
    2. 如果某个 Section 过长（> max_chunk_tokens），按段落二次切分
    3. 相邻 Chunk 重叠 64 tokens
    4. 保留 section_path 用于上下文增强
    """

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        min_chunk_tokens: int = 100,
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def chunk(self, document: Document) -> list[Chunk]:
        """
        对文档进行智能切分，返回 Chunk 列表。
        """
        lines = document.content.split("\n")

        # 1. 解析标题层级，构建 Section 树
        sections = self._parse_markdown_sections(lines)

        # 2. 对每个 Section 进行 Chunking
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(
                lines=lines,
                section=section,
                document=document,
            )
            chunks.extend(section_chunks)

        # 3. 过滤过短的 chunk（合并到前一个或后一个）
        chunks = self._merge_short_chunks(chunks)

        return chunks

    def _parse_markdown_sections(self, lines: list[str]) -> list[dict]:
        """
        解析 Markdown 标题结构。
        返回 Section 列表，每个 Section 含：
        - section_path: ["H1 Title", "H2 Subtitle", ...]
        - start_line, end_line
        """
        sections = []
        current_path = []
        current_start = 0
        current_level = 0

        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+)", line)
            if match:
                level = len(match.group(1))
                title = match.group(2)

                # 保存前一个 Section
                if current_start < i:
                    sections.append({
                        "section_path": list(current_path),
                        "start_line": current_start,
                        "end_line": i,
                    })

                # 更新路径
                if level <= current_level:
                    current_path = current_path[:level-1] + [title]
                else:
                    current_path.append(title)

                current_level = level
                current_start = i

        # 最后一个 Section
        if current_start < len(lines):
            sections.append({
                "section_path": list(current_path),
                "start_line": current_start,
                "end_line": len(lines),
            })

        return sections

    def _chunk_section(
        self,
        lines: list[str],
        section: dict,
        document: Document,
    ) -> list[Chunk]:
        """
        将单个 Section 切分为 token 大小合适的 Chunk。
        """
        section_text = "\n".join(lines[section["start_line"]:section["end_line"]])
        tokens = self._tokenize(section_text)

        if len(tokens) <= self.max_chunk_tokens:
            # 无需再切分
            return [self._create_chunk(
                content=section_text,
                section=section,
                document=document,
                start_line=section["start_line"],
                end_line=section["end_line"],
            )]

        # 需要切分：按段落 + 滑动窗口
        chunks = []
        current_start = 0

        while current_start < len(tokens):
            current_end = min(current_start + self.max_chunk_tokens, len(tokens))

            # 尝试在段落边界切断
            chunk_tokens = tokens[current_start:current_end]
            chunk_text = self._detokenize(chunk_tokens)

            # 映射回原始行号
            original_line_start = self._token_to_line(
                section["start_line"], lines, current_start
            )
            original_line_end = self._token_to_line(
                section["start_line"], lines, current_end
            )

            chunks.append(self._create_chunk(
                content=chunk_text,
                section=section,
                document=document,
                start_line=original_line_start,
                end_line=original_line_end,
            ))

            # 滑动窗口（含重叠）
            current_start = current_end - self.overlap_tokens

        return chunks

    def _create_chunk(
        self,
        content: str,
        section: dict,
        document: Document,
        start_line: int,
        end_line: int,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{document.doc_id}-chunk-{start_line}",
            doc_id=document.doc_id,
            content=content,
            context_prefix=None,  # 后续由 ContextualEmbedding 填充
            embedding_dense=[],
            embedding_sparse={},
            start_line=start_line,
            end_line=end_line,
            section_path=section["section_path"],
            metadata={**document.metadata, "doc_title": document.title},
            index_version="v1",
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简易 tokenizer（生产级用 tiktoken）"""
        # 按空格 + 标点切分，作为简易 token 计数
        return re.findall(r"\S+", text)

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        return " ".join(tokens)

    def _token_to_line(self, section_start: int, lines: list[str], token_pos: int) -> int:
        """将 token 位置映射回原始行号"""
        current_tokens = 0
        for i in range(section_start, len(lines)):
            current_tokens += len(self._tokenize(lines[i]))
            if current_tokens >= token_pos:
                return i
        return len(lines) - 1

    def _merge_short_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """合并过短的 chunk"""
        result = []
        for chunk in chunks:
            token_count = len(self._tokenize(chunk.content))
            if token_count < self.min_chunk_tokens and result:
                # 合并到前一个 chunk
                result[-1].content += "\n" + chunk.content
                result[-1].end_line = chunk.end_line
            else:
                result.append(chunk)
        return result
```

### 5.2 Contextual Embeddings 增强

```python
class ContextualEmbeddingEnhancer:
    """
    为每个 Chunk 生成上下文前缀，解决孤立 chunk 语义丢失问题。
    参考：Anthropic 的 Contextual Retrieval 方法。

    策略：仅对高价值文档类型启用（P0: runbook, incident_report），
    其他类型文档跳过增强以节省 LLM 调用成本。
    """

    P0_DOC_TYPES = {"runbook", "incident_report"}

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def enhance_batch(self, chunks: list[Chunk], document: Document) -> list[Chunk]:
        """
        为每个 chunk 生成上下文前缀。
        前缀描述该 chunk 在整个文档中的位置和作用。

        仅对 P0 文档类型启用：runbook, incident_report。
        其他类型跳过，context_prefix 保持为空。
        """
        doc_type = document.metadata.get("doc_type", "")
        if doc_type not in self.P0_DOC_TYPES:
            return chunks  # 跳过增强，直接返回

        prompt = CONTEXT_PREFIX_PROMPT.format(
            document_title=document.title,
            whole_document=document.content[:3000],  # 截断
        )

        chunk_texts = "\n\n---\n\n".join(
            f"[CHUNK {i+1}] Section: {' > '.join(c.section_path)}\n{c.content}"
            for i, c in enumerate(chunks)
        )

        response = await self.llm.complete(
            prompt=f"{prompt}\n\nChunks:\n{chunk_texts}\n\n"
                    "Generate context prefixes for each chunk.",
            system_prompt="You are a document structuring assistant.",
        )

        # 解析 LLM 返回的每个 chunk 的前缀
        prefixes = self._parse_prefixes(response.content, len(chunks))

        for chunk, prefix in zip(chunks, prefixes):
            chunk.context_prefix = prefix

        return chunks

    @staticmethod
    def _parse_prefixes(response: str, count: int) -> list[str]:
        """解析 LLM 返回的上下文前缀列表"""
        prefix_pattern = re.findall(r"\[CHUNK \d+\].*?\n(.*?)(?=\[CHUNK|\Z)", response, re.DOTALL)
        if len(prefix_pattern) >= count:
            return [p.strip()[:256] for p in prefix_pattern[:count]]
        return [""] * count


CONTEXT_PREFIX_PROMPT = """
You are structuring a document for semantic search. For each chunk below, write a short
context prefix that situates the chunk within the overall document. The prefix should
help an embedding model understand what the chunk is about in the broader context.

Document Title: {document_title}

Document (first part):
{whole_document}
"""
```

---

## 6. Connector（数据接入层）

### 6.1 BaseConnector 抽象

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable

class BaseConnector(ABC):
    """
    数据源连接器基类。
    所有数据源接入只需实现此接口。
    """

    connector_name: str
    supported_types: list[str]  # ["markdown", "json", "pdf", "confluence"]

    @abstractmethod
    async def extract(self, source: str) -> AsyncIterator[Document]:
        """
        从数据源提取文档。
        source: 数据源路径/URI。
        返回 AsyncIterator，支持流式处理大文件。
        """
        ...

    async def validate(self, source: str) -> bool:
        """
        验证数据源是否可访问、格式是否正确。
        默认返回 True，子类可覆盖。
        """
        return True

    async def watch(
        self,
        source: str,
        callback: Callable[[Document], Awaitable[None]],
    ) -> None:
        """
        监听数据源变更（CDC）。
        接口预留，Demo 阶段不实现。
        生产级可通过 Kafka/Webhook 实现。
        """
        raise NotImplementedError("CDC watching not implemented in Demo")

    async def can_handle(self, source_type: str) -> bool:
        """判断是否能处理该类型的数据源"""
        return source_type in self.supported_types
```

### 6.2 EnterpriseRAG-Bench Connector

```python
class EnterpriseRAGBenchConnector(BaseConnector):
    """
    EnterpriseRAG-Bench 数据集专用 Connector。
    支持处理 Bench 中的 Markdown、JSON 格式文档。
    """

    connector_name = "enterprise_rag_bench"
    supported_types = ["markdown", "json", "csv"]

    async def extract(self, source: str) -> AsyncIterator[Document]:
        """
        解析 EnterpriseRAG-Bench 数据集。
        source: 数据集目录路径。
        """
        bench_path = Path(source)

        # 遍历所有子目录（每个子目录对应一个数据源类别）
        for category_dir in bench_path.iterdir():
            if not category_dir.is_dir():
                continue

            for file_path in category_dir.rglob("*"):
                if file_path.suffix not in [".md", ".json"]:
                    continue

                try:
                    doc = await self._parse_file(file_path, category_dir.name)
                    if doc:
                        yield doc
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
                    continue

    async def _parse_file(self, file_path: Path, category: str) -> Document | None:
        """解析单个文件"""
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            return None

        # 元数据提取
        metadata = {
            "category": category,
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": file_path.stat().st_size,
        }

        # 尝试从 Front Matter 提取更多元数据
        front_matter = self._extract_front_matter(content)
        if front_matter:
            metadata.update(front_matter)
            content = self._strip_front_matter(content)

        title = metadata.get("title", file_path.stem)

        return Document(
            doc_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path))),
            source=f"bench://{category}/{file_path.name}",
            source_type=file_path.suffix.lstrip("."),
            title=title,
            content=content,
            metadata=metadata,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version="1.0",
            status="active",
        )

    @staticmethod
    def _extract_front_matter(content: str) -> dict | None:
        """提取 YAML Front Matter"""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                yaml_str = content[3:end]
                try:
                    import yaml
                    return yaml.safe_load(yaml_str)
                except Exception:
                    pass
        return None

    @staticmethod
    def _strip_front_matter(content: str) -> str:
        """去除 Front Matter"""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return content[end + 3:].strip()
        return content
```

---

## 7. 存储层 Schema

### 7.1 SQLite 表设计

```sql
-- Session 审计表
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'created',  -- created|running|interrupted|completed|failed|transferred
    context       TEXT,  -- JSON
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- 工具调用审计日志表
CREATE TABLE IF NOT EXISTS tool_audit_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id      TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    tool_params   TEXT,  -- JSON
    tool_result   TEXT,  -- JSON
    status        TEXT NOT NULL,  -- success|failure|timeout
    duration_ms   REAL,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_trace ON tool_audit_logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_session ON tool_audit_logs(session_id);

-- 中断事件表
CREATE TABLE IF NOT EXISTS interrupt_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    reason        TEXT NOT NULL,
    confidence    REAL,
    options       TEXT,  -- JSON list
    human_input   TEXT,
    resolution    TEXT,  -- continue|modify|transfer
    created_at    TEXT NOT NULL,
    resolved_at   TEXT
);

-- 知识库版本表
CREATE TABLE IF NOT EXISTS knowledge_versions (
    version_id    TEXT PRIMARY KEY,
    description   TEXT,
    doc_count     INTEGER,
    chunk_count   INTEGER,
    created_at    TEXT NOT NULL,
    is_active     INTEGER DEFAULT 0
);
```

### 7.2 Redis 数据结构

| Key Pattern | 类型 | 内容 | TTL |
|-------------|------|------|-----|
| `session:{id}` | String | Session JSON | 24h |
| `checkpoint:{id}:chain` | List | Checkpoint key 链 | 24h |
| `checkpoint:{id}:{stage}` | String | 阶段状态 JSON | 24h |
| `stream:agent:{name}` | Stream | Agent 消息流 | 无（maxlen=10000） |
| `stream:dlq` | Stream | 死信队列 | 无（maxlen=5000） |
| `processed:messages` | Set | 已处理消息 ID（幂等） | 1h |
| `lock:session:{id}` | String | 分布式锁 | 30s |
| `rate_limit:{user_id}` | String | 令牌桶状态 | 无 |
| `broadcast:{channel}` | PubSub | 广播频道 | N/A |

---

## 8. 可观测性

### 8.1 Prometheus 指标

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# 检索延迟
retrieval_latency = Histogram(
    "opsmind_retrieval_latency_seconds",
    "Retrieval latency by stage",
    ["stage"],  # dense, sparse, fusion, rerank
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Agent 迭代次数
agent_iterations = Counter(
    "opsmind_agent_iterations_total",
    "Total agent iterations",
    ["agent"],  # reason, retrieve, execute
)

# 工具执行
tool_execution = Counter(
    "opsmind_tool_execution_total",
    "Tool execution count",
    ["tool", "status"],  # success, failure, timeout
)

# 中断事件
interrupt_count = Counter(
    "opsmind_interrupt_total",
    "Interrupt events",
    ["reason"],  # low_confidence, tool_failure, manual
)

# 消息队列积压
message_bus_lag = Gauge(
    "opsmind_message_bus_lag_seconds",
    "Message bus lag by stream",
    ["stream"],
)

# Session 状态
active_sessions = Gauge(
    "opsmind_active_sessions",
    "Currently active sessions",
)

# LLM Token 用量
llm_token_usage = Counter(
    "opsmind_llm_token_usage_total",
    "LLM token usage",
    ["model", "type"],  # prompt, completion
)
```

### 8.2 结构化日志

```python
import logging
import json
from datetime import datetime, timezone
from opentelemetry import trace

class JSONFormatter(logging.Formatter):
    """结构化 JSON 日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # 注入 OpenTelemetry 上下文
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            log_entry["trace_id"] = format(ctx.trace_id, "032x")
            log_entry["span_id"] = format(ctx.span_id, "016x")

        # 注入额外字段
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # 注入异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str = "opsmind") -> logging.Logger:
    """获取结构化日志实例"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

---

## 9. 变更日志

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-06-20 | 初始版本 | AI-assisted Design |
