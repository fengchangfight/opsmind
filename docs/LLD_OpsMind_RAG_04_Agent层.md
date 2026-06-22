# OpsMind-RAG 详细设计 — Agent 层 (LLD-04)

**版本**: v1.0
**日期**: 2026-06-20
**对应 HLD**: HLD_OpsMind_RAG.md §2.1 Agent 层, §3.2-3.5

---

## 1. 模块总览

| Agent | 职责 | LangGraph 状态机节点数 | 关键能力 |
|-------|------|----------------------|---------|
| **RetrieveAgent** | 混合检索 + 重排序 | 7 个节点 | 稠密/稀疏双路召回, RRF 融合, Cross-Encoder 重排序, 查询扩展, 上下文构建 |
| **ReasonAgent** | 迭代推理 + 置信度评估 + 多轮上下文管理 | 7 个节点 + ContextOrchestrator | 假设生成, 证据评估, 多跳检索, 置信度计算, 中断恢复, 对话压缩, 长期记忆 |
| **ExecuteAgent** | 工具调用 + 熔断限流 | 4 个节点 | ToolRegistry, Token Bucket 限流, Circuit Breaker 熔断, 指数退避重试 |

---

## 2. BaseAgent 抽象

```python
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from opsmind.core.message_bus import MessageBus

class BaseAgent(ABC):
    """
    所有 Agent 的基类。
    每个 Agent 内部使用 LangGraph 定义状态机。
    Agent 间通过 MessageBus 通信。
    """

    name: str
    state_graph: StateGraph
    compiled_graph: CompiledGraph  # StateGraph.compile(checkpointer=...)
    message_bus: MessageBus

    def __init__(self, name: str, message_bus: MessageBus, checkpointer: CheckpointManager):
        self.name = name
        self.message_bus = message_bus
        self.checkpointer = checkpointer
        self.state_graph = StateGraph(self._get_state_type())
        self._build_graph()

    @abstractmethod
    def _get_state_type(self) -> type:
        """返回该 Agent 的状态类型（TypedDict）"""
        ...

    @abstractmethod
    def _build_graph(self):
        """构建 LangGraph 状态机（定义节点、边、条件边）"""
        ...

    async def run(
        self,
        initial_state: dict,
        config: RunnableConfig,
    ) -> dict:
        """执行 Agent 状态机，返回最终状态"""
        return await self.compiled_graph.ainvoke(initial_state, config)

    async def initialize(self):
        """启动时初始化（如预热模型、连接外部服务）"""
        self.compiled_graph = self.state_graph.compile(
            checkpointer=self.checkpointer.get_langgraph_checkpointer()
        )

    async def shutdown(self):
        """优雅关闭"""
        pass
```

---

## 3. RetrieveAgent（检索 Agent）

### 3.1 状态机图

```
         ┌─────────────┐
         │ parse_query │  解析用户查询，提取关键词、实体、意图
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │expand_query │  LLM 生成 3-5 个查询变体（Query Expansion）
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │embed_query  │  生成查询的稠密 + 稀疏向量（用于 Milvus Hybrid Search）
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │hybrid_search│  Milvus Hybrid Search（dense+sparse）+ 内置 RRFRanker 融合
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │   rerank    │  Cross-Encoder 粗排 Top-50 → 精排 Top-5
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │build_context│  动态截断 + Citation 标记 + 上下文拼接
         └─────────────┘
```

### 3.2 State 定义

```python
from typing import TypedDict, NotRequired

class RetrieveState(TypedDict):
    query: str                          # 原始查询
    filters: dict                       # 元数据过滤条件
    top_k: int                          # 最终返回数量
    expanded_queries: list[str]         # 扩展查询变体
    dense_query_vector: list[float]     # 查询稠密向量
    sparse_query_vector: dict           # 查询稀疏向量
    fused_results: list[ChunkResult]    # Milvus Hybrid Search + RRF 融合结果
    reranked_results: list[ChunkResult] # 重排序后结果
    final_context: str                  # 最终拼接的上下文
    citations: list[Citation]           # 溯源引用列表
    latency: dict[str, float]           # 各阶段延迟

class ChunkResult:
    chunk_id: str
    content: str
    score: float                        # 检索得分
    rerank_score: float                 # 重排序得分
    doc_id: str
    doc_id: str
    doc_title: str
    section_path: list[str]
    start_line: int
    end_line: int
```

### 3.3 核心节点实现

#### 3.3.1 EmbedQuery 节点

```python
async def embed_query_node(state: RetrieveState) -> dict:
    """生成查询的稠密向量和稀疏向量，供后续 Hybrid Search 使用"""
    t0 = time.monotonic()

    dense_vector = await embedder.encode([state["query"]])
    sparse_vector = await embedder.encode_sparse([state["query"]])

    state["latency"]["embed_query"] = time.monotonic() - t0
    return {
        "dense_query_vector": dense_vector[0],
        "sparse_query_vector": sparse_vector[0],
        "latency": state["latency"],
    }
```

#### 3.3.2 HybridSearch 节点

```python
async def hybrid_search_node(state: RetrieveState) -> dict:
    """
    Milvus Hybrid Search：稠密 + 稀疏向量混合检索，服务端 RRFRanker 融合。
    一次网络往返完成，性能最优。
    """
    t0 = time.monotonic()

    results = await milvus_store.hybrid_search(
        dense_vector=state["dense_query_vector"],
        sparse_vector=state["sparse_query_vector"],
        top_k=min(state["top_k"] * 3, 50),  # 粗排多取，留给 reranker
        filters=state["filters"],
        rerank_strategy="rrf",  # Milvus 内置 RRFRanker(k=60)
    )

    state["latency"]["hybrid_search"] = time.monotonic() - t0
    return {
        "fused_results": results,
        "latency": state["latency"],
    }
```

#### 3.3.3 RRF Fusion 节点

```python
async def rrf_fusion_node(state: RetrieveState) -> dict:
    """
    Reciprocal Rank Fusion (RRF) 融合。
    直接使用 Milvus 内置 hybrid_search() + RRFRanker，
    一次网络往返完成稠密+稀疏检索+融合，性能最优。
    """
    t0 = time.monotonic()

    dense_vector = state.get("dense_query_vector")
    sparse_vector = state.get("sparse_query_vector")

    # 使用 Milvus 内置融合（RRFRanker, k=60）
    results = await milvus_store.hybrid_search(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=state["top_k"],
        filters=state["filters"],
        rerank_strategy="rrf",  # Milvus 内置 RRFRanker(k=60)
    )

    state["latency"]["fusion"] = time.monotonic() - t0
    return {
        "fused_results": results,
        "latency": state["latency"],
    }
```

> **附录：RRF 算法原理**
>
> RRF 公式极简：`score(d) = Σ (1 / (k + rank_i(d)))`，其中 k=60。算法本身没有复杂逻辑，不"黑盒"。
> 使用 Milvus 内置 RRFRanker 的理由：
> 1. 服务端融合，一次网络往返，延迟更低
> 2. 无需自行维护两组搜索结果的排序映射
> 3. 与大厂实践一致（Anthropic, Cohere 均用服务端融合）
>
> 如需自建（教学用途），可参考以下代码：
> ```python
> # 仅供学习参考，生产不用
> def rrf_manual(dense: list, sparse: list, k=60) -> list:
>     scores = {}
>     for rank, r in enumerate(dense, 1):
>         scores[r.chunk_id] = 1.0 / (k + rank)
>     for rank, r in enumerate(sparse, 1):
>         scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank)
>     return sorted(scores, key=scores.get, reverse=True)
> ```

#### 3.3.3 Rerank 节点

```python
async def rerank_node(state: RetrieveState) -> dict:
    """
    Cross-Encoder 重排序。
    粗排 Top-50 → BGE-Reranker 精细打分 → Top-5。
    """
    t0 = time.monotonic()
    top_n = min(len(state["fused_results"]), 50)

    # 批量构造 (query, doc) 对
    pairs = [
        (state["query"], r.content)
        for r in state["fused_results"][:top_n]
    ]

    # BGE-Reranker-Large (ONNX) 批量打分
    scores = await reranker.compute_scores(pairs)

    # 按重排序分数排序
    for r, score in zip(state["fused_results"][:top_n], scores):
        r.rerank_score = score

    ranked = sorted(
        state["fused_results"][:top_n],
        key=lambda r: r.rerank_score,
        reverse=True,
    )

    state["latency"]["rerank"] = time.monotonic() - t0
    return {
        "reranked_results": ranked[:state["top_k"]],
        "latency": state["latency"],
    }
```

#### 3.3.4 BuildContext 节点

```python
async def build_context_node(state: RetrieveState) -> dict:
    """
    构建最终上下文：
    1. 动态截断（不超过 LLM context window）
    2. 生成 Citation 引用标记
    3. 拼接为 prompt 可用格式
    """
    max_tokens = estimate_context_window()  # 如 60% of model max
    context_parts = []
    citations = []

    token_count = 0
    for i, result in enumerate(state["reranked_results"]):
        segment = f"[{i+1}] Source: {result.doc_title}\n{result.content}\n"

        # 动态截断：超出限制后丢弃低分结果
        if token_count + estimate_tokens(segment) > max_tokens:
            logger.warning(f"Context truncated at {i}/{len(state['reranked_results'])} chunks")
            break

        context_parts.append(segment)
        token_count += estimate_tokens(segment)

        citations.append(Citation(
            citation_id=str(i + 1),
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            doc_title=result.doc_title,
            excerpt=result.content[:300],
            source_url=f"opmind://doc/{result.doc_id}#L{result.start_line}",
            relevance_score=result.rerank_score,
        ))

    return {
        "final_context": "\n---\n".join(context_parts),
        "citations": citations,
    }
```

### 3.4 Query Expansion 节点

```python
async def expand_query_node(state: RetrieveState) -> dict:
    """
    使用 LLM 生成查询变体，提升召回率。
    """
    prompt = f"""Generate 3-5 alternative search queries for the following user question.
Vary keywords and perspectives to improve retrieval recall.

Original: "{state['query']}"

Return JSON: {{"variants": ["variant1", "variant2", ...]}}"""

    response = await llm.complete(prompt, response_format={"type": "json_object"})
    variants = json.loads(response.content)["variants"]

    return {"expanded_queries": [state["query"]] + variants}
```

### 3.5 LangGraph 构建

```python
def _build_graph(self):
    self.state_graph.add_node("parse_query", parse_query_node)
    self.state_graph.add_node("expand_query", expand_query_node)
    self.state_graph.add_node("embed_query", embed_query_node)
    self.state_graph.add_node("hybrid_search", hybrid_search_node)
    self.state_graph.add_node("rerank", rerank_node)
    self.state_graph.add_node("build_context", build_context_node)

    self.state_graph.set_entry_point("parse_query")
    self.state_graph.add_edge("parse_query", "expand_query")
    self.state_graph.add_edge("expand_query", "embed_query")
    self.state_graph.add_edge("embed_query", "hybrid_search")
    self.state_graph.add_edge("hybrid_search", "rerank")
    self.state_graph.add_edge("rerank", "build_context")
    self.state_graph.set_finish_point("build_context")
```

---

## 4. ReasonAgent（推理 Agent）

### 4.1 状态机图

```
         ┌─────────────┐
         │ analyze_query│  理解查询意图，生成初始推理方向
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │generate_hypo │  基于检索上下文生成 1-3 个假设
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │eval_evidence │  评估每个假设的证据充分性
         └──────┬──────┘
                │
        ┌───────▼────────┐
        │check_confidence │  计算置信度
        └───────┬────────┘
                │
         ┌──────▼──────┐     ┌──────────────┐
  ──YES──│  interrupt   │     │extract_gaps  │  置信度 < 0.7 且还有迭代余量
         └──────────────┘     └──────┬───────┘
                                    │
                              ┌─────▼──────┐
                              │ re_retrieve│  发起新检索 → 回到 eval_evidence
                              └────────────┘
                │
         ┌──────▼──────┐
         │gen_final_ans│  置信度 >= 0.7 → 生成最终答案
         └─────────────┘
```

### 4.2 State 定义

```python
class ReasonState(TypedDict):
    query: str
    context: str                       # RetrieveAgent 提供的检索上下文
    citations: list[Citation]
    hypotheses: list[Hypothesis]       # 生成的假设列表
    evidence: dict[str, dict]          # hypothesis_id → evidence items
    knowledge_gaps: list[str]          # 当前知识缺口
    iteration: int                     # 当前迭代轮次
    max_iterations: int                # 最大迭代次数（默认 3）
    confidence: ConfidenceScore        # 置信度
    final_answer: str | None           # 最终答案
    status: str                        # running | interrupted | completed

class Hypothesis:
    hypothesis_id: str
    statement: str                     # 假设陈述
    confidence: float                  # 该假设的置信度
    supporting_evidence: list[str]     # 支持证据（chunk_id 列表）
    counter_evidence: list[str]        # 反对证据

class ConfidenceScore:
    score: float                       # 0-1 总分
    coverage: float                    # 证据覆盖度（0-1）
    consistency: float                 # 来源一致性（0-1）
    freshness: float                   # 信息新鲜度（0-1）
```

### 4.3 置信度计算

```python
def compute_confidence(
    hypotheses: list[Hypothesis],
    evidence: dict[str, dict],
    citations: list[Citation],
) -> ConfidenceScore:
    """
    置信度 = weighted(c_coverage, c_consistency, c_freshness)

    coverage:   假设陈述中多少关键点有直接证据支持
    consistency: 不同来源的证据是否一致（无矛盾）
    freshness:   证据文档的最后更新时间（越新越好）
    """
    # 覆盖度：有证据支持的假设数量 / 总假设数量
    supported = sum(
        1 for h in hypotheses
        if len(h.supporting_evidence) > len(h.counter_evidence)
    )
    coverage = supported / max(len(hypotheses), 1)

    # 一致性：互斥假设对数 / 总假设对数
    if len(hypotheses) <= 1:
        consistency = 1.0
    else:
        conflicting = 0
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                if _are_conflicting(hypotheses[i], hypotheses[j]):
                    conflicting += 1
        total_pairs = len(hypotheses) * (len(hypotheses) - 1) / 2
        consistency = 1 - (conflicting / total_pairs)

    # 新鲜度：证据文档的平均新鲜度
    avg_recency = _compute_avg_recency(citations)

    # 加权综合（权重可按场景调整）
    score = (coverage * 0.5) + (consistency * 0.3) + (avg_recency * 0.2)

    return ConfidenceScore(
        score=round(score, 3),
        coverage=round(coverage, 3),
        consistency=round(consistency, 3),
        freshness=round(avg_recency, 3),
    )
```

### 4.4 中断条件

```python
def check_confidence_node(state: ReasonState) -> str:
    """
    条件边：根据置信度决定下一步。
    返回节点名或 LangGraph END。
    """
    confidence = state["confidence"].score
    iterations_left = state["iteration"] < state["max_iterations"]

    if confidence >= 0.7:
        return "gen_final_ans"
    elif iterations_left:
        return "extract_gaps"          # 还有余量，继续深挖
    else:
        return "interrupt"             # 需要人工介入

def interrupt_node(state: ReasonState) -> dict:
    """
    触发 LangGraph interrupt 暂停。
    保存 checkpoint，等待人工输入后恢复。
    """
    return {
        "status": "interrupted",
        "interrupt_data": {
            "reason": f"置信度不足 ({state['confidence'].score:.2f})，"
                      f"已尝试 {state['iteration']} 轮推理",
            "confidence": state["confidence"].score,
            "options": ["continue", "modify", "transfer"],
            "knowledge_gaps": state.get("knowledge_gaps", []),
        },
    }
```

### 4.5 知识缺口提取

```python
async def extract_gaps_node(state: ReasonState) -> dict:
    """
    分析当前证据缺口，生成新的检索关键词。
    这是迭代推理的核心：证据不足 → 找缺口 → 重新检索。
    """
    prompt = f"""You are analyzing an ops troubleshooting task.
Current hypotheses and evidence:
{json.dumps([h.model_dump() for h in state["hypotheses"]], ensure_ascii=False)}

Identify specific knowledge gaps and generate focused search queries to fill them.
Return JSON: {{"gaps": ["gap1", ...], "new_queries": ["query1", ...]}}
"""
    response = await llm.complete(prompt, response_format={"type": "json_object"})
    result = json.loads(response.content)

    return {
        "knowledge_gaps": result["gaps"],
        "new_queries": result["new_queries"],
        "iteration": state["iteration"] + 1,
    }
```

### 4.6 LangGraph 构建

```python
def _build_graph(self):
    g = self.state_graph

    g.add_node("analyze_query", analyze_query_node)
    g.add_node("generate_hypo", generate_hypo_node)
    g.add_node("eval_evidence", eval_evidence_node)
    g.add_node("check_confidence", check_confidence_node)
    g.add_node("interrupt", interrupt_node)
    g.add_node("extract_gaps", extract_gaps_node)
    g.add_node("re_retrieve", re_retrieve_node)
    g.add_node("gen_final_ans", gen_final_ans_node)

    g.set_entry_point("analyze_query")
    g.add_edge("analyze_query", "generate_hypo")
    g.add_edge("generate_hypo", "eval_evidence")
    g.add_edge("eval_evidence", "check_confidence")

    # 条件边：根据置信度选择路径
    g.add_conditional_edges(
        "check_confidence",
        route_by_confidence,   # 返回 "gen_final_ans" | "extract_gaps" | "interrupt"
        {
            "gen_final_ans": "gen_final_ans",
            "extract_gaps": "extract_gaps",
            "interrupt": "interrupt",
        },
    )

    g.add_edge("extract_gaps", "re_retrieve")
    g.add_edge("re_retrieve", "eval_evidence")  # 循环回到证据评估

    g.set_finish_point("gen_final_ans")
    g.set_finish_point("interrupt")
```

### 4.7 多轮对话上下文

ReasonAgent 接收可选的 `history` 参数（对话历史消息列表），注入 LLM 请求中实现多轮对话感知：

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if history:
    messages.extend(history[-10:])  # 最近 10 轮
messages.append({"role": "user", "content": user_message})
```

前端通过 API 的 `history` 参数（Base64 JSON）传递最近 10 轮对话，LLM 可引用前轮内容。

> **注：上述为 Demo 最小实现。生产级多轮上下文管理见第 6 节。**

---

## 6. 多轮对话上下文管理（生产级设计）

**参考来源**: Claude Code (`src/services/compact/`), Hermes (`agent/context_compressor.py`), OpenCode (`packages/core/src/session/compaction.ts`)

三方系统的共同核心模式：

| 模式 | Claude Code | Hermes | OpenCode |
|------|-----------|--------|---------|
| **Head/Tail 分割** | api-round grouping + compact boundary | 前 N 条保护 + 尾部 token 预算 | turn-based select + checkpoint message |
| **Token 预算驱动** | 按 context window 比例计算阈值 | `threshold_tokens` + `tail_token_budget` | `preserve_recent_tokens` (~25%) |
| **结构化摘要** | 9-section summary + scratchpad | Historical Task Snapshot + Pending Asks | Goal/Progress/Decisions/Next Steps |
| **迭代更新** | 重新生成摘要 | 前一次摘要注入 prompt 增量更新 | `update previous summary` |
| **摘要防护** | compact_boundary marker | SUMMARY_PREFIX + `_compressed_summary` tag | `<conversation-checkpoint>` XML |
| **工具输出裁剪** | microCompact (缓存编辑) | 三遍裁剪 (去重+摘要+截断) | 背景 prune (40K 保护) |
| **防抖/反抖** | circuit breaker (3 次失败) | 2 次连续 <10% 节省则停 | 与防抖同逻辑 |
| **多路径触发** | autoCompact + reactive + manual | preflight + post-turn + error-driven | compactIfNeeded + compactAfterOverflow |
| **长时记忆** | SESSION_MEMORY.md 文件 | MEMORY.md + memory_provider 插件 | 无独立系统 (依赖 checkpoint) |

---

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ContextOrchestrator                       │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │TokenBudget   │  │Conversation-     │  │SessionMemory   │  │
│  │(token分配)   │  │Compactor         │  │(长期记忆)      │  │
│  │              │  │(对话压缩)        │  │               │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────┬───────┘  │
│         │                   │                    │          │
│  ┌──────▼───────────────────▼────────────────────▼───────┐  │
│  │              CompactionTrigger                         │  │
│  │  preflight (每轮前) │ post-turn (每轮后) │ manual │    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

### 6.2 TokenBudget — 上下文 Token 分配器

**职责**: 在每次 LLM 调用前，根据上下文窗口大小，将 token 预算分配到四个区域：

```
Model Context Window (e.g. 128K)
├── System Prompt + Long-term Memory:  ~5K  (固定)
├── Retrieved Documents (RAG):         ~30% (动态)
├── Conversation History:              ~40% (动态)
│   ├── Protected Tail (最近轮次):     ~25%
│   └── Compaction Summary (历史):     ~15%
├── Output Buffer:                     ~20K (保留)
└── Safety Margin:                     ~5K  (保留)
```

```python
from dataclasses import dataclass, field

@dataclass
class TokenBudget:
    context_window: int = 128_000         # 模型上下文窗口
    output_buffer: int = 20_000           # 输出保留空间
    safety_margin: int = 5_000            # 安全余量
    system_tokens: int = 5_000            # System prompt + 记忆

    @property
    def usable(self) -> int:
        """可供历史消息和检索文档使用的 token 总量"""
        return self.context_window - self.output_buffer - self.safety_margin

    @property
    def history_budget(self) -> int:
        """对话历史 token 预算 (~40% of usable)"""
        return int(self.usable * 0.45)

    @property
    def retrieval_budget(self) -> int:
        """检索文档 token 预算 (~30% of usable)"""
        return int(self.usable * 0.30)

    @property
    def tail_budget(self) -> int:
        """尾部保护 token 预算 (~60% of history)"""
        return int(self.history_budget * 0.60)

    @property
    def summary_budget(self) -> int:
        """压缩摘要 token 预算 (~40% of history)"""
        return self.history_budget - self.tail_budget

    @property
    def compaction_threshold(self) -> int:
        """触发压缩的 token 阈值"""
        # 当 (system + history) 超过 usable * 70% 时触发
        return int(self.usable * 0.70)

    def allocation_report(self, used: dict[str, int]) -> str:
        """生成分配报告"""
        return (
            f"Context: {sum(used.values())}/{self.usable} used "
            f"(system={used.get('system',0)}, "
            f"history={used.get('history',0)}, "
            f"retrieval={used.get('retrieval',0)})"
        )
```

---

### 6.3 ConversationCompactor — 对话压缩器

**职责**: 当对话历史超过阈值时，将中间部分压缩为结构化摘要，仅保留头部（system prompt）和尾部（最近轮次）原文。

**核心算法 — Head/Tail/Summary 分割**:

```
完整对话历史:
[SYS] [U1] [A1] [U2] [A2] [U3] [A3] [U4] [A4] [U5] [A5] [U6] [A6]
 ──────────────────────Head──────────────── ────Tail───
 │← 保护头部 (系统 + 首轮)                │← 预算保护   │
                                           │  (最近 N 轮)│
                                        
压缩后上下文:
[SYS] [CompactionSummary] [U5] [A5] [U6] [A6]
       ↑
       结构化摘要 (取代 U1~U4)
```

```python
from typing import Optional

@dataclass
class CompactionResult:
    """压缩结果"""
    summary: str                         # LLM 生成的结构化摘要
    head_messages: list[dict]            # 保留的头部消息
    tail_messages: list[dict]            # 保留的尾部消息
    pre_tokens: int                      # 压缩前总 token 数
    post_tokens: int                     # 压缩后总 token 数
    savings_pct: float                   # 节省比例
    reason: str                          # 触发原因 (auto/manual/overflow)
    timestamp: str                       # ISO 8601


class ConversationCompactor:
    """
    对话历史压缩器。
    参考: Hermes ContextCompressor + OpenCode SessionCompaction。
    """

    # 结构化摘要模板 (参考 Claude Code 9-section + OpenCode 锚定模式)
    SUMMARY_TEMPLATE = """You are summarizing an SRE/DevOps troubleshooting session.
Create a structured summary of the conversation history below.
The summary will replace old messages in the context window.

<conversation-to-summarize>
{serialized_history}
</conversation-to-summarize>

{focus_instruction}

Generate a summary in the following structured format:

<summary>
## Primary Topics
(List the main troubleshooting/investigation topics discussed)

## Key Findings
(Important facts, configurations, or insights discovered)

## Decisions Made
(Any decisions, tradeoffs, or choices made during the conversation)

## Files/Resources Referenced
(Relevant documents, tools, runbooks, or citations mentioned)

## Unresolved Questions
(Questions that were asked but not yet answered)

## Current State
(What the user was working on when compaction was triggered)

## User Messages Summary
(Brief summary of each user message in order)
</summary>"""

    def __init__(
        self,
        llm: "LLMClient",
        budget: TokenBudget,
    ):
        self.llm = llm
        self.budget = budget
        self._last_compaction: Optional[CompactionResult] = None
        self._consecutive_ineffective: int = 0
        self._effectiveness_threshold: float = 0.10  # 10% 节省阈值

    def should_compact(self, total_tokens: int) -> bool:
        """判断是否需要压缩"""
        if total_tokens < self.budget.compaction_threshold:
            return False
        if self._consecutive_ineffective >= 2:
            return False  # 反抖：连续两次低效压缩则暂停
        return True

    def _select_head_tail(
        self, messages: list[dict],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        将消息分为 head (保护), middle (压缩), tail (保护)。
        
        策略:
        1. Head: system prompt + 首轮对话 (永远保护)
        2. Tail: 从后向前累计直到达到 tail_budget
        3. 保证不拆分 tool_use/tool_result 对
        """
        head = []
        tail = []
        middle = []
        
        # 寻找 system + 第一轮 user 消息
        head_end = 0
        found_first_user = False
        for i, m in enumerate(messages):
            if m["role"] == "system":
                head.append(m)
                head_end = i + 1
            elif m["role"] == "user" and not found_first_user:
                head.append(m)
                head_end = i + 1
                found_first_user = True
            elif found_first_user:
                break
            else:
                head.append(m)
                head_end = i + 1

        # 从后向前累计 tail
        remaining = messages[head_end:]
        tail_tokens = 0
        tail_start = len(remaining)
        for i in range(len(remaining) - 1, -1, -1):
            msg_tokens = self._estimate_tokens(remaining[i])
            if tail_tokens + msg_tokens > self.budget.tail_budget:
                tail_start = i + 1
                break
            tail_tokens += msg_tokens
            tail_start = i

        # 调整 tail_start 以保证不拆分 tool pair
        tail_start = self._align_to_message_boundary(remaining, tail_start)

        tail = remaining[tail_start:]
        middle = remaining[:tail_start]

        return head, tail, middle

    async def compact(
        self,
        messages: list[dict],
        focus_topic: str | None = None,
        reason: str = "auto",
    ) -> CompactionResult:
        """
        执行对话压缩。
        
        Args:
            messages: 完整对话历史
            focus_topic: 可选焦点主题
            reason: 触发原因 (auto/manual/overflow)
        
        Returns:
            CompactionResult 含摘要和压缩前后统计
        """
        head, tail, middle = self._select_head_tail(messages)

        if len(middle) <= 1:
            # 中间部分太短，无需压缩
            return CompactionResult(
                summary="",
                head_messages=head,
                tail_messages=tail,
                pre_tokens=sum(self._estimate_tokens(m) for m in messages),
                post_tokens=sum(self._estimate_tokens(m) for m in head + tail),
                savings_pct=0,
                reason=reason,
                timestamp="",
            )

        # 序列化中间消息
        serialized = self._serialize_messages(middle)

        # 构建 prompt
        focus = (
            f"Focus on context related to: {focus_topic}"
            if focus_topic else ""
        )
        prompt = self.SUMMARY_TEMPLATE.format(
            serialized_history=serialized,
            focus_instruction=focus,
        )

        # 如果有前一次摘要，注入为增量更新
        if self._last_compaction and self._last_compaction.summary:
            prompt += (
                f"\n\n<previous-summary>\n{self._last_compaction.summary}\n"
                f"</previous-summary>\n"
                f"Update this summary by preserving still-relevant details, "
                f"removing stale details, and merging in new facts."
            )

        # 调用 LLM 生成摘要
        response = await self.llm.complete(
            system_prompt="You are a context summarization assistant.",
            prompt=prompt,
            max_tokens=min(self.budget.summary_budget // 4, 4000),
            temperature=0.1,
        )
        summary = response.content

        # 组装压缩后消息
        compacted_messages = head + [
            {"role": "assistant", "content": summary,
             "_meta": {"compaction_summary": True, "reason": reason}}
        ] + tail

        pre_tokens = sum(self._estimate_tokens(m) for m in messages)
        post_tokens = sum(self._estimate_tokens(m) for m in compacted_messages)
        savings_pct = (pre_tokens - post_tokens) / max(pre_tokens, 1)

        # 反抖跟踪
        if savings_pct < self._effectiveness_threshold:
            self._consecutive_ineffective += 1
        else:
            self._consecutive_ineffective = 0

        result = CompactionResult(
            summary=summary,
            head_messages=head,
            tail_messages=tail,
            pre_tokens=pre_tokens,
            post_tokens=post_tokens,
            savings_pct=savings_pct,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._last_compaction = result
        return result

    # --- 辅助方法 ---

    def _serialize_messages(self, messages: list[dict]) -> str:
        """将消息列表序列化为纯文本"""
        parts = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            # 截断过长内容
            if len(content) > 2000:
                content = content[:2000] + "...[truncated]"
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _estimate_tokens(msg: dict) -> int:
        """粗略 token 估算: 4 char/token (参考 OpenCode)"""
        return len(str(msg.get("content", ""))) // 4

    @staticmethod
    def _align_to_message_boundary(
        messages: list[dict], tail_start: int,
    ) -> int:
        """确保不在 tool_use / tool_result 中间切分"""
        while tail_start > 0 and tail_start < len(messages):
            prev_msg = messages[tail_start - 1]
            if prev_msg.get("_tool_call_id") and messages[tail_start].get("role") == "tool":
                tail_start -= 1
            else:
                break
        return max(tail_start, 0)
```

---

### 6.4 SessionMemory — 长期记忆

**职责**: 跨 Session 持久化关键信息（用户偏好、项目上下文、常见故障模式），在每次新 Session 启动时注入 system prompt。

**参考**: Claude Code SESSION_MEMORY.md + Hermes MEMORY.md / memory_provider。

```python
import json
from pathlib import Path

class SessionMemory:
    """
    长期对话记忆，跨 Session 持久化。
    
    存储格式: Markdown 文件，按 Section 分区。
    位于: <data_dir>/memory/
    """

    MEMORY_FILE = "session_memory.md"
    USER_PREFERENCES_FILE = "user_preferences.md"

    def __init__(self, data_dir: str = "./data/memory"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --- 读 ---

    def get_context(self) -> str:
        """获取当前记忆上下文，用于注入 system prompt"""
        parts = []
        
        memory_path = self.data_dir / self.MEMORY_FILE
        if memory_path.exists():
            content = memory_path.read_text(encoding="utf-8")
            if content.strip():
                parts.append(
                    f"<session-memory>\n{content}\n</session-memory>\n\n"
                    "[System note: The above is recalled memory context from "
                    "previous sessions. Treat as authoritative reference.]"
                )

        user_path = self.data_dir / self.USER_PREFERENCES_FILE
        if user_path.exists():
            content = user_path.read_text(encoding="utf-8")
            if content.strip():
                parts.append(
                    f"<user-preferences>\n{content}\n</user-preferences>\n\n"
                    "[System note: The above are user preferences learned "
                    "from interactions. Follow unless contradicted.]"
                )

        return "\n".join(parts)

    # --- 写 ---

    async def extract_and_save(
        self,
        messages: list[dict],
        llm: "LLMClient",
    ):
        """
        从对话历史中提取关键信息并持久化。
        
        在以下时机调用:
        - Session 结束时
        - 上下文压缩前 (防止旧消息被丢弃)
        - 用户手动触发 /remember
        """
        # 序列化对话
        serialized = "\n\n".join(
            f"[{m['role']}]: {m.get('content', '')[:1000]}"
            for m in messages[-20:]  # 只提取最近 20 条
        )
        
        prompt = f"""Review this SRE/DevOps conversation and extract important information 
to persist for future sessions. Focus on:

1. User's technical environment (tools, stack, infra)
2. Recurring problems or patterns
3. Key decisions or preferences expressed
4. Unresolved issues to follow up on

Conversation:
{serialized}

Return a JSON with:
{{
  "memory_updates": "Markdown notes to append to session memory",
  "preference_updates": "Markdown notes to append to user preferences"
}}

Only include genuinely new/important information. Skip trivial details."""
        
        response = await llm.complete(
            system_prompt="You are a knowledge extraction assistant.",
            prompt=prompt,
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.1,
        )
        
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            return

        # 追加到记忆文件 (带时间戳)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if data.get("memory_updates"):
            memory_path = self.data_dir / self.MEMORY_FILE
            content = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
            if content:
                content += "\n"
            content += f"### {now}\n{data['memory_updates']}\n"
            # 限制文件大小 (~8000 chars)
            if len(content) > 8000:
                content = content[-8000:]
            memory_path.write_text(content, encoding="utf-8")

        if data.get("preference_updates"):
            pref_path = self.data_dir / self.USER_PREFERENCES_FILE
            content = pref_path.read_text(encoding="utf-8") if pref_path.exists() else ""
            if content:
                content += "\n"
            content += f"### {now}\n{data['preference_updates']}\n"
            if len(content) > 4000:
                content = content[-4000:]
            pref_path.write_text(content, encoding="utf-8")

    # --- 生命周期回调 ---

    async def on_session_start(self) -> str:
        """Session 开始时加载记忆"""
        return self.get_context()

    async def on_session_end(self, messages: list[dict], llm: "LLMClient"):
        """Session 结束时提取并保存记忆"""
        # 只在会话有意义时保存 (至少 3 轮对话)
        user_messages = [m for m in messages if m.get("role") == "user"]
        if len(user_messages) >= 3:
            await self.extract_and_save(messages, llm)

    async def on_pre_compact(self, messages: list[dict], llm: "LLMClient"):
        """压缩前提取记忆（避免旧消息被丢弃后无法提取）"""
        if self._last_compaction and self._last_compaction.savings_pct > 0.3:
            await self.extract_and_save(messages, llm)
```

---

### 6.5 CompactionTrigger — 多路径触发

```python
from enum import Enum

class TriggerReason(Enum):
    PREFLIGHT = "preflight"      # 每轮 LLM 调用前检查
    POST_TURN = "post_turn"      # 每轮 LLM 响应后检查
    OVERFLOW = "overflow"        # 上下文溢出错误
    MANUAL = "manual"            # 用户手动 /compact
    SESSION_START = "session_start"  # 加载历史 session 时


class CompactionTrigger:
    """
    多路径压缩触发器。
    参考: Claude Code autoCompact + reactiveCompact,
          Hermes preflight + post-turn + error-driven,
          OpenCode compactIfNeeded + compactAfterOverflow。
    """

    def __init__(
        self,
        compactor: ConversationCompactor,
        memory: SessionMemory,
        budget: TokenBudget,
    ):
        self.compactor = compactor
        self.memory = memory
        self.budget = budget
        self._consecutive_overflows = 0
        self._max_consecutive_overflows = 3

    async def check_preflight(
        self, messages: list[dict], llm: "LLMClient",
    ) -> list[dict]:
        """
        每轮 LLM 调用前检查 (preflight)。
        
        如果 token 超阈值，在发 API 请求前主动压缩，
        避免 provider 端拒绝请求。
        """
        total_tokens = sum(
            self.compactor._estimate_tokens(m) for m in messages
        )
        if not self.compactor.should_compact(total_tokens):
            return messages

        result = await self.compactor.compact(messages, reason="preflight")
        return result.head_messages + [
            {"role": "assistant", "content": result.summary,
             "_meta": {"compaction_summary": True}}
        ] + result.tail_messages

    async def check_post_turn(
        self, messages: list[dict], llm: "LLMClient",
    ):
        """
        每轮响应后检查 (post-turn)。
        
        在以下时机:
        - 后台异步执行，不阻塞用户
        - 触发 SessionMemory.extract_and_save()
        """
        total_tokens = sum(
            self.compactor._estimate_tokens(m) for m in messages
        )
        if self.compactor.should_compact(total_tokens):
            await self.memory.on_pre_compact(messages, llm)

    async def check_overflow(
        self, messages: list[dict], llm: "LLMClient",
    ) -> list[dict]:
        """
        上下文溢出时响应 (reactive)。
        
        当 provider 返回 context-overflow 错误时调用。
        比正常压缩更激进：tail_budget 减半。
        """
        self._consecutive_overflows += 1
        
        # 连续溢出: 可能是摘要本身太大
        if self._consecutive_overflows >= self._max_consecutive_overflows:
            # 激进模式: 丢弃中间所有消息，只保留 head + 最后 2 轮
            head, tail, _ = self.compactor._select_head_tail(messages)
            # 覆盖 tail_budget 为极小值
            self.budget.tail_budget = self.budget.tail_budget // 4
            result = await self.compactor.compact(
                head + tail, reason="overflow",
            )
            self._consecutive_overflows = 0
            return result.head_messages + result.tail_messages

        result = await self.compactor.compact(messages, reason="overflow")
        return result.head_messages + [
            {"role": "assistant", "content": result.summary,
             "_meta": {"compaction_summary": True}}
        ] + result.tail_messages

    async def compact_manual(
        self, messages: list[dict], focus: str | None = None,
    ) -> list[dict]:
        """用户手动触发 /compact"""
        result = await self.compactor.compact(
            messages, focus_topic=focus, reason="manual",
        )
        return result.head_messages + [
            {"role": "assistant", "content": result.summary,
             "_meta": {"compaction_summary": True}}
        ] + result.tail_messages

    async def load_session(
        self,
        messages: list[dict],
        memory_context: str,
    ) -> list[dict]:
        """
        加载历史 Session 时重构上下文。
        
        1. 如果有前次 CompactionResult，使用摘要 + 尾部
        2. 注入长期记忆到 system prompt
        3. 如果历史仍然过长，触发压缩
        """
        # 注入长期记忆
        if memory_context and messages:
            if messages[0]["role"] == "system":
                messages[0]["content"] = (
                    memory_context + "\n\n" + messages[0]["content"]
                )
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": memory_context,
                })

        # 如果历史过长，主动压缩
        return await self.check_preflight(messages, None)
```

---

### 6.6 ContextOrchestrator — 上下文编排器

```python
class ContextOrchestrator:
    """
    上下文编排器: 将所有组件串联为完整的请求构建流程。
    
    每次 API 调用前执行:
    1. 注入长期记忆 (SessionMemory)
    2. 合并对话历史 (ConversationCompactor)
    3. 分配 token 预算 (TokenBudget)
    4. 构建最终 messages 数组
    """

    def __init__(
        self,
        budget: TokenBudget,
        compactor: ConversationCompactor,
        memory: SessionMemory,
        trigger: CompactionTrigger,
    ):
        self.budget = budget
        self.compactor = compactor
        self.memory = memory
        self.trigger = trigger

    async def build_context(
        self,
        messages: list[dict],
        retrieved_docs: list[str],
        llm: "LLMClient",
    ) -> list[dict]:
        """
        构建完整的 LLM 上下文消息数组。
        
        Args:
            messages: 完整对话历史
            retrieved_docs: RAG 检索到的文档片段
            llm: LLM 客户端
        
        Returns:
            可直接传入 chat.completions.create() 的 messages 列表
        """
        # Step 1: Preflight 压缩
        messages = await self.trigger.check_preflight(messages, llm)

        # Step 2: 分配检索文档 token
        retrieval_budget = self.budget.retrieval_budget
        docs_text = ""
        for doc in retrieved_docs:
            doc_tokens = len(doc) // 4
            if retrieval_budget - doc_tokens < 0:
                break
            docs_text += doc + "\n\n"
            retrieval_budget -= doc_tokens

        # Step 3: 注入检索文档到 system prompt 或末尾
        # 策略：作为独立 system message 注入，避免与历史混淆
        if docs_text:
            messages.append({
                "role": "system",
                "content": (
                    "<retrieved-documents>\n"
                    f"{docs_text}\n"
                    "</retrieved-documents>\n\n"
                    "[System note: Use ONLY the above documents to answer. "
                    "Cite sources with [number] notation.]"
                ),
            })

        # Step 4: 后台触发记忆保存
        # fire-and-forget, 不阻塞响应
        import asyncio
        asyncio.create_task(self.trigger.check_post_turn(messages, llm))

        return messages

    async def build_context_with_recovery(
        self,
        messages: list[dict],
        retrieved_docs: list[str],
        llm: "LLMClient",
        on_overflow: bool = False,
    ) -> list[dict]:
        """带溢出恢复的上下文构建 (用于 reactive path)"""
        if on_overflow:
            messages = await self.trigger.check_overflow(messages, llm)
        return await self.build_context(messages, retrieved_docs, llm)
```

---

### 6.7 与现有 ReasonAgent 的集成点

```python
# 在 ReasonAgent 中，reason_stream() 使用 ContextOrchestrator:

class ReasonAgent:
    def __init__(self):
        # ... existing init ...
        budget = TokenBudget(context_window=128_000)
        compactor = ConversationCompactor(llm=self, budget=budget)
        memory = SessionMemory(data_dir="./data/memory")
        trigger = CompactionTrigger(compactor, memory, budget)
        self.context_orchestrator = ContextOrchestrator(
            budget, compactor, memory, trigger,
        )

    async def reason_stream(
        self,
        query: str,
        results: list[SearchResult],
        citations: list[Citation],
        history: list[dict] | None = None,
    ):
        # 构建完整消息列表
        raw_messages = []
        if history:
            raw_messages.extend(history)
        raw_messages.append({"role": "user", "content": query})

        # 通过 ContextOrchestrator 处理
        docs_text = [
            f"[{c.citation_id}] {r.doc_title}\n{r.content}"
            for r, c in zip(results, citations)
        ]
        messages = await self.context_orchestrator.build_context(
            messages=raw_messages,
            retrieved_docs=docs_text,
            llm=self,
        )

        # 调用 LLM
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

---

### 6.8 目录结构

```
app/
├── context/
│   ├── __init__.py
│   ├── token_budget.py            # TokenBudget — 上下文 Token 分配器
│   ├── conversation_compactor.py  # ConversationCompactor — 对话压缩器
│   ├── session_memory.py          # SessionMemory — 长期记忆
│   ├── compaction_trigger.py      # CompactionTrigger — 多路径触发
│   └── context_orchestrator.py    # ContextOrchestrator — 上下文编排器
├── data/
│   └── memory/
│       ├── session_memory.md      # 跨 Session 记忆
│       └── user_preferences.md    # 用户偏好
```

---

### 6.9 实现阶段规划

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| **Phase 1 (Demo+)** | TokenBudget + ConversationCompactor (head/tail/compress) + CompactionTrigger (preflight + manual) | P0 |
| **Phase 2** | SessionMemory (跨 session 持久化) + CompactionTrigger.post_turn (后台记忆提取) | P1 |
| **Phase 3** | CompactionTrigger.overflow (溢出恢复) + Circuit Breaker (防抖) | P1 |
| **Phase 4** | 前端 `/compact` 命令 + 上下文使用率可视化 | P2 |
| **Phase 5** | MemoryProvider 插件体系 (外部记忆后端) + MemoryManager | P2 |

## 5. ExecuteAgent（执行 Agent）

### 5.1 状态机图

```
         ┌─────────────┐
         │ plan_calls   │  解析工具调用指令，排序（独立→并行，有依赖→串行）
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ execute_tools│  执行工具调用（并行 + 串行），含限流/熔断/重试
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ audit_log   │  记录审计日志（trace_id, tool, params, result, duration）
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ build_result │  汇总工具执行结果
         └─────────────┘
```

### 5.2 ToolRegistry

```python
import time
from enum import Enum
from typing import Awaitable, Callable, Optional
import asyncio

class ToolStatus(Enum):
    HEALTHY = "healthy"          # 正常
    DEGRADED = "degraded"       # 部分失败
    CIRCUIT_OPEN = "open"       # 熔断

class BaseTool(ABC):
    """工具基类"""
    name: str
    description: str
    permissions: list[str] = []                  # 允许的用户角色
    rate_limit: tuple[float, int] = (1.0, 10)    # (rate, capacity)
    timeout: int = 10                            # 默认超时
    max_retries: int = 3
    circuit_breaker_threshold: int = 5           # 连续失败阈值
    circuit_breaker_recovery: float = 30.0       # 半开探测等待时间
    fallback_tool: Optional[str] = None          # 降级工具名

    @abstractmethod
    async def execute(self, params: dict, context: ExecutionContext) -> ToolResult:
        """执行工具逻辑"""
        ...

    async def can_execute(self, user: dict) -> bool:
        """权限检查"""
        if not self.permissions:
            return True
        return user.get("role", "user") in self.permissions


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._limiters: dict[str, TokenBucket] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def register(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool
        # 初始化限流器
        self._limiters[tool.name] = TokenBucket(
            rate=tool.rate_limit[0],
            capacity=tool.rate_limit[1],
        )
        # 初始化熔断器
        self._circuit_breakers[tool.name] = CircuitBreaker(
            failure_threshold=tool.circuit_breaker_threshold,
            recovery_timeout=tool.circuit_breaker_recovery,
        )

    def unregister(self, tool_name: str):
        """注销工具"""
        self._tools.pop(tool_name, None)
        self._limiters.pop(tool_name, None)
        self._circuit_breakers.pop(tool_name, None)

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self, user: dict) -> list[dict]:
        """列出用户可用的所有工具（用于 LLM function calling 描述）"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t._get_schema(),
            }
            for t in self._tools.values()
            if t.can_execute(user)
        ]

    async def execute(
        self,
        tool_name: str,
        params: dict,
        context: ExecutionContext,
    ) -> ToolResult:
        """执行工具调用（带限流、熔断、重试、降级）"""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool '{tool_name}' not registered")

        # 1. 权限检查
        if not await tool.can_execute(context.user):
            raise PermissionDeniedError(f"User lacks permission for '{tool_name}'")

        # 2. 限流检查
        limiter = self._limiters[tool_name]
        if not limiter.consume():
            raise RateLimitError(f"Tool '{tool_name}' rate limited")

        # 3. 熔断检查
        breaker = self._circuit_breakers[tool_name]
        if breaker.is_open:
            # 尝试降级
            if tool.fallback_tool:
                logger.info(f"Circuit open for '{tool_name}', using fallback '{tool.fallback_tool}'")
                return await self.execute(tool.fallback_tool, params, context)
            raise CircuitBreakerOpenError(f"Circuit open for '{tool_name}'")

        # 4. 执行 + 重试
        last_error = None
        for attempt in range(1, tool.max_retries + 1):
            try:
                t0 = time.monotonic()
                result = await asyncio.wait_for(
                    tool.execute(params, context),
                    timeout=tool.timeout,
                )
                latency = time.monotonic() - t0

                # 成功后重置熔断器
                breaker.on_success()
                result.latency = latency
                result.tool_name = tool_name
                return result

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Tool '{tool_name}' timeout after {tool.timeout}s")
            except Exception as e:
                last_error = e

            # 指数退避
            wait = 2 ** (attempt - 1)
            logger.warning(f"Tool '{tool_name}' attempt {attempt}/{tool.max_retries} failed: {last_error}, retrying in {wait}s")
            await asyncio.sleep(wait)

        # 所有重试失败
        breaker.on_failure()
        raise ToolExecutionError(
            f"Tool '{tool_name}' failed after {tool.max_retries} attempts",
            cause=last_error,
        )
```

### 5.3 CircuitBreaker 实现

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"            # 正常
    OPEN = "open"                # 熔断
    HALF_OPEN = "half_open"      # 半开探测

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float | None = None

    @property
    def is_open(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return False
        if self.state == CircuitState.OPEN:
            # 检查是否可以进入半开
            if (
                self.last_failure_time
                and (time.monotonic() - self.last_failure_time) >= self.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker moved to HALF_OPEN")
                return False
            return True
        # HALF_OPEN → 允许尝试
        return False

    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
```

### 5.4 执行节点核心逻辑

```python
async def execute_tools_node(state: ExecuteState) -> dict:
    """
    将工具调用分组并行执行：
    1. 无依赖的工具 → 并行执行
    2. 有依赖的工具 → 串行执行
    """
    tool_registry: ToolRegistry = state["tool_registry"]
    tool_calls = state["tool_calls"]
    user = state["user_context"]

    # 构建依赖图，分配并行/串行批次
    batches = _build_execution_batches(tool_calls)

    all_results = []
    for batch in batches:
        # 同批次并行执行
        batch_tasks = [
            tool_registry.execute(
                tool_name=tc.name,
                params=tc.parameters,
                context=ExecutionContext(
                    user=user,
                    trace_id=state["trace_id"],
                    session_id=state["session_id"],
                ),
            )
            for tc in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)

    # 分类结果
    success_results = [r for r in all_results if isinstance(r, ToolResult)]
    error_results = [r for r in all_results if isinstance(r, Exception)]

    return {
        "tool_results": success_results,
        "tool_errors": [
            {"tool": tc.name, "error": str(e)}
            for tc, e in zip(tool_calls, error_results)
            if isinstance(e, Exception)
        ],
        "all_success": len(error_results) == 0,
    }
```

---

## 6. 变更日志

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-06-20 | 初始版本 | AI-assisted Design |
