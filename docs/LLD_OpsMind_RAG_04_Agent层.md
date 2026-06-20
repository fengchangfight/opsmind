# OpsMind-RAG 详细设计 — Agent 层 (LLD-04)

**版本**: v1.0
**日期**: 2026-06-20
**对应 HLD**: HLD_OpsMind_RAG.md §2.1 Agent 层, §3.2-3.5

---

## 1. 模块总览

| Agent | 职责 | LangGraph 状态机节点数 | 关键能力 |
|-------|------|----------------------|---------|
| **RetrieveAgent** | 混合检索 + 重排序 | 7 个节点 | 稠密/稀疏双路召回, RRF 融合, Cross-Encoder 重排序, 查询扩展, 上下文构建 |
| **ReasonAgent** | 迭代推理 + 置信度评估 | 6 个节点 | 假设生成, 证据评估, 多跳检索, 置信度计算, 中断恢复 |
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
         │dense_search │  并行 ┐
         ├─────────────┤       ├── 两路检索同时执行
         │sparse_search│  并行 ┘
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  rrf_fusion │  倒数排名融合 (RRF)，合并两路结果
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
    dense_results: list[ChunkResult]    # 稠密检索结果
    sparse_results: list[ChunkResult]   # 稀疏检索结果
    fused_results: list[ChunkResult]    # RRF 融合结果
    reranked_results: list[ChunkResult] # 重排序后结果
    final_context: str                  # 最终拼接的上下文
    citations: list[Citation]           # 溯源引用列表
    latency: dict[str, float]           # 各阶段延迟

class ChunkResult:
    chunk_id: str
    content: str
    score: float                        # 检索得分
    doc_id: str
    doc_title: str
    section_path: list[str]
    start_line: int
    end_line: int
```

### 3.3 核心节点实现

#### 3.3.1 DenseSearch 节点

```python
async def dense_search_node(state: RetrieveState) -> dict:
    """稠密向量检索"""
    t0 = time.monotonic()

    embeddings = await embedder.encode(state["query"])
    # Milvus.search(
    #     collection_name="chunks",
    #     data=[embeddings],
    #     anns_field="embedding_dense",
    #     param={"metric_type": "IP", "params": {"ef": 128}},
    #     limit=top_k * 2,  # 粗排多取一些，留给 RRF 融合
    #     expr=build_filter_expr(state["filters"]),
    # )
    results = await milvus_store.dense_search(
        vectors=embeddings,
        top_k=state["top_k"] * 2,
        filters=state["filters"],
    )

    state["latency"]["dense"] = time.monotonic() - t0
    return {"dense_results": results}
```

#### 3.3.2 SparseSearch 节点

```python
async def sparse_search_node(state: RetrieveState) -> dict:
    """稀疏关键词检索（BM25 等效）"""
    t0 = time.monotonic()

    # 使用 BGE-M3 的稀疏编码能力
    sparse_vector = await embedder.encode_sparse(state["query"])
    results = await milvus_store.sparse_search(
        vectors=sparse_vector,
        top_k=state["top_k"] * 2,
        filters=state["filters"],
    )

    state["latency"]["sparse"] = time.monotonic() - t0
    return {"sparse_results": results}
```

#### 3.3.3 RRF Fusion 节点（核心算法）

```python
def rrf_fusion_node(state: RetrieveState) -> dict:
    """
    Reciprocal Rank Fusion (RRF) 融合算法。
    不依赖 Milvus 内置融合，自己实现以加深理解。

    公式: score(doc) = Σ (1 / (k + rank_i(doc)))
    其中 k = 60（经验值），rank_i 是 doc 在第 i 路检索中的排名。
    """
    t0 = time.monotonic()
    k = 60
    dense_results = state["dense_results"]
    sparse_results = state["sparse_results"]

    # chunk_id → RRF score
    scores: dict[str, float] = {}
    doc_meta: dict[str, ChunkResult] = {}

    # 稠密排名贡献
    for rank, result in enumerate(dense_results, start=1):
        scores[result.chunk_id] = 1.0 / (k + rank)
        doc_meta[result.chunk_id] = result

    # 稀疏排名贡献（累加）
    for rank, result in enumerate(sparse_results, start=1):
        if result.chunk_id in scores:
            scores[result.chunk_id] += 1.0 / (k + rank)
        else:
            scores[result.chunk_id] = 1.0 / (k + rank)
            doc_meta[result.chunk_id] = result

    # 按 RRF 分数排序
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results = []
    for chunk_id in sorted_ids:
        r = doc_meta[chunk_id]
        r.score = scores[chunk_id]
        fused_results.append(r)

    state["latency"]["fusion"] = time.monotonic() - t0
    return {
        "fused_results": fused_results[:state["top_k"]],
        "latency": state["latency"],
    }
```

#### 3.3.4 Rerank 节点

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

#### 3.3.5 BuildContext 节点

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
    self.state_graph.add_node("dense_search", dense_search_node)
    self.state_graph.add_node("sparse_search", sparse_search_node)
    self.state_graph.add_node("rrf_fusion", rrf_fusion_node)
    self.state_graph.add_node("rerank", rerank_node)
    self.state_graph.add_node("build_context", build_context_node)

    self.state_graph.set_entry_point("parse_query")
    self.state_graph.add_edge("parse_query", "expand_query")
    self.state_graph.add_edge("expand_query", "dense_search")
    self.state_graph.add_edge("expand_query", "sparse_search")
    # 注意：LangGraph 中，dense/search 并行执行后汇聚到 rrf_fusion
    self.state_graph.add_edge("dense_search", "rrf_fusion")
    self.state_graph.add_edge("sparse_search", "rrf_fusion")
    self.state_graph.add_edge("rrf_fusion", "rerank")
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

---

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
