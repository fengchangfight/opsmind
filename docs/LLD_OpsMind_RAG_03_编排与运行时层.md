# OpsMind-RAG 详细设计 — 编排与运行时层 (LLD-03)

**版本**: v1.0
**日期**: 2026-06-20
**对应 HLD**: HLD_OpsMind_RAG.md §2.1 编排与运行时层, §3.1 AgentRuntime

---

## 1. 模块职责

| 子模块 | 职责 |
|--------|------|
| **AgentRuntime** | 多 Agent 编排引擎：任务分解、调度、状态管理、中断恢复 |
| **TaskDecomposer** | 将用户查询分解为可并行/串行的子任务 |
| **ExecutionPlanner** | 生成多阶段执行计划，同阶段任务并行 |
| **SessionManager** | Session 生命周期管理、状态持久化、Checkpoint 读写 |
| **MessageBus** | 基于 Redis Streams 的 Agent 间异步通信基础设施 |
| **CheckpointManager** | LangGraph Checkpoint 的保存与恢复（Redis 后端） |

---

## 2. 目录结构

```
app/core/
├── __init__.py
├── agent_runtime.py        # AgentRuntime 主类
├── task_decomposer.py      # 任务分解
├── execution_planner.py    # 执行计划生成
├── session_manager.py      # Session 与状态管理
├── message_bus.py          # Redis Streams 封装
├── checkpoint_manager.py   # Checkpoint 持久化
└── event_queue.py          # SSE 事件队列封装
```

---

## 3. AgentRuntime 核心设计

### 3.1 类结构

```python
from typing import Optional
from opsmind.agents.retrieve_agent import RetrieveAgent
from opsmind.agents.reason_agent import ReasonAgent
from opsmind.agents.execute_agent import ExecuteAgent

class AgentRuntime:
    """多 Agent 编排引擎，对标 Hermes Agent Runtime"""

    def __init__(
        self,
        session_manager: SessionManager,
        message_bus: MessageBus,
        task_decomposer: TaskDecomposer,
        execution_planner: ExecutionPlanner,
        checkpoint_manager: CheckpointManager,
    ):
        self.session_manager = session_manager
        self.message_bus = message_bus
        self.task_decomposer = task_decomposer
        self.execution_planner = execution_planner
        self.checkpoint_manager = checkpoint_manager

        # Agent 注册表
        self._agents: dict[str, BaseAgent] = {}

        # 当前活跃的推理任务（用于取消）
        self._active_tasks: dict[str, asyncio.Task] = {}

    def register_agent(self, name: str, agent: BaseAgent):
        """注册 Agent"""
        self._agents[name] = agent
        # 订阅 Message Bus 中的 Agent 专属 Stream
        self.message_bus.create_consumer_group(f"agent:{name}", name)

    def get_agent(self, name: str) -> BaseAgent:
        return self._agents[name]

    async def initialize(self):
        """启动时初始化所有 Agent 和基础设施"""
        for agent in self._agents.values():
            await agent.initialize()
        await self.message_bus.initialize()

    async def shutdown(self):
        """优雅关闭"""
        for task in self._active_tasks.values():
            task.cancel()
        for agent in self._agents.values():
            await agent.shutdown()
        await self.message_bus.close()
```

### 3.2 主查询流程

```python
async def execute_query(
    self,
    query: str,
    session: Session,
    options: QueryOptions,
    event_queue: asyncio.Queue[SSEMessage],
    trace_ctx: SpanContext,
) -> None:
    """
    完整 RAG + Agent 推理链路。
    通过 event_queue 将实时事件推送到 SSE 流。
    """
    tracer = get_tracer()

    try:
        # ---- Phase 1: 任务分解 ----
        with tracer.start_span("task_decomposition") as span:
            tasks = await self.task_decomposer.decompose(
                query=query,
                context=session.context,
                options=options,
            )
            span.set_attribute("task_count", len(tasks))

        # 保存到 Session
        session.tasks = tasks
        await self.session_manager.save_snapshot(session)

        # ---- Phase 2: 执行计划 ----
        with tracer.start_span("execution_planning") as span:
            execution_plan = await self.execution_planner.plan(tasks)
            span.set_attribute("stage_count", len(execution_plan.stages))

        # ---- Phase 3: 按阶段执行 ----
        accumulated_context = {}

        for stage in execution_plan.stages:
            # 发射 Agent 开始事件
            for agent_id in stage.agents:
                await event_queue.put(SSEMessage(
                    event="agent_start",
                    data={"agent_id": agent_id, "stage": stage.name},
                ))

            # 并行执行当前阶段的所有任务
            stage_results = await self._execute_stage(
                stage=stage,
                session=session,
                accumulated_context=accumulated_context,
                event_queue=event_queue,
                trace_ctx=trace_ctx,
            )

            # 合并结果
            accumulated_context.update(stage_results)

            # Checkpoint 保存
            await self.checkpoint_manager.save(
                session_id=session.session_id,
                stage_name=stage.name,
                state={
                    "accumulated_context": accumulated_context,
                    "completed_stages": [
                        s.name for s in execution_plan.stages
                        if execution_plan.stages.index(s) <= execution_plan.stages.index(stage)
                    ],
                },
            )

            # 检查是否需要中断
            if stage_results.get("interrupted"):
                session.status = "interrupted"
                await self.session_manager.save_snapshot(session)
                return  # 中断，等待人工恢复

        # ---- Phase 4: 最终答案 ----
        final_answer = accumulated_context.get("final_answer", {})
        await event_queue.put(SSEMessage(
            event="final_answer",
            data=final_answer,
        ))

        session.status = "completed"
        await self.session_manager.save_snapshot(session)

    except Exception as e:
        logger.error(f"Query execution failed: {e}", extra={"trace_id": trace_ctx.trace_id})
        await event_queue.put(SSEMessage(
            event="error",
            data={"code": "INTERNAL_ERROR", "message": str(e), "trace_id": trace_ctx.trace_id},
        ))
        session.status = "failed"
        await self.session_manager.save_snapshot(session)
```

### 3.3 中断恢复流程

```python
async def resume_session(
    self,
    session: Session,
    human_input: str,
    option: str,
    event_queue: asyncio.Queue[SSEMessage],
    trace_ctx: SpanContext,
) -> None:
    """
    从中断点恢复执行。
    option: continue | modify | transfer
    """
    tracer = get_tracer()

    # 1. 从 Redis 加载最后的 Checkpoint
    with tracer.start_span("checkpoint_load"):
        checkpoint = await self.checkpoint_manager.load(session.session_id)
        if not checkpoint:
            raise RuntimeError(f"No checkpoint found for session {session.session_id}")

    # 2. 从 Checkpoint 中恢复状态
    completed_stages = checkpoint.state.get("completed_stages", [])
    accumulated_context = checkpoint.state.get("accumulated_context", {})

    # 3. 根据用户选项处理
    with tracer.start_span("resume_processing") as span:
        span.set_attribute("resume_option", option)

        if option == "continue":
            # 将人工输入注入到上下文
            accumulated_context["human_feedback"] = human_input

        elif option == "modify":
            # 修改查询后重新开始
            accumulated_context["modified_query"] = human_input
            # 可能需要重新规划（简化处理：继续从当前阶段）

        elif option == "transfer":
            # 转人工处理，生成总结
            await event_queue.put(SSEMessage(
                event="final_answer",
                data={"answer": f"已转人工处理。当前进度：{completed_stages}。"},
            ))
            session.status = "transferred"
            await self.session_manager.save_snapshot(session)
            return

    # 4. 从下一个阶段继续执行
    execution_plan = await self.execution_planner.plan(session.tasks)
    remaining_stages = [
        s for s in execution_plan.stages
        if s.name not in completed_stages
    ]

    for stage in remaining_stages:
        stage_results = await self._execute_stage(
            stage=stage,
            session=session,
            accumulated_context=accumulated_context,
            event_queue=event_queue,
            trace_ctx=trace_ctx,
        )
        accumulated_context.update(stage_results)

        await self.checkpoint_manager.save(
            session_id=session.session_id,
            stage_name=stage.name,
            state={"accumulated_context": accumulated_context},
        )

    # 5. 最终答案
    session.status = "completed"
    await self.session_manager.save_snapshot(session)
```

---

## 4. TaskDecomposer

### 4.1 设计

```python
class TaskDecomposer:
    """将用户自然语言查询分解为结构化子任务"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def decompose(
        self,
        query: str,
        context: QueryContext,
        options: QueryOptions,
    ) -> list[Task]:
        """
        返回任务列表。
        使用 LLM 进行任务分解，返回结构化 JSON。
        """
        prompt = DECOMPOSITION_PROMPT.format(
            query=query,
            context=context.model_dump_json(),
            available_tools=options.tools,
        )

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
        )

        raw_tasks = json.loads(response.content)["tasks"]

        tasks = []
        for i, raw in enumerate(raw_tasks):
            tasks.append(Task(
                task_id=f"task-{uuid.uuid4().hex[:8]}",
                task_type=raw["type"],
                description=raw["description"],
                parameters=raw.get("parameters", {}),
                dependencies=raw.get("depends_on", []),
                priority=i,
                timeout=raw.get("timeout", 30),
                max_retries=raw.get("max_retries", 2),
            ))

        # 解析依赖：将逻辑索引转为实际 task_id
        for task in tasks:
            task.dependencies = [
                tasks[int(d)].task_id for d in task.dependencies
                if isinstance(d, int)
            ]

        return tasks
```

### 4.2 分解 Prompt 模板

```
系统提示：
你是一个运维任务分解专家。根据用户查询，将其分解为可独立执行的子任务。
每个子任务应包含类型(type)、描述(description)、参数(parameters)、依赖(depends_on)。

可用任务类型：
- retrieve: 检索知识库获取相关信息
- reason: 分析上下文并推理
- execute: 调用外部工具
- composite: 组合多个子任务

返回格式：
{
  "tasks": [
    {
      "type": "retrieve",
      "description": "检索 MySQL 主从延迟相关文档",
      "parameters": {"query": "MySQL 主从延迟排查", "top_k": 10},
      "depends_on": [],
      "timeout": 15,
      "max_retries": 2
    }
  ]
}
```

---

## 5. ExecutionPlanner

### 5.1 核心算法（拓扑排序）

```python
class ExecutionPlanner:
    """
    基于任务依赖关系生成多阶段执行计划。
    使用拓扑排序 + 依赖分组算法。
    """

    async def plan(self, tasks: list[Task]) -> ExecutionPlan:
        """
        输入：无序任务列表
        输出：ExecutionPlan { stages: [ExecutionStage] }
        每个 stage 包含可并行执行的任务（无互相依赖）
        """
        # 构建任务图
        task_map = {t.task_id: t for t in tasks}
        in_degree = {t.task_id: 0 for t in tasks}

        # 计算入度（被依赖数）
        for task in tasks:
            for dep_id in task.dependencies:
                in_degree[task.task_id] += 1

        # 分组算法：每一轮取出所有入度为 0 的任务
        stages = []
        processed = set()

        while len(processed) < len(tasks):
            current_stage = []
            for task in tasks:
                if task.task_id in processed:
                    continue
                if all(d in processed for d in task.dependencies):
                    current_stage.append(task)

            if not current_stage:
                raise CircularDependencyError("Circular dependency detected")

            stages.append(ExecutionStage(
                name=f"stage-{len(stages)+1}",
                tasks=current_stage,
                agents=self._assign_agents(current_stage),
                parallel=True,
            ))

            processed.update(t.task_id for t in current_stage)

        return ExecutionPlan(stages=stages)

    def _assign_agents(self, tasks: list[Task]) -> list[str]:
        """根据任务类型分配 Agent"""
        type_to_agent = {
            "retrieve": "retrieve",
            "reason": "reason",
            "execute": "execute",
            "composite": "reason",  # 复合任务由推理 Agent 协调
        }
        return list(set(type_to_agent[t.task_type] for t in tasks))
```

---

## 6. SessionManager

### 6.1 Session 状态机

```
         ┌──────────────────────────────┐
         │                              │
         ▼                              │
    ┌─────────┐    execute_query    ┌────────┐
    │ created │────────────────────▶│ running│
    └─────────┘                     └───┬────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
              ┌───────────┐      ┌───────────┐      ┌───────────┐
              │interrupted│      │ completed │      │  failed   │
              └─────┬─────┘      └───────────┘      └───────────┘
                    │
                    │ resume
                    ▼
              ┌───────────┐
              │  running  │ (恢复)
              └───────────┘
                    │
                    │ transfer (人为决定)
                    ▼
              ┌──────────────┐
              │ transferred  │ (终端状态)
              └──────────────┘
```

### 6.2 实现

```python
class SessionManager:
    """Session 生命周期管理 + 状态持久化"""

    def __init__(self, redis: Redis):
        self.redis = redis
        self.SESSION_PREFIX = "session:"
        self.SESSION_TTL = 3600 * 24  # 24 小时

    async def create_session(
        self,
        user_id: str,
        context: QueryContext | None = None,
    ) -> Session:
        session = Session(
            session_id=f"sess-{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            context=context,
            status="created",
            created_at=datetime.now(timezone.utc),
        )
        await self.save_snapshot(session)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        key = f"{self.SESSION_PREFIX}{session_id}"
        data = await self.redis.get(key)
        if not data:
            return None
        return Session.model_validate_json(data)

    async def save_snapshot(self, session: Session):
        key = f"{self.SESSION_PREFIX}{session.session_id}"
        session.updated_at = datetime.now(timezone.utc)
        await self.redis.setex(
            key, self.SESSION_TTL, session.model_dump_json()
        )

    async def delete_session(self, session_id: str):
        key = f"{self.SESSION_PREFIX}{session_id}"
        await self.redis.delete(key)
        # 同时清理关联的 checkpoints
        await self.redis.delete(f"checkpoint:{session_id}:*")

    async def list_user_sessions(self, user_id: str) -> list[Session]:
        """列出用户的所有 session（简化实现）"""
        pattern = f"{self.SESSION_PREFIX}*"
        keys = await self.redis.keys(pattern)
        sessions = []
        for key in keys:
            data = await self.redis.get(key)
            if data:
                s = Session.model_validate_json(data)
                if s.user_id == user_id:
                    sessions.append(s)
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)
```

---

## 7. MessageBus (Redis Streams)

### 7.1 核心接口

```python
from redis.asyncio import Redis
from typing import AsyncIterator, Callable, Optional

class MessageBus:
    """基于 Redis Streams 的 Agent 间异步通信基础设施"""

    # Stream 命名约定: "stream:agent:{agent_name}"
    # Consumer Group:   "group:{service_name}"
    # Dead Letter:      "stream:dlq"

    def __init__(self, redis: Redis):
        self.redis = redis
        self._consumers: dict[str, asyncio.Task] = {}

    async def initialize(self):
        """创建必要的 Consumer Groups"""
        for agent_name in self._known_agents:
            try:
                await self.create_consumer_group(agent_name, agent_name)
            except Exception:
                pass  # Group 可能已存在

    # --- 点对点 (P2P) ---
    async def send_to_agent(
        self,
        target_agent: str,
        message: Message,
    ) -> str:
        """
        发送消息到指定 Agent 的 Stream。
        返回 message_id。
        """
        stream_key = f"stream:agent:{target_agent}"
        return await self.redis.xadd(
            stream_key,
            {
                "data": message.model_dump_json(),
                "timestamp": message.timestamp,
            },
            maxlen=10_000,  # 限制 Stream 长度
        )

    # --- Subscribe (Consumer Group 模式) ---
    async def subscribe(
        self,
        agent_name: str,
        consumer_name: str,
        handler: Callable[[Message], Awaitable[None]],
        batch_size: int = 10,
        block_ms: int = 5000,
    ):
        """
        消费指定 Agent Stream 中的消息。
        使用 Consumer Group 支持多个消费者负载均衡。
        """
        stream_key = f"stream:agent:{agent_name}"
        group_name = f"group:{agent_name}"

        pending = await self.redis.xpending_range(
            stream_key, group_name, min="-", max="+", count=100
        )
        for entry in pending:
            await self._process_pending(entry, stream_key, group_name, handler)

        # 主消费循环
        while True:
            try:
                messages = await self.redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {stream_key: ">"},
                    count=batch_size,
                    block=block_ms,
                )

                for stream, entries in messages:
                    for msg_id, fields in entries:
                        await self._process_message(
                            msg_id, fields, stream_key, group_name, handler
                        )

            except asyncio.CancelledError:
                break

    async def _process_message(self, msg_id, fields, stream_key, group_name, handler):
        """处理单条消息 + 幂等性检查 + ACK/死信"""
        try:
            msg_data = fields.get(b"data", b"{}").decode()
            message = Message.model_validate_json(msg_data)

            # 幂等性检查：以 message_id 为去重依据
            if await self._is_duplicate(message.message_id):
                await self.redis.xack(stream_key, group_name, msg_id)
                return

            await handler(message)

            # 确认处理成功
            await self.redis.xack(stream_key, group_name, msg_id)
            await self._mark_processed(message.message_id)

        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            # 重试逻辑：Redis Streams XPENDING 会自动重试
            # 如果失败次数超过阈值 → 发送到 DLQ
            retry_count = fields.get(b"retry_count", 0)
            if int(retry_count) >= 3:
                await self._send_to_dlq(msg_id, fields, str(e))
                await self.redis.xack(stream_key, group_name, msg_id)

    # --- 广播 (Pub/Sub) ---
    async def broadcast(self, channel: str, message: dict):
        """
        通过 Redis PUBLISH 广播事件。
        用于状态变更通知等场景。
        """
        await self.redis.publish(
            f"broadcast:{channel}",
            json.dumps(message),
        )

    async def subscribe_broadcast(
        self, channel: str, handler: Callable[[dict], Awaitable[None]]
    ):
        """订阅广播频道"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"broadcast:{channel}")

        async for raw in pubsub.listen():
            if raw["type"] == "message":
                data = json.loads(raw["data"])
                await handler(data)

    # --- 死信队列 ---
    async def _send_to_dlq(self, original_msg_id, fields, error):
        await self.redis.xadd(
            "stream:dlq",
            {
                "original_msg_id": original_msg_id,
                "fields": json.dumps(fields),
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # --- 背压控制 ---
    async def get_stream_length(self, agent_name: str) -> int:
        """检查 Stream 积压量，用于背压决策"""
        return await self.redis.xlen(f"stream:agent:{agent_name}")

    # --- 幂等性 ---
    async def _is_duplicate(self, message_id: str) -> bool:
        return await self.redis.sismember("processed:messages", message_id)

    async def _mark_processed(self, message_id: str):
        await self.redis.sadd("processed:messages", message_id)
        await self.redis.expire("processed:messages", 3600)
```

### 7.2 Message 数据结构

```python
from pydantic import BaseModel

class Message(BaseModel):
    message_id: str            # 全局唯一消息 ID
    correlation_id: str        # 关联的 Trace ID
    sender: str                # "agent://retrieve"
    recipient: str             # "agent://reason"
    message_type: str          # REQUEST | RESPONSE | BROADCAST | EVENT
    payload: dict              # 消息负载
    timestamp: str             # ISO 8601
    ttl: int = 300             # 过期时间（秒）

class MessageType:
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    BROADCAST = "BROADCAST"
    EVENT = "EVENT"
```

---

## 8. CheckpointManager

```python
class CheckpointManager:
    """
    LangGraph Checkpoint 的 Redis 持久化后端。
    也用于 AgentRuntime 级别的阶段进度保存。
    """

    def __init__(self, redis: Redis):
        self.redis = redis
        self.PREFIX = "checkpoint:"
        self.TTL = 86400  # 24h

    async def save(
        self,
        session_id: str,
        stage_name: str,
        state: dict,
    ):
        """保存当前阶段快照"""
        key = f"{self.PREFIX}{session_id}:{stage_name}"
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "state": json.dumps(state, default=str),
        }
        await self.redis.setex(key, self.TTL, json.dumps(data))

        # 维护 session 的 checkpoint 链
        chain_key = f"{self.PREFIX}{session_id}:chain"
        await self.redis.rpush(chain_key, key)
        await self.redis.expire(chain_key, self.TTL)

    async def load(self, session_id: str) -> Checkpoint | None:
        """加载最近的 checkpoint"""
        chain_key = f"{self.PREFIX}{session_id}:chain"
        last_key = await self.redis.lindex(chain_key, -1)
        if not last_key:
            return None
        data = await self.redis.get(last_key)
        if not data:
            return None
        raw = json.loads(data)
        return Checkpoint(
            stage_name=last_key.split(":")[-1],
            saved_at=raw["saved_at"],
            state=json.loads(raw["state"]),
        )

    async def list_checkpoints(self, session_id: str) -> list[str]:
        """列出 session 的所有 checkpoint 阶段名"""
        chain_key = f"{self.PREFIX}{session_id}:chain"
        keys = await self.redis.lrange(chain_key, 0, -1)
        return [k.split(":")[-1] for k in keys]
```

---

## 9. 线程安全与并发

| 场景 | 策略 |
|------|------|
| 同一 Session 并发请求 | Redis 分布式锁 `SETNX lock:session:{id}` |
| Agent 注册/注销 | `asyncio.Lock` 保护 `self._agents` |
| Checkpoint 写入 | Redis 单线程天然原子操作 |
| Event Queue 写入 | `asyncio.Queue`（内置线程安全） |
| Message Bus 消费 | Redis Consumer Group 自动负载均衡 |

---

## 10. 关键时序

### 10.1 正常查询链路

```
API Layer         AgentRuntime       TaskDecomposer    ExecutionPlanner   Agent       SessionManager   Checkpoint
  │                    │                   │                  │              │              │               │
  │──execute_query────▶│                   │                  │              │              │               │
  │                    │──decompose───────▶│                  │              │              │               │
  │                    │◀──tasks───────────│                  │              │              │               │
  │                    │──save─────────────────────────────────────────────────────────────▶│               │
  │                    │──plan──────────────────────────────▶│              │              │               │
  │                    │◀──execution_plan────────────────────│              │              │               │
  │                    │                                       │              │              │               │
  │◀──SSE:agent_start──│                                       │              │              │               │
  │                    │──execute_stage─────────────────────────────────────▶│              │               │
  │◀──SSE:events───────│                                       │              │              │               │
  │                    │──save_checkpoint─────────────────────────────────────────────────────────────────▶│
  │                    │  (each stage)                          │              │              │               │
  │◀──SSE:final_answer │                                       │              │              │               │
  │                    │──save "completed"──────────────────────────────────────────────────▶│               │
```

### 10.2 中断恢复链路

```
API Layer         AgentRuntime       CheckpointManager     SessionManager    Agent
  │                    │                     │                    │              │
  │──resume_session──▶ │                     │                    │              │
  │                    │──load_checkpoint───▶│                    │              │
  │                    │◀──checkpoint────────│                    │              │
  │                    │──save "running"─────────────────────────▶│              │
  │                    │──resume_from_stage────────────────────────────────────▶│
  │◀──SSE:events───────│                     │                    │              │
  │◀──SSE:final_answer │                     │                    │              │
```

---

## 11. 变更日志

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-06-20 | 初始版本 | AI-assisted Design |
