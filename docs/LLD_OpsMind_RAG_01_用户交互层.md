# OpsMind-RAG 详细设计 — 用户交互层 (LLD-01)

**版本**: v1.0
**日期**: 2026-06-20
**对应 HLD**: HLD_OpsMind_RAG.md §2.1 用户交互层

---

## 1. 模块职责

| 子模块 | 职责 |
|--------|------|
| **ChatPanel** | 对话式问答主界面，流式展示 AI 回答与引用标记 |
| **CitationPanel** | 右侧溯源面板，展示引用来源、原文摘录、跳转链接 |
| **AgentTrace** | Agent 执行过程可视化，展示活跃 Agent、步骤、耗时 |
| **InterruptDialog** | 人机协作中断面板，展示中断原因、选项（继续/修改/转人工） |
| **SessionManager** | 对话历史管理，支持多会话、历史恢复、断点续传 |

---

## 2. 组件树

```
<App>
├── <Header>
│   ├── Logo / Title
│   ├── SessionSelector          // 多会话切换
│   └── SettingsButton           // 配置入口
├── <Layout>
│   ├── <ChatPanel>              // 左侧：对话区
│   │   ├── <MessageList>        // 消息列表（滚动容器）
│   │   │   ├── <UserBubble>
│   │   │   │   ├── Content
│   │   │   │   └── Options (重试/编辑)
│   │   │   └── <AIBubble>
│   │   │       ├── <StreamingContent>  // 流式渲染 Markdown
│   │   │       ├── <CitationMarker>    // 行内引用标记 [1][2]
│   │   │       └── <ToolCallBadge>     // 工具调用标识
│   │   ├── <InputBox>
│   │   │   ├── TextArea (Shift+Enter 换行)
│   │   │   ├── AttachmentButton
│   │   │   └── SendButton
│   │   └── <AgentTrace>         // 可折叠的执行过程面板
│   │       ├── TraceTimeline    // 时间线展示
│   │       │   ├── TraceNode (agent_start|retrieval|reasoning|tool_call|final)
│   │       │   │   ├── Icon
│   │       │   │   ├── Label
│   │       │   │   ├── Duration
│   │       │   │   └── ExpandableDetail
│   │       │   └── ...
│   │       └── InterruptBanner  // 中断状态横幅
│   └── <CitationPanel>          // 右侧：溯源面板
│       ├── CitationList
│       │   └── <CitationCard>
│       │       ├── CitationId [1]
│       │       ├── DocTitle (可点击跳转)
│       │       ├── Excerpt (高亮匹配)
│       │       └── RelevanceScore
│       └── EmptyState
├── <InterruptDialog>            // 模态弹窗
│   ├── Reason (中断原因)
│   ├── Options (继续/修改查询/转人工)
│   ├── TextArea (人工输入)
│   └── ActionButtons
└── <ToastNotifier>              // 全局通知
```

---

## 3. 状态管理 (Zustand Store)

### 3.1 chatStore.ts

```typescript
interface ChatState {
  // === Session ===
  sessions: Session[];
  activeSessionId: string | null;
  createSession: () => string;
  switchSession: (id: string) => void;
  deleteSession: (id: string) => void;

  // === Messages ===
  messages: Record<string, Message[]>;  // sessionId -> messages
  isStreaming: boolean;
  addMessage: (sessionId: string, msg: Message) => void;
  appendStreamToken: (sessionId: string, msgId: string, token: string) => void;

  // === Agent Trace ===
  traceEvents: TraceEvent[];
  activeAgents: Set<string>;
  pushTraceEvent: (event: TraceEvent) => void;
  clearTrace: () => void;

  // === Interrupt ===
  interruptState: InterruptState | null;  // null = no interrupt
  setInterrupt: (state: InterruptState) => void;
  clearInterrupt: () => void;

  // === Citations ===
  citations: Citation[];
  setCitations: (citations: Citation[]) => void;
  highlightCitation: (citationId: string) => void;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  citations?: string[];       // 关联的 citation_id 列表
  traceEvents?: TraceEvent[]; // 该消息对应的 trace
  createdAt: number;
  isStreaming?: boolean;
}

interface TraceEvent {
  eventId: string;
  eventType: 'agent_start' | 'agent_finish' | 'retrieval_result'
           | 'reasoning_step' | 'tool_call' | 'tool_result'
           | 'interrupted' | 'final_answer' | 'error';
  agentId: string;
  timestamp: number;
  duration?: number;
  data: Record<string, unknown>;
}

interface InterruptState {
  reason: string;
  confidence: number;
  options: string[];
  sessionId: string;
}

interface Citation {
  citationId: string;
  docId: string;
  docTitle: string;
  excerpt: string;
  sourceUrl: string | null;
  relevanceScore: number;
  isHighlighted: boolean;
}
```

### 3.2 Store 设计决策

| 决策 | 理由 |
|------|------|
| **Zustand 而非 Redux** | 更轻量，适合中小型单页应用，减少 Boilerplate |
| **Record<sessionId, messages[]>** | 支持多会话隔离，切换时不丢失上下文 |
| **traceEvents 为全局数组** | 同一时刻只有一个活跃 Agent 链路，简化状态同步 |
| **interruptState 为单例** | 一次只有一个中断，简化 UI 逻辑 |

---

## 4. SSE 事件处理

### 4.1 EventSource 连接管理

```typescript
// hooks/useSSE.ts
function useSSE(sessionId: string) {
  const eventSourceRef = useRef<EventSource | null>(null);

  const connect = useCallback((query: string) => {
    const url = `/api/query?query=${encodeURIComponent(query)}&session_id=${sessionId}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data: SSEMessage = JSON.parse(event.data);
      eventRouter.dispatch(data);
    };

    es.onerror = () => {
      // 自动重连逻辑（SSE 原生支持）
      // 3 次失败后降级为轮询
    };
  }, [sessionId]);

  const disconnect = useCallback(() => {
    eventSourceRef.current?.close();
  }, []);

  return { connect, disconnect };
}
```

### 4.2 事件路由表

| SSE Event | 触发动作 | 目标 Store 方法 |
|-----------|---------|----------------|
| `agent_start` | 创建新的 TraceNode 并标记为活跃 | `pushTraceEvent` |
| `agent_finish` | 关闭 TraceNode，记录耗时 | `pushTraceEvent` |
| `retrieval_result` | 更新 CitationPanel 数据 | `setCitations` |
| `reasoning_step` | 展示推理步骤和假设 | `pushTraceEvent` |
| `tool_call` | 展示工具调用标识 | `pushTraceEvent` |
| `tool_result` | 更新工具调用结果 | `pushTraceEvent` |
| `interrupted` | 弹出 InterruptDialog | `setInterrupt` |
| `final_answer` | 流式渲染最终答案 + 引用 | `appendStreamToken` |
| `chunk` | 追加 token 到当前 AI 消息 | `appendStreamToken` |
| `error` | 显示 Toast 错误通知 | `Toast.show` |

### 4.3 流式渲染 (StreamingContent)

```typescript
// 逐 token 渲染 Markdown，使用分区缓冲避免重复解析
function StreamingContent({ content, citations }: Props) {
  // 策略：每 50ms 或遇到 \n 时重新渲染一次 Markdown
  // 使用 debounce 减少解析频率
  // 内联引用标记 [1] → 点击高亮右侧 CitationCard
}
```

---

## 5. 组件交互时序

### 5.1 主查询链路

```
用户输入查询 → ChatPanel
  └─→ InputBox.onSubmit()
       └─→ chatStore.addMessage(userMessage)
       └─→ useSSE.connect(query)
            └─→ EventSource → /api/query (SSE)
                 └─→ eventRouter.dispatch(event)
                      ├─→ chatStore.pushTraceEvent (agent_start)
                      ├─→ chatStore.setCitations (retrieval_result)
                      ├─→ chatStore.appendStreamToken (chunk)
                      ├─→ chatStore.setInterrupt (interrupted)
                      │    └─→ InterruptDialog.open()
                      │         └─→ 用户选择 → POST /api/resume
                      │              └─→ 继续 SSE 事件流
                      └─→ chatStore.appendStreamToken (final_answer)
```

### 5.2 中断恢复链路

```
InterruptDialog → 用户选择 "继续" + 输入补充信息
  └─→ POST /api/resume { session_id, human_input, option: "continue" }
       └─→ 后端从中断点恢复，SSE 继续推送事件
            └─→ chatStore.clearInterrupt()
            └─→ 继续接收 reasoning_step / final_answer 事件
```

---

## 6. 关键交互设计

### 6.1 Citation 联动

```
AIBubble 中点击 [1] 引用标记
  └─→ chatStore.highlightCitation("1")
       └─→ CitationPanel 中对应 Card 高亮 + 滚动到可视区域
       └─→ AIBubble 中标记变为蓝色下划线样式
```

### 6.2 AgentTrace 展开/折叠

| 状态 | 行为 |
|------|------|
| 默认 | 折叠在 InputBox 上方，仅显示进度条 |
| 展开 | 展示完整 Timeline，自动滚动到最新节点 |
| 完成 | 显示总耗时，5s 后自动折叠 |

### 6.3 会话切换

```
用户点击 SessionSelector 切换会话
  └─→ chatStore.switchSession(newSessionId)
       └─→ messages 列表切换为对应会话
       └─→ citations 清空
       └─→ AgentTrace 清空
       └─→ 如有活跃 SSE 连接 → 断开并提示用户
```

---

## 7. UI/UX 规范

### 7.1 布局（参考 Claude Code）

```
┌────────────────────────────────────────────────────────┐
│  Header: OpsMind RAG  |  [Session ▼]  |  [Settings]   │
├─────────────────────────────────┬──────────────────────┤
│                                 │                      │
│  ChatPanel (flex: 2)           │  CitationPanel       │
│  ┌───────────────────────────  │  (flex: 1)           │
│  │                             │                      │
│  │  User: MySQL 主从延迟如何排查？│  📚 引用来源         │
│  │                             │  ┌────────────────── │
│  │  AI: 根据检索结果... [1]     │  │ [1] MySQL Runbook │
│  │      ... [2][3]             │  │ "主从延迟常见原因..│
│  │                             │  │  relevance: 0.92  │
│  │                             │  ├────────────────── │
│  │                             │  │ [2] Jira INC-456  │
│  │                             │  │ ...              │
│  ├────────────────────────────  └──────────────────── │
│  │ ▸ Agent Trace (2.3s)                             │
│  │  RetrieveAgent → ReasonAgent                      │
│  ├────────────────────────────────────────────────── │
│  │ [Input box...                        ] [Send]     │
│  └────────────────────────────────────────────────── │
│                                 │                      │
└─────────────────────────────────┴──────────────────────┘
```

### 7.2 色彩语义

| 元素 | 颜色 | 含义 |
|------|------|------|
| 用户气泡 | 蓝色系 | 用户输入 |
| AI 气泡 | 灰色系 | AI 回复 |
| 引用标记 [1] | 绿色 | 可点击溯源 |
| TraceNode-成功 | 绿色 | 阶段完成 |
| TraceNode-进行中 | 蓝色闪烁 | 正在执行 |
| TraceNode-失败 | 红色 | 执行失败 |
| 中断横幅 | 橙色 | 等待用户输入 |
| 工具调用 | 紫色 | 调用外部工具 |
| 置信度 < 0.5 | 红色 | 低置信度 |
| 置信度 0.5-0.7 | 橙色 | 中等置信度 |
| 置信度 > 0.7 | 绿色 | 高置信度 |

---

## 8. Error Boundary 策略

```
<App>
  <ErrorBoundary fallback={<AppCrashFallback />}>
    <ChatPanel>
      <ErrorBoundary fallback={<ChatErrorFallback />}>
        {/* 聊天出错不影响全局 */}
      </ErrorBoundary>
    </ChatPanel>
    <CitationPanel>
      <ErrorBoundary fallback={<CitationErrorFallback />}>
        {/* 溯源出错不影响聊天 */}
      </ErrorBoundary>
    </CitationPanel>
  </ErrorBoundary>
</App>
```

各 fallback 组件：
- **AppCrashFallback**: 显示错误信息 + 刷新按钮
- **ChatErrorFallback**: 消息区域显示 "加载失败，点击重试"
- **CitationErrorFallback**: 面板显示 "溯源数据加载失败"
- 每个 ErrorBoundary 上报错误到 Sentry/console

---

## 9. 性能优化

| 策略 | 实现 |
|------|------|
| **虚拟滚动** | MessageList 使用 `react-virtuoso`，长对话不卡顿 |
| **Markdown 解析防抖** | StreamingContent 50ms debounce，减少 React 渲染 |
| **Citation 懒加载** | 仅在用户点击引用标记时才渲染完整 CitationCard 内容 |
| **Trace 节点折叠** | 超过 10 个节点自动折叠中间节点，只展示首尾 |
| **组件级 memo** | AIBubble / CitationCard 使用 `React.memo` |
| **代码分割** | AgentTrace / InterruptDialog 使用 `React.lazy` |

---

## 10. 目录结构

```
frontend/
├── src/
│   ├── components/
│   │   ├── App.tsx
│   │   ├── Header.tsx
│   │   ├── Layout.tsx
│   │   ├── Chat/
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── MessageList.tsx
│   │   │   ├── UserBubble.tsx
│   │   │   ├── AIBubble.tsx
│   │   │   ├── StreamingContent.tsx
│   │   │   ├── CitationMarker.tsx
│   │   │   ├── ToolCallBadge.tsx
│   │   │   └── InputBox.tsx
│   │   ├── Citation/
│   │   │   ├── CitationPanel.tsx
│   │   │   ├── CitationList.tsx
│   │   │   └── CitationCard.tsx
│   │   ├── AgentTrace/
│   │   │   ├── AgentTrace.tsx
│   │   │   ├── TraceTimeline.tsx
│   │   │   └── TraceNode.tsx
│   │   ├── Interrupt/
│   │   │   └── InterruptDialog.tsx
│   │   ├── Session/
│   │   │   └── SessionSelector.tsx
│   │   └── common/
│   │       ├── Toast.tsx
│   │       └── ErrorBoundary.tsx
│   ├── stores/
│   │   └── chatStore.ts
│   ├── hooks/
│   │   ├── useSSE.ts            // EventSource 管理
│   │   └── useStreamingContent.ts
│   ├── utils/
│   │   ├── eventRouter.ts       // SSE 事件分发
│   │   └── markdown.ts          // Markdown 解析配置
│   ├── types/
│   │   └── index.ts
│   └── main.tsx
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

---

## 11. 接口依赖（对 API 网关层）

| 接口 | 方法 | 用途 |
|------|------|------|
| `/api/query` | GET (SSE) | 主查询，发起 SSE 连接 |
| `/api/resume` | POST | 从中断点恢复 |
| `/api/retrieve` | POST | 仅检索（调试用） |
| `/api/admin/sessions/{session_id}` | GET | 查看 session 状态（调试用） |
| `/api/admin/metrics` | GET | Prometheus 指标 |

---

## 12. 变更日志

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-06-20 | 初始版本 | AI-assisted Design |
