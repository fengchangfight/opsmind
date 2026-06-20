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

## 11. 构建与工程化 (Vite + TypeScript)

### 11.1 Vite 配置

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['@shadcn/ui'],
        },
      },
    },
  },
});
```

### 11.2 tsconfig.json 关键配置

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  },
  "include": ["src"]
}
```

### 11.3 Docker 多阶段构建

```dockerfile
# Stage 1: build
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

# Stage 2: serve
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

---

## 12. CSS 方案 (Tailwind CSS + shadcn/ui)

### 12.1 基础配置

```js
// tailwind.config.js
export default {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // 映射到 §7.2 色彩语义
        'user-bubble':   { DEFAULT: '#3B82F6', light: '#DBEAFE' },
        'ai-bubble':     { DEFAULT: '#F3F4F6', dark: '#D1D5DB' },
        'citation':      { DEFAULT: '#10B981' },
        'trace-success': { DEFAULT: '#10B981' },
        'trace-active':  { DEFAULT: '#3B82F6' },
        'trace-error':   { DEFAULT: '#EF4444' },
        'interrupt':     { DEFAULT: '#F59E0B' },
        'tool-call':     { DEFAULT: '#8B5CF6' },
        'confidence-low':    '#EF4444',
        'confidence-medium': '#F59E0B',
        'confidence-high':   '#10B981',
      },
    },
  },
  plugins: [],
};
```

### 12.2 shadcn/ui 组件选用

| shadcn 组件 | 对应 UI 元素 |
|-------------|------------|
| `Card` | AIBubble 容器、CitationCard |
| `Dialog` | InterruptDialog 弹窗 |
| `Select` | SessionSelector 下拉 |
| `Button` | SendButton、ActionButtons |
| `Textarea` | InputBox、InterruptDialog 输入 |
| `ScrollArea` | MessageList 滚动容器 |
| `Tooltip` | CitationMarker hover 提示 |
| `Badge` | ToolCallBadge、TraceNode 状态 |
| `Toast` | 全局通知 |

### 12.3 全局样式变量映射

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --color-user: theme('colors.user-bubble.DEFAULT');
  --color-ai: theme('colors.ai-bubble.DEFAULT');
  --color-citation: theme('colors.citation.DEFAULT');
}
```

---

## 13. 路由设计 (React Router v6)

### 13.1 路由表

| 路径 | 组件 | 认证要求 | 说明 |
|------|------|---------|------|
| `/chat` | `ChatPage` | 需要 | 主对话界面（默认入口） |
| `/chat/:sessionId` | `ChatPage` | 需要 | 恢复指定 session |
| `/admin` | `AdminPage` | 需要 + admin | 调试面板（session 查看、指标） |
| `/login` | `LoginPage` | 不需要 | 认证页面 |

### 13.2 路由配置

```tsx
// src/router.tsx
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AuthGuard } from '@/components/common/AuthGuard';
import { AdminGuard } from '@/components/common/AdminGuard';
import App from '@/components/App';
import ChatPage from '@/pages/ChatPage';
import AdminPage from '@/pages/AdminPage';
import LoginPage from '@/pages/LoginPage';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      {
        path: 'chat',
        element: <AuthGuard><ChatPage /></AuthGuard>,
      },
      {
        path: 'chat/:sessionId',
        element: <AuthGuard><ChatPage /></AuthGuard>,
      },
      {
        path: 'admin',
        element: <AdminGuard><AdminPage /></AdminGuard>,
      },
      { path: 'login', element: <LoginPage /> },
    ],
  },
]);
```

### 13.3 路由守卫

```tsx
// AuthGuard.tsx
function AuthGuard({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((s) => s.token);
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

// AdminGuard.tsx
function AdminGuard({ children }: { children: React.ReactNode }) {
  const role = useAuthStore((s) => s.role);
  if (role !== 'admin') return <Navigate to="/chat" replace />;
  return <>{children}</>;
}
```

---

## 14. API 层封装

### 14.1 fetch 客户端 (client.ts)

```typescript
// src/api/client.ts
const BASE_URL = '/api';

class ApiClient {
  private getHeaders(): HeadersInit {
    const token = useAuthStore.getState().token;
    return {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };
  }

  async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new ApiError(res.status, err.message || 'Request failed');
    }
    return res.json();
  }

  async get<T>(path: string): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new ApiError(res.status, 'Request failed');
    return res.json();
  }
}

export const api = new ApiClient();
```

### 14.2 SSE 客户端 (sse.ts)

```typescript
// src/api/sse.ts
type SSEEventHandler = (event: string, data: unknown) => void;

class SSEClient {
  private es: EventSource | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect(url: string, handler: SSEEventHandler): void {
    this.es = new EventSource(url);

    this.es.onmessage = (event) => {
      try {
        const { event: type, data } = JSON.parse(event.data);
        handler(type, data);
        if (type === 'final_answer' || type === 'error') this.close();
      } catch {
        // 心跳忽略
      }
    };

    this.es.onerror = () => {
      this.reconnectAttempts++;
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.close();
        handler('error', { code: 'SSE_FAILED', message: 'Connection lost, please retry' });
      }
      // 否则依赖 SSE 原生的自动重连
    };
  }

  close(): void {
    this.es?.close();
    this.es = null;
    this.reconnectAttempts = 0;
  }
}

export const sse = new SSEClient();
```

---

## 15. 开发工作流 (MSW + Vitest)

### 15.1 MSW 模拟后端

```typescript
// src/mocks/handlers.ts
import { http, HttpResponse } from 'msw';

export const handlers = [
  // 模拟 SSE 查询接口
  http.get('/api/query', ({ request }) => {
    const url = new URL(request.url);
    const query = url.searchParams.get('query') || '';

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        const events = [
          { event: 'agent_start', data: { agent_id: 'retrieve' } },
          { event: 'retrieval_result', data: { chunks: [], citations: [] } },
          { event: 'agent_start', data: { agent_id: 'reason' } },
          { event: 'final_answer', data: { answer: `Mock answer for: ${query}` } },
        ];
        for (const e of events) {
          controller.enqueue(encoder.encode(`event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`));
        }
        controller.close();
      },
    });

    return new HttpResponse(stream, {
      headers: { 'Content-Type': 'text/event-stream' },
    });
  }),

  // 模拟 retrieve 接口
  http.post('/api/retrieve', async ({ request }) => {
    const body = await request.json() as any;
    return HttpResponse.json({
      query: body.query,
      results: [{ chunk_id: 'mock-1', content: 'Mock chunk', score: 0.95 }],
      latency_ms: 120,
    });
  }),
];
```

```typescript
// src/mocks/browser.ts
import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';
export const worker = setupWorker(...handlers);
```

```typescript
// src/main.tsx (开发时启用)
async function main() {
  if (import.meta.env.DEV) {
    const { worker } = await import('./mocks/browser');
    await worker.start({ onUnhandledRequest: 'bypass' });
  }
  // ... render
}
```

### 15.2 Vitest 测试

```typescript
// src/components/Chat/__tests__/InputBox.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { InputBox } from '../InputBox';

describe('InputBox', () => {
  it('calls onSubmit with trimmed value', () => {
    const onSubmit = vi.fn();
    render(<InputBox onSubmit={onSubmit} disabled={false} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: '  hello  ' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));

    expect(onSubmit).toHaveBeenCalledWith('hello');
  });

  it('disables submit when empty', () => {
    render(<InputBox onSubmit={vi.fn()} disabled={false} />);
    const btn = screen.getByRole('button', { name: /send/i });
    expect(btn).toBeDisabled();
  });

  it('disables submit when streaming', () => {
    render(<InputBox onSubmit={vi.fn()} disabled={true} />);
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'test' } });
    const btn = screen.getByRole('button', { name: /send/i });
    expect(btn).toBeDisabled();
  });
});
```

```yaml
# vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    globals: true,
  },
});
```

---

## 16. 依赖清单 (package.json)

```json
{
  "name": "opsmind-rag-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "lint": "eslint src --ext .ts,.tsx --fix"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.23.1",
    "zustand": "^4.5.2",
    "react-virtuoso": "^4.7.10",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0"
  },
  "devDependencies": {
    "typescript": "^5.4.5",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "vite": "^5.2.11",
    "@vitejs/plugin-react": "^4.3.0",
    "tailwindcss": "^3.4.3",
    "postcss": "^8.4.38",
    "autoprefixer": "^10.4.19",
    "@shadcn/ui": "^0.8.0",
    "msw": "^2.3.0",
    "vitest": "^1.6.0",
    "@testing-library/react": "^15.0.7",
    "@testing-library/jest-dom": "^6.4.5",
    "jsdom": "^24.1.0",
    "eslint": "^8.57.0",
    "@typescript-eslint/eslint-plugin": "^7.8.0",
    "@typescript-eslint/parser": "^7.8.0"
  }
}
```

---

## 17. 接口依赖（对 API 网关层）

| 接口 | 方法 | 用途 |
|------|------|------|
| `/api/query` | GET (SSE) | 主查询，发起 SSE 连接 |
| `/api/resume` | POST | 从中断点恢复 |
| `/api/retrieve` | POST | 仅检索（调试用） |
| `/api/admin/sessions/{session_id}` | GET | 查看 session 状态（调试用） |
| `/api/admin/metrics` | GET | Prometheus 指标 |

---

## 18. 变更日志

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-06-20 | 初始版本 | AI-assisted Design |
| v1.1 | 2026-06-20 | 补充构建工程化、CSS方案、路由设计、API封装、MSW开发工作流、依赖清单 | AI-assisted Design |
