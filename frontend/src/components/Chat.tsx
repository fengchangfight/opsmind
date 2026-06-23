import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message, Citation } from '../types';
import { connectSSE, disconnectSSE, fetchSessions, fetchSession, getMcpStatus, deleteSession } from '../api/client';

let msgIdCounter = 0;
function nextId(): string {
  return `msg-${++msgIdCounter}`;
}

interface SessionMeta {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

interface Props {
  onLogout: () => void;
}

const SESSION_KEY = 'opsmind_session_id';

function loadStoredSid(): string | null {
  try { return localStorage.getItem(SESSION_KEY); } catch { return null; }
}

function storeSid(sid: string) {
  try { localStorage.setItem(SESSION_KEY, sid); } catch {}
}

export default function Chat({ onLogout }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [activeAgents, setActiveAgents] = useState<Set<string>>(new Set());
  const [sessionId, setSessionId] = useState<string>(loadStoredSid() || '');
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [mcpStatus, setMcpStatus] = useState<Record<string, any>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  // Load session list + MCP status on mount
  useEffect(() => {
    fetchSessions().then((list) => {
      setSessions(list);
      const cur = list.find((s: SessionMeta) => s.session_id === sessionId);
      if (cur) loadSession(cur.session_id);
      else setLoaded(true);
    });
    getMcpStatus().then((s) => setMcpStatus(s.servers || {}));
  }, []);

  async function loadSession(sid: string) {
    const data = await fetchSession(sid);
    if (data.messages) {
      setMessages(data.messages.map((m: any) => ({
        id: `db-${m.id}`,
        role: m.role,
        content: m.content,
        citations: m.citations || [],
      })));
    }
    setSessionId(sid);
    storeSid(sid);
    setLoaded(true);
  }

  async function handleDelete(sid: string, e: React.MouseEvent) {
    e.stopPropagation();
    await deleteSession(sid);
    if (sessionId === sid) {
      setMessages([]);
      setSessionId('');
      localStorage.removeItem(SESSION_KEY);
    }
    fetchSessions().then(setSessions);
  }

  async function newSession() {
    setMessages([]);
    setCitations([]);
    setSessionId('');
    localStorage.removeItem(SESSION_KEY);
    setLoaded(true);
  }

  const handleSend = useCallback(() => {
    const query = input.trim();
    if (!query || isStreaming) return;

    const userMsg: Message = { id: nextId(), role: 'user', content: query };
    const aiMsg: Message = { id: nextId(), role: 'assistant', content: '', isStreaming: true };

    setMessages((prev) => [...prev, userMsg, aiMsg]);
    setInput('');
    setIsStreaming(true);
    setCitations([]);
    setActiveAgents(new Set());

    let streamContent = '';

    const params = new URLSearchParams();
    params.set('query', query);
    params.set('top_k', '5');
    if (sessionId) params.set('session_id', sessionId);

    const handler = (event: string, data: any) => {
      switch (event) {
        case 'agent_start':
          setActiveAgents((prev) => new Set(prev).add(data.agent_id || 'agent'));
          if (data.session_id && !sessionId) {
            setSessionId(data.session_id);
            storeSid(data.session_id);
          }
          break;

        case 'retrieval_result':
          break;

        case 'chunk':
          streamContent += data.content || '';
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m
            )
          );
          break;

        case 'tool_call':
          streamContent += `\n\n🔧 **调用工具**: \`${data.tool_name}\`\n`;
          if (data.arguments && Object.keys(data.arguments).length > 0) {
            streamContent += `参数: \`${JSON.stringify(data.arguments)}\`\n`;
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m
            )
          );
          break;

        case 'tool_result':
          streamContent += `📋 **结果**: ${(data.result || '').slice(0, 300)}\n\n`;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m
            )
          );
          break;

        case 'reasoning_step':
          const ci = data.confidence;
          const confStr = typeof ci === 'number' && !isNaN(ci) ? `${(ci * 100).toFixed(0)}%` : '';
          const iter = (data.iteration || data.step || 0) + 1;
          const maxIter = data.max_iterations || 3;
          streamContent += `\n🔄 **迭代 ${iter}/${maxIter}**${confStr ? ` — 置信度: ${confStr}` : ''}\n`;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m
            )
          );
          break;

        case 'interrupted':
          streamContent += `\n⏸️ **中断**: ${data.reason}\n`;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m
            )
          );
          break;

        case 'final_answer':
          setCitations(data.citations || []);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id
                ? { ...m, content: data.answer || streamContent, isStreaming: false, citations: data.citations }
                : m
            )
          );
          setIsStreaming(false);
          disconnectSSE();
          // Refresh session list
          fetchSessions().then(setSessions);
          break;

        case 'error':
          if (data.code === 'UNAUTHORIZED' || data.message?.includes('unauthorized')) {
            localStorage.removeItem('opsmind_token');
            localStorage.removeItem('opsmind_user');
            window.location.reload();
            break;
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id
                ? { ...m, content: `❌ 错误: ${data.message || '未知错误'}`, isStreaming: false }
                : m
            )
          );
          setIsStreaming(false);
          disconnectSSE();
          break;
      }
    };

    connectSSE(`/api/query?${params.toString()}`, handler);
  }, [input, isStreaming, sessionId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen">
      {/* Left Sidebar — Sessions */}
      <aside className="w-64 bg-gray-50 border-r flex flex-col shrink-0">
        <div className="px-4 py-3 border-b bg-white">
          <h2 className="text-sm font-semibold text-gray-700">会话</h2>
        </div>
        <div className="px-3 py-2">
          <button
            onClick={newSession}
            className="w-full text-left px-2 py-1.5 text-xs rounded hover:bg-blue-50 text-blue-600 font-medium"
          >
            + 新建会话
          </button>
        </div>
        <div className="flex-1 overflow-y-auto divide-y border-t">
          {sessions.map((s) => (
            <button
              key={s.session_id}
              onClick={() => loadSession(s.session_id)}
              className={`w-full text-left px-3 py-2 text-xs hover:bg-gray-100 flex items-center justify-between ${
                s.session_id === sessionId ? 'bg-blue-50 border-l-2 border-blue-400' : ''
              }`}
            >
              <div className="flex-1 min-w-0">
                <div className="truncate text-gray-700">{s.title || '(新会话)'}</div>
                <div className="text-gray-400 mt-0.5">{s.updated_at?.slice(0, 16) || ''}</div>
              </div>
              <button
                onClick={(e) => handleDelete(s.session_id, e)}
                className="ml-1 px-1 text-gray-300 hover:text-red-500 shrink-0"
                title="删除会话"
              >
                ✕
              </button>
            </button>
          ))}
        </div>
      </aside>

      {/* Main Chat */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="bg-white border-b px-6 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <img src="/opsmindlogo.png" alt="OpsMind" className="h-6 w-6" />
            <div>
              <h1 className="text-lg font-bold text-gray-800">OpsMind RAG</h1>
              <p className="text-xs text-gray-500">Agentic RAG for Enterprise Operations</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* MCP Status */}
            {Object.entries(mcpStatus).map(([name, s]) => (
              <span
                key={name}
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  s.connected ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-400'
                }`}
                title={`MCP: ${name} (${s.tools_count} tools)`}
              >
                {s.connected ? '🔌' : '🔌'} {name}
              </span>
            ))}
            {Array.from(activeAgents).map((a) => (
              <span key={a} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                {a}
              </span>
            ))}
            {isStreaming && (
              <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs font-medium animate-pulse">
                thinking...
              </span>
            )}
            <span className="text-xs text-gray-500">
              {JSON.parse(localStorage.getItem('opsmind_user') || '{}').display_name || ''}
            </span>
            <button
              onClick={onLogout}
              className="px-2 py-0.5 text-xs text-gray-400 hover:text-gray-600 border rounded"
            >
              退出
            </button>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">
              <p className="text-2xl font-bold mb-2">OpsMind RAG</p>
              <p className="text-sm">输入运维问题，AI 将从知识库中检索并回答</p>
              <div className="mt-4 text-xs text-gray-300">
                示例: "How to ensure deterministic evaluation results in CI?" · "What is the retention policy?" · "How does evaluator backpressure work?"
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white border text-gray-800 shadow-sm'
                }`}
              >
                {msg.role === 'assistant' ? (
                  <div className="prose prose-sm max-w-none text-gray-800">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content || (msg.isStreaming ? '▊' : '')}
                    </ReactMarkdown>
                    {msg.isStreaming && msg.content && (
                      <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-0.5" />
                    )}
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t px-6 py-3 shrink-0">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的运维问题... (Enter 发送, Shift+Enter 换行)"
              rows={2}
              disabled={isStreaming}
              className="flex-1 border rounded-lg px-4 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:bg-gray-50"
            />
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              发送
            </button>
          </div>
        </div>
      </div>

      {/* Citation Panel — Right */}
      <aside className="w-72 bg-white border-l overflow-y-auto shrink-0">
        <div className="px-4 py-3 border-b">
          <h2 className="text-sm font-semibold text-gray-700">引用来源</h2>
        </div>
        {citations.length === 0 ? (
          <div className="px-4 py-8 text-center text-gray-400 text-xs">
            检索结果将显示在这里
          </div>
        ) : (
          <div className="divide-y">
            {citations.map((c) => (
              <div key={c.citation_id} className="px-4 py-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-green-600">[{c.citation_id}]</span>
                  <span className="text-xs text-gray-400">
                    {(c.relevance_score * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-xs font-medium text-gray-700 mb-1 truncate" title={c.doc_title}>
                  {c.doc_title}
                </p>
                <p className="text-xs text-gray-500 line-clamp-3">{c.excerpt}</p>
              </div>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}
