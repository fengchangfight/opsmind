import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { connectSSE, disconnectSSE, fetchSessions, fetchSession, getMcpStatus, deleteSession } from '../api/client';
let msgIdCounter = 0;
function nextId() {
    return `msg-${++msgIdCounter}`;
}
const SESSION_KEY = 'opsmind_session_id';
function loadStoredSid() {
    try {
        return localStorage.getItem(SESSION_KEY);
    }
    catch {
        return null;
    }
}
function storeSid(sid) {
    try {
        localStorage.setItem(SESSION_KEY, sid);
    }
    catch { }
}
export default function Chat({ onLogout }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [citations, setCitations] = useState([]);
    const [activeAgents, setActiveAgents] = useState(new Set());
    const [sessionId, setSessionId] = useState(loadStoredSid() || '');
    const [sessions, setSessions] = useState([]);
    const [loaded, setLoaded] = useState(false);
    const [mcpStatus, setMcpStatus] = useState({});
    const messagesEndRef = useRef(null);
    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);
    useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);
    // Load session list + MCP status on mount
    useEffect(() => {
        fetchSessions().then((list) => {
            setSessions(list);
            const cur = list.find((s) => s.session_id === sessionId);
            if (cur)
                loadSession(cur.session_id);
            else
                setLoaded(true);
        });
        getMcpStatus().then((s) => setMcpStatus(s.servers || {}));
    }, []);
    async function loadSession(sid) {
        const data = await fetchSession(sid);
        if (data.messages) {
            setMessages(data.messages.map((m) => ({
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
    async function handleDelete(sid, e) {
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
        if (!query || isStreaming)
            return;
        const userMsg = { id: nextId(), role: 'user', content: query };
        const aiMsg = { id: nextId(), role: 'assistant', content: '', isStreaming: true };
        setMessages((prev) => [...prev, userMsg, aiMsg]);
        setInput('');
        setIsStreaming(true);
        setCitations([]);
        setActiveAgents(new Set());
        let streamContent = '';
        const params = new URLSearchParams();
        params.set('query', query);
        params.set('top_k', '5');
        if (sessionId)
            params.set('session_id', sessionId);
        const handler = (event, data) => {
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
                    setMessages((prev) => prev.map((m) => m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m));
                    break;
                case 'tool_call':
                    streamContent += `\n\n🔧 **调用工具**: \`${data.tool_name}\`\n`;
                    if (data.arguments && Object.keys(data.arguments).length > 0) {
                        streamContent += `参数: \`${JSON.stringify(data.arguments)}\`\n`;
                    }
                    setMessages((prev) => prev.map((m) => m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m));
                    break;
                case 'tool_result':
                    streamContent += `📋 **结果**: ${(data.result || '').slice(0, 300)}\n\n`;
                    setMessages((prev) => prev.map((m) => m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m));
                    break;
                case 'reasoning_step':
                    const ci = data.confidence;
                    const confStr = typeof ci === 'number' && !isNaN(ci) ? `${(ci * 100).toFixed(0)}%` : '';
                    const rawIter = data.iteration ?? data.step ?? 0;
                    const rawMax = data.max_iterations ?? 0;
                    const iter = rawIter + 1;
                    const maxIter = rawMax + 1;
                    const msg = data.message ? ` — ${data.message}` : '';
                    // Replace previous reasoning line (single-line overwrite)
                    streamContent = streamContent.split('\n').filter(l => !l.includes('🔄 **迭代')).join('\n');
                    streamContent += `\n🔄 **迭代 ${iter}/${maxIter}**${confStr ? ` — 置信度: ${confStr}` : ''}${msg}\n`;
                    setMessages((prev) => prev.map((m) => m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m));
                    break;
                case 'interrupted':
                    streamContent += `\n⏸️ **中断**: ${data.reason}\n`;
                    setMessages((prev) => prev.map((m) => m.id === aiMsg.id ? { ...m, content: streamContent, isStreaming: true } : m));
                    break;
                case 'final_answer':
                    setCitations(data.citations || []);
                    setMessages((prev) => prev.map((m) => m.id === aiMsg.id
                        ? { ...m, content: data.answer || streamContent, isStreaming: false, citations: data.citations }
                        : m));
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
                    setMessages((prev) => prev.map((m) => m.id === aiMsg.id
                        ? { ...m, content: `❌ 错误: ${data.message || '未知错误'}`, isStreaming: false }
                        : m));
                    setIsStreaming(false);
                    disconnectSSE();
                    break;
            }
        };
        connectSSE(`/api/query?${params.toString()}`, handler);
    }, [input, isStreaming, sessionId]);
    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };
    return (_jsxs("div", { className: "flex h-screen", children: [_jsxs("aside", { className: "w-64 bg-gray-50 border-r flex flex-col shrink-0", children: [_jsx("div", { className: "px-4 py-3 border-b bg-white", children: _jsx("h2", { className: "text-sm font-semibold text-gray-700", children: "\u4F1A\u8BDD" }) }), _jsx("div", { className: "px-3 py-2", children: _jsx("button", { onClick: newSession, className: "w-full text-left px-2 py-1.5 text-xs rounded hover:bg-blue-50 text-blue-600 font-medium", children: "+ \u65B0\u5EFA\u4F1A\u8BDD" }) }), _jsx("div", { className: "flex-1 overflow-y-auto divide-y border-t", children: sessions.map((s) => (_jsxs("button", { onClick: () => loadSession(s.session_id), className: `w-full text-left px-3 py-2 text-xs hover:bg-gray-100 flex items-center justify-between ${s.session_id === sessionId ? 'bg-blue-50 border-l-2 border-blue-400' : ''}`, children: [_jsxs("div", { className: "flex-1 min-w-0", children: [_jsx("div", { className: "truncate text-gray-700", children: s.title || '(新会话)' }), _jsx("div", { className: "text-gray-400 mt-0.5", children: s.updated_at?.slice(0, 16) || '' })] }), _jsx("button", { onClick: (e) => handleDelete(s.session_id, e), className: "ml-1 px-1 text-gray-300 hover:text-red-500 shrink-0", title: "\u5220\u9664\u4F1A\u8BDD", children: "\u2715" })] }, s.session_id))) })] }), _jsxs("div", { className: "flex-1 flex flex-col min-w-0", children: [_jsxs("header", { className: "bg-white border-b px-6 py-3 flex items-center justify-between shrink-0", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("img", { src: "/opsmindlogo.png", alt: "OpsMind", className: "h-6 w-6" }), _jsxs("div", { children: [_jsx("h1", { className: "text-lg font-bold text-gray-800", children: "OpsMind RAG" }), _jsx("p", { className: "text-xs text-gray-500", children: "Agentic RAG for Enterprise Operations" })] })] }), _jsxs("div", { className: "flex items-center gap-3", children: [Object.entries(mcpStatus).map(([name, s]) => (_jsxs("span", { className: `px-2 py-0.5 rounded text-xs font-medium ${s.connected ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-400'}`, title: `MCP: ${name} (${s.tools_count} tools)`, children: [s.connected ? '🔌' : '🔌', " ", name] }, name))), Array.from(activeAgents).map((a) => (_jsx("span", { className: "px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium", children: a }, a))), isStreaming && (_jsx("span", { className: "px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs font-medium animate-pulse", children: "thinking..." })), _jsx("span", { className: "text-xs text-gray-500", children: JSON.parse(localStorage.getItem('opsmind_user') || '{}').display_name || '' }), _jsx("button", { onClick: onLogout, className: "px-2 py-0.5 text-xs text-gray-400 hover:text-gray-600 border rounded", children: "\u9000\u51FA" })] })] }), _jsxs("div", { className: "flex-1 overflow-y-auto px-6 py-4 space-y-4", children: [messages.length === 0 && (_jsxs("div", { className: "text-center text-gray-400 mt-20", children: [_jsx("p", { className: "text-2xl font-bold mb-2", children: "OpsMind RAG" }), _jsx("p", { className: "text-sm", children: "\u8F93\u5165\u8FD0\u7EF4\u95EE\u9898\uFF0CAI \u5C06\u4ECE\u77E5\u8BC6\u5E93\u4E2D\u68C0\u7D22\u5E76\u56DE\u7B54" }), _jsx("div", { className: "mt-4 text-xs text-gray-300", children: "\u793A\u4F8B: \"How to ensure deterministic evaluation results in CI?\" \u00B7 \"What is the retention policy?\" \u00B7 \"How does evaluator backpressure work?\"" })] })), messages.map((msg) => (_jsx("div", { className: `flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`, children: _jsx("div", { className: `max-w-[80%] rounded-lg px-4 py-3 ${msg.role === 'user'
                                        ? 'bg-blue-500 text-white'
                                        : 'bg-white border text-gray-800 shadow-sm'}`, children: msg.role === 'assistant' ? (_jsxs("div", { className: "prose prose-sm max-w-none text-gray-800", children: [_jsx(ReactMarkdown, { remarkPlugins: [remarkGfm], children: msg.content || (msg.isStreaming ? '▊' : '') }), msg.isStreaming && msg.content && (_jsx("span", { className: "inline-block w-2 h-4 bg-blue-500 animate-pulse ml-0.5" }))] })) : (_jsx("p", { className: "whitespace-pre-wrap text-sm", children: msg.content })) }) }, msg.id))), _jsx("div", { ref: messagesEndRef })] }), _jsx("div", { className: "bg-white border-t px-6 py-3 shrink-0", children: _jsxs("div", { className: "flex gap-2", children: [_jsx("textarea", { value: input, onChange: (e) => setInput(e.target.value), onKeyDown: handleKeyDown, placeholder: "\u8F93\u5165\u4F60\u7684\u8FD0\u7EF4\u95EE\u9898... (Enter \u53D1\u9001, Shift+Enter \u6362\u884C)", rows: 2, disabled: isStreaming, className: "flex-1 border rounded-lg px-4 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:bg-gray-50" }), _jsx("button", { onClick: handleSend, disabled: isStreaming || !input.trim(), className: "px-6 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors", children: "\u53D1\u9001" })] }) })] }), _jsxs("aside", { className: "w-72 bg-white border-l overflow-y-auto shrink-0", children: [_jsx("div", { className: "px-4 py-3 border-b", children: _jsx("h2", { className: "text-sm font-semibold text-gray-700", children: "\u5F15\u7528\u6765\u6E90" }) }), citations.length === 0 ? (_jsx("div", { className: "px-4 py-8 text-center text-gray-400 text-xs", children: "\u68C0\u7D22\u7ED3\u679C\u5C06\u663E\u793A\u5728\u8FD9\u91CC" })) : (_jsx("div", { className: "divide-y", children: citations.map((c) => (_jsxs("div", { className: "px-4 py-3", children: [_jsxs("div", { className: "flex items-center justify-between mb-1", children: [_jsxs("span", { className: "text-xs font-bold text-green-600", children: ["[", c.citation_id, "]"] }), _jsxs("span", { className: "text-xs text-gray-400", children: [(c.relevance_score * 100).toFixed(0), "%"] })] }), _jsx("p", { className: "text-xs font-medium text-gray-700 mb-1 truncate", title: c.doc_title, children: c.doc_title }), _jsx("p", { className: "text-xs text-gray-500 line-clamp-3", children: c.excerpt })] }, c.citation_id))) }))] })] }));
}
