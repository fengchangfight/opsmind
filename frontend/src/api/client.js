let eventSource = null;
let serverErrorReceived = false; // tracks if server already sent an error event
function getToken() {
    return localStorage.getItem('opsmind_token');
}
function handleAuthError(res) {
    if (res.status === 401) {
        localStorage.removeItem('opsmind_token');
        localStorage.removeItem('opsmind_user');
        localStorage.removeItem('opsmind_session_id');
        window.location.reload();
    }
}
function authHeaders() {
    const token = getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
}
export function connectSSE(url, handler) {
    const token = getToken();
    const sep = url.includes('?') ? '&' : '?';
    eventSource = new EventSource(token ? `${url}${sep}_token=${encodeURIComponent(token)}` : url);
    const events = ['agent_start', 'retrieval_result', 'chunk', 'final_answer', 'tool_call', 'tool_result', 'reasoning_step', 'interrupted'];
    events.forEach((eventName) => {
        eventSource.addEventListener(eventName, (e) => {
            try {
                const data = JSON.parse(e.data);
                handler(eventName, data);
            }
            catch {
                handler(eventName, {});
            }
        });
    });
    // Dedicated error listener: track server-sent errors, prevent onerror overwrite
    serverErrorReceived = false;
    eventSource.addEventListener('error', (e) => {
        serverErrorReceived = true;
        try {
            const data = JSON.parse(e.data);
            handler('error', data);
        }
        catch {
            handler('error', {});
        }
    });
    eventSource.onerror = async () => {
        // If server already sent an error event, don't overwrite with generic "连接中断"
        if (serverErrorReceived)
            return;
        // Check if it's an auth failure by trying a quick API call
        try {
            const res = await fetch('/api/sessions', { headers: authHeaders() });
            if (res.status === 401) {
                localStorage.removeItem('opsmind_token');
                localStorage.removeItem('opsmind_user');
                localStorage.removeItem('opsmind_session_id');
                window.location.reload();
                return;
            }
        }
        catch { }
        handler('error', { code: 'SSE_FAILED', message: '连接中断' });
    };
}
export function disconnectSSE() {
    eventSource?.close();
    eventSource = null;
}
export async function fetchSessions() {
    const res = await fetch('/api/sessions', { headers: authHeaders() });
    handleAuthError(res);
    const data = await res.json();
    return data.sessions || [];
}
export async function fetchSession(sessionId) {
    const res = await fetch(`/api/sessions/${sessionId}`, { headers: authHeaders() });
    handleAuthError(res);
    return res.json();
}
export async function deleteSession(sessionId) {
    const res = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE', headers: authHeaders() });
    handleAuthError(res);
}
export async function getMcpStatus() {
    const res = await fetch('/api/mcp/status', { headers: authHeaders() });
    handleAuthError(res);
    return res.json();
}
