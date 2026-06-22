import type { Citation } from '../types';

type EventHandler = (event: string, data: any) => void;

let eventSource: EventSource | null = null;

function getToken(): string | null {
  return localStorage.getItem('opsmind_token');
}

export function connectSSE(url: string, handler: EventHandler): void {
  // SSE via EventSource doesn't support custom headers. Workaround: pass token as query param.
  const token = getToken();
  const sep = url.includes('?') ? '&' : '?';
  eventSource = new EventSource(token ? `${url}${sep}_token=${encodeURIComponent(token)}` : url);

  const events = ['agent_start', 'retrieval_result', 'chunk', 'final_answer', 'error'] as const;

  events.forEach((eventName) => {
    eventSource!.addEventListener(eventName, (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        handler(eventName, data);
      } catch {
        handler(eventName, {});
      }
    });
  });

  eventSource.onerror = () => {
    handler('error', { code: 'SSE_FAILED', message: '连接中断' });
  };
}

export function disconnectSSE(): void {
  eventSource?.close();
  eventSource = null;
}

export async function fetchSessions(): Promise<any[]> {
  const res = await fetch('/api/sessions', {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  const data = await res.json();
  return data.sessions || [];
}

export async function fetchSession(sessionId: string): Promise<any> {
  const res = await fetch(`/api/sessions/${sessionId}`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`/api/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${getToken()}` },
  });
}
