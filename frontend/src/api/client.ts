import type { Citation } from '../types';

type EventHandler = (event: string, data: any) => void;

let eventSource: EventSource | null = null;

export function connectSSE(url: string, handler: EventHandler): void {
  eventSource = new EventSource(url);

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
  const res = await fetch('/api/sessions');
  const data = await res.json();
  return data.sessions || [];
}

export async function fetchSession(sessionId: string): Promise<any> {
  const res = await fetch(`/api/sessions/${sessionId}`);
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
}
