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

export async function postQuery(query: string, topK: number = 5): Promise<Response> {
  return fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
}

export async function postRetrieve(query: string, topK: number = 10) {
  const res = await fetch('/api/retrieve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
  return res.json();
}

export async function postResume(sessionId: string, humanInput: string, option: string = 'continue') {
  return fetch('/api/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, human_input: humanInput, option }),
  });
}
