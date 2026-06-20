export interface Citation {
  citation_id: string;
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  excerpt: string;
  relevance_score: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
}

export interface SSEMessage {
  event: string;
  data: any;
}
