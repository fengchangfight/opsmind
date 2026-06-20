import re
from opmind.models import Document, Chunk


class SimpleChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._counter = 0

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current = ""
        current_start = 0
        line_offset = 0

        for para, para_start_line, _ in paragraphs:
            para_tokens = len(self._tokenize(para))

            if para_tokens > self.chunk_size:
                if current.strip():
                    self._counter += 1
                    chunks.append(self._make_chunk(current, current_start, line_offset, document))
                for sub in self._split_long_paragraph(para):
                    self._counter += 1
                    chunks.append(self._make_chunk(sub, para_start_line, para_start_line, document))
                current = ""
                current_start = line_offset + 1
            elif len(self._tokenize(current)) + para_tokens <= self.chunk_size:
                current = (current + "\n" + para).strip() if current else para
            else:
                self._counter += 1
                chunks.append(self._make_chunk(current, current_start, line_offset, document))
                current = para
                current_start = para_start_line

            line_offset = para_start_line + para.count("\n") + 1

        if current.strip():
            self._counter += 1
            chunks.append(self._make_chunk(current, current_start, line_offset, document))

        return chunks

    def _split_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        lines = text.split("\n")
        results = []
        current = ""
        current_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "" and current.strip():
                results.append((current.strip(), current_start, i))
                current = ""
                current_start = i + 1
            elif stripped.startswith("#") and current.strip():
                results.append((current.strip(), current_start, i))
                current = line
                current_start = i
            else:
                current = (current + "\n" + line).strip() if current else line

        if current.strip():
            results.append((current.strip(), current_start, len(lines)))

        return results

    def _split_long_paragraph(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        for s in sentences:
            if len(self._tokenize(current + " " + s)) > self.chunk_size and current:
                chunks.append(current.strip())
                current = s
            else:
                current = (current + " " + s).strip()
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _make_chunk(self, content: str, start: int, end: int, doc: Document) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}-{self._counter}",
            doc_id=doc.doc_id,
            content=content,
            start_line=start,
            end_line=end,
            metadata={
                **doc.metadata,
                "doc_id": doc.doc_id,
                "doc_title": doc.title,
                "title": doc.title,
            },
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\S+", text)
