import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from app.connectors.base import BaseConnector
from app.models import Document


class TxtConnector(BaseConnector):
    connector_name = "txt"
    supported_types = ["txt"]

    async def extract(self, source: str) -> AsyncIterator[Document]:
        path = Path(source)
        if not path.exists():
            return

        for file_path in path.rglob("*.txt"):
            try:
                doc = self._parse_file(file_path)
                if doc:
                    yield doc
            except Exception:
                continue

    def _parse_file(self, file_path: Path) -> Document | None:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            return None

        category = self._infer_category(file_path)
        title = self._extract_title(content) or file_path.stem

        metadata = {
            "filename": file_path.name,
            "category": category,
            "extension": ".txt",
            "size_bytes": file_path.stat().st_size,
        }

        doc_type = self._infer_doc_type(file_path, category)
        if doc_type:
            metadata["doc_type"] = doc_type

        return Document(
            doc_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"txt://{file_path}")),
            source=f"file://{file_path}",
            source_type="txt",
            title=title,
            content=content,
            metadata=metadata,
            created_at=datetime.fromtimestamp(file_path.stat().st_ctime, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc),
        )

    def _infer_category(self, file_path: Path) -> str:
        parts = file_path.parts
        for cat in ["confluence", "github", "gmail", "jira", "hubspot", "google_drive", "fireflies"]:
            if cat in parts:
                return cat
        return "unknown"

    def _infer_doc_type(self, file_path: Path, category: str) -> str:
        path_lower = str(file_path).lower()
        if category == "confluence":
            if any(k in path_lower for k in ["sre", "oncall", "incident"]):
                return "runbook"
        if "incident" in path_lower:
            return "incident_report"
        return ""

    def _extract_title(self, content: str) -> str | None:
        first_line = content.split("\n", 1)[0].strip()
        if first_line and len(first_line) < 200 and not first_line.startswith("---"):
            return first_line
        return None
