"""Get current date and time."""
from app.tools.base import BaseTool
from datetime import datetime, timezone


class DateTimeTool(BaseTool):
    name = "get_current_time"
    description = "Get the current date and time in ISO 8601 format."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, arguments: dict) -> str:
        return datetime.now(timezone.utc).isoformat()
