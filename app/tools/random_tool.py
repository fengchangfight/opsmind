"""Generate random numbers."""
import random
from app.tools.base import BaseTool


class RandomTool(BaseTool):
    name = "random_number"
    description = "Generate a random integer between min and max (inclusive)."
    parameters = {
        "type": "object",
        "properties": {
            "min": {"type": "integer", "description": "Minimum value (default 1)"},
            "max": {"type": "integer", "description": "Maximum value (default 100)"},
        },
    }

    async def execute(self, arguments: dict) -> str:
        lo = int(arguments.get("min", 1))
        hi = int(arguments.get("max", 100))
        return str(random.randint(lo, hi))
