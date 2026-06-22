"""
BaseTool: abstract interface for OpsMind native tools.
Mimics OpenAI function calling schema for registration.

New tools: subclass BaseTool, implement execute(), register with ToolRegistry.
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract native tool. Subclass to create new tools."""

    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool and return a text result."""
        ...

    def to_openai_function(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.get("properties", {}),
                    "required": self.parameters.get("required", []),
                    "additionalProperties": False,
                },
            },
        }
