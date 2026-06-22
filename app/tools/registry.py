"""
ToolRegistry: centralized tool registration, discovery, execution.
Supports native tools (BaseTool) + external MCP tools.
"""
from typing import Optional
from app.tools.base import BaseTool
from app.tools.datetime_tool import DateTimeTool
from app.tools.calculator_tool import CalculatorTool
from app.tools.random_tool import RandomTool


class ToolRegistry:
    """Central registry for all OpsMind native tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a native tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_all_openai_functions(self) -> list[dict]:
        """Return all tools in OpenAI function calling format."""
        return [t.to_openai_function() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> str:
        """Execute a native tool by name, with timeout protection."""
        import asyncio
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"
        try:
            return await asyncio.wait_for(
                tool.execute(arguments),
                timeout=tool.timeout,
            )
        except asyncio.TimeoutError:
            return f"Error: Tool '{name}' timed out after {tool.timeout}s"
        except Exception as e:
            return f"Error executing '{name}': {e}"

    @property
    def count(self) -> int:
        return len(self._tools)


def create_default_registry() -> ToolRegistry:
    """Factory: create a ToolRegistry with default built-in tools."""
    registry = ToolRegistry()
    registry.register(DateTimeTool())
    registry.register(CalculatorTool())
    registry.register(RandomTool())
    return registry
