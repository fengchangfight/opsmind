"""
ToolRegistry: centralized tool registration, discovery, execution.
Supports native tools (BaseTool) + external MCP tools.

Production features:
- PermissionChecker: pluggable RBAC (demo: AllowAll)
- CircuitBreaker: per-tool failure tracking (demo: inactive until used)
- Timeout: asyncio.wait_for with per-tool timeout
"""
from typing import Optional
from app.tools.base import BaseTool, PermissionChecker, AllowAllPermissionChecker
from app.tools.datetime_tool import DateTimeTool
from app.tools.calculator_tool import CalculatorTool
from app.tools.random_tool import RandomTool


class ToolRegistry:
    """Central registry for all OpsMind native tools."""

    def __init__(self, permission_checker: PermissionChecker | None = None):
        self._tools: dict[str, BaseTool] = {}
        self._permission_checker = permission_checker or AllowAllPermissionChecker()

    def set_permission_checker(self, checker: PermissionChecker):
        """Swap permission model (e.g., RBAC for production)."""
        self._permission_checker = checker

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

    async def execute(self, name: str, arguments: dict, user_context: dict | None = None) -> str:
        """
        Execute a native tool with permission check, circuit breaker, and timeout.

        Args:
            name: tool name
            arguments: tool arguments
            user_context: optional user context for permission check (dict with 'role' key)
        """
        import asyncio
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        # Permission check
        if not self._permission_checker.check(name, user_context):
            return f"Error: Permission denied for tool '{name}'"

        # Circuit breaker check
        cb = tool._get_circuit_breaker()
        if cb.is_open:
            return f"Error: Circuit breaker open for tool '{name}' (too many failures)"

        try:
            result = await asyncio.wait_for(tool.execute(arguments), timeout=tool.timeout)
            cb.on_success()
            return result
        except asyncio.TimeoutError:
            cb.on_failure()
            return f"Error: Tool '{name}' timed out after {tool.timeout}s"
        except Exception as e:
            cb.on_failure()
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
