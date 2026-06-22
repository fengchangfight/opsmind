"""
MCP Manager: singleton orchestrating all MCP server connections.
Exposes tools to ReasonAgent for function calling.
Reference: Hermes MCP startup + tool registration, OpenCode MCP service.
"""
import asyncio
import logging
from typing import Optional

from app.mcp.config import McpServerConfig
from app.mcp.server_task import McpServerTask
from app.mcp.tool_adapter import ToolAdapter

logger = logging.getLogger(__name__)


class McpManager:
    """Central MCP server manager."""

    def __init__(self):
        self._servers: dict[str, McpServerTask] = {}
        self._started = False

    # ── Server Lifecycle ────────────────────────────────────

    def add_server(self, config: McpServerConfig):
        """Add a server configuration (does not connect yet)."""
        if config.name in self._servers:
            logger.warning(f"[MCP] Server '{config.name}' already registered")
            return
        self._servers[config.name] = McpServerTask(config)
        logger.info(f"[MCP] Registered server: {config.name} ({config.transport.transport})")

    async def start_all(self):
        """Connect all enabled servers."""
        if self._started:
            return
        self._started = True

        for name, task in self._servers.items():
            if task.config.enabled:
                logger.info(f"[MCP] Starting server: {name}")
                await task.start()

        # Wait for connections (with timeout)
        await self._wait_for_connections(timeout=15)

    async def stop_all(self):
        """Disconnect all servers (fast shutdown)."""
        self._started = False
        import asyncio
        # Stop in parallel with timeout
        tasks = [asyncio.create_task(t.stop()) for t in list(self._servers.values())]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=2.0)
            for t in pending:
                t.cancel()
        self._servers.clear()
        logger.info("[MCP] All servers stopped")

    async def _wait_for_connections(self, timeout: float):
        """Wait for all enabled servers to be ready."""
        deadline = asyncio.get_event_loop().time() + timeout
        for task in self._servers.values():
            if not task.config.enabled:
                continue
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(task._ready.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning(f"[MCP] Server '{task.name}' connection timed out")

    async def restart_server(self, name: str):
        """Restart a single server."""
        task = self._servers.get(name)
        if task:
            await task.stop()
            await task.start()

    # ── Tool Access ─────────────────────────────────────────

    def get_server(self, name: str) -> Optional[McpServerTask]:
        return self._servers.get(name)

    def get_all_tools(self) -> list[dict]:
        """Get all connected tools in OpenAI function calling format."""
        tools = []
        for task in self._servers.values():
            if task.is_connected:
                for tool in task.tools:
                    tools.append(ToolAdapter.to_openai_function(task.name, tool))
        return tools

    def get_all_tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self.get_all_tools()]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Execute an MCP tool by name.
        
        Tool name format: mcp_{server}_{tool}
        Parses the name to find the server and original tool name.
        """
        if not tool_name.startswith("mcp_"):
            return f"Error: Unknown tool namespace for '{tool_name}'"

        # Parse: mcp_{server}_{tool_name}
        # Server name may contain underscores, so find the server from registered names
        for server_name, task in self._servers.items():
            prefix = ToolAdapter.sanitize_name(server_name, "")
            if tool_name.startswith(prefix):
                original_tool = tool_name[len(prefix):]
                return await task.call_tool(original_tool, arguments)

        return f"Error: No MCP server found for tool '{tool_name}'"

    def status(self) -> dict:
        """Get status of all servers."""
        return {
            name: {
                "connected": task.is_connected,
                "tools_count": len(task.tools),
                "server_name": name,
            }
            for name, task in self._servers.items()
        }
