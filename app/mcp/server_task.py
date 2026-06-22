"""
MCP server task: per-server connection lifecycle, tool discovery, tool calling.
Async task pattern (Hermes-style): each server runs as a long-lived asyncio Task.
"""
import asyncio
import logging
from typing import Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

from app.mcp.config import McpServerConfig, StdioConfig, HttpConfig, SseConfig

logger = logging.getLogger(__name__)


class McpServerTask:
    """Manages a single MCP server connection and its tools."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self._tools: list[dict] = []
        self._ready = asyncio.Event()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._transport_ctx = None  # Holds transport context alive
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 2.0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tools(self) -> list[dict]:
        return self._tools

    @property
    def is_connected(self) -> bool:
        return self._ready.is_set() and self.session is not None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        await self._disconnect()

    async def _run_loop(self):
        while self._running:
            try:
                await self._connect()
                self._reconnect_attempts = 0
                self._ready.set()
                logger.info(f"[MCP] {self.name}: connected, {len(self._tools)} tools")
                # Stay connected until disconnection or stop
                while self._running and self.session is not None:
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[MCP] {self.name}: error: {e}")
            finally:
                self._ready.clear()
                await self._cleanup_context()

            if not self._running:
                break

            self._reconnect_attempts += 1
            if self._reconnect_attempts > self._max_reconnect_attempts:
                logger.error(f"[MCP] {self.name}: max reconnects reached")
                break

            delay = min(self._reconnect_delay * (2 ** self._reconnect_attempts), 60)
            logger.info(f"[MCP] {self.name}: reconnecting in {delay}s ({self._reconnect_attempts})")
            await asyncio.sleep(delay)

    async def _connect(self):
        await self._cleanup_context()

        transport_cfg = self.config.transport

        if isinstance(transport_cfg, StdioConfig):
            params = StdioServerParameters(
                command=transport_cfg.command,
                args=transport_cfg.args,
                env=transport_cfg.env or None,
            )
            self._transport_ctx = stdio_client(params)
            read, write = await self._transport_ctx.__aenter__()
            session_ctx = ClientSession(read, write)
            self.session = await session_ctx.__aenter__()
            # Store session context for cleanup
            self._session_ctx = session_ctx

        elif isinstance(transport_cfg, HttpConfig):
            self._transport_ctx = streamablehttp_client(
                transport_cfg.url,
                headers=transport_cfg.headers or None,
            )
            read, write = await self._transport_ctx.__aenter__()
            session_ctx = ClientSession(read, write)
            self.session = await session_ctx.__aenter__()
            self._session_ctx = session_ctx

        elif isinstance(transport_cfg, SseConfig):
            self._transport_ctx = sse_client(
                transport_cfg.url,
                headers=transport_cfg.headers or None,
            )
            read, write = await self._transport_ctx.__aenter__()
            session_ctx = ClientSession(read, write)
            self.session = await session_ctx.__aenter__()
            self._session_ctx = session_ctx

        # Initialize and discover
        await self.session.initialize()
        await self._discover_tools()

    async def _discover_tools(self):
        if not self.session:
            return
        result = await self.session.list_tools()
        raw_tools = result.tools
        if self.config.tool_include:
            raw_tools = [t for t in raw_tools if t.name in self.config.tool_include]
        if self.config.tool_exclude:
            raw_tools = [t for t in raw_tools if t.name not in self.config.tool_exclude]
        self._tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if hasattr(t, 'inputSchema') else {"type": "object", "properties": {}},
            }
            for t in raw_tools
        ]

    async def _cleanup_context(self):
        self.session = None
        if hasattr(self, '_session_ctx'):
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
        if self._transport_ctx:
            try:
                await self._transport_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_ctx = None
        self._tools = []

    async def _disconnect(self):
        self._ready.clear()
        await self._cleanup_context()

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self.session or not self._ready.is_set():
            return f"Error: MCP server '{self.name}' not connected"
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments=arguments),
                timeout=self.config.tool_timeout,
            )
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else "(no output)"
        except asyncio.TimeoutError:
            return f"Error: Tool '{tool_name}' timed out"
        except Exception as e:
            return f"Error: {e}"
