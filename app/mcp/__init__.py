"""
MCP (Model Context Protocol) integration framework for OpsMind RAG.

Architecture:
    McpManager (singleton)
    ├── McpServerTask × N (per-server async task)
    │   ├── Transport (stdio / http / sse)
    │   ├── ClientSession (MCP SDK)
    │   └── Tool List (discovered + filtered)
    └── ToolAdapter → OpenAI function calling format

Design reference: Hermes mcp_tool.py, OpenCode mcp/index.ts, Claude Code mcp/client.ts

Usage:
    from app.mcp import McpManager, McpServerConfig, StdioConfig

    manager = McpManager()
    manager.add_server(McpServerConfig(
        name="filesystem",
        transport=StdioConfig(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
    ))
    await manager.start_all()
    tools = manager.get_all_tools()  # OpenAI function calling format
    result = await manager.call_tool("mcp_filesystem_read_file", {"path": "/tmp/test.txt"})
"""
from app.mcp.config import McpServerConfig, StdioConfig, HttpConfig, SseConfig
from app.mcp.manager import McpManager
from app.mcp.tool_adapter import ToolAdapter
from app.mcp.server_task import McpServerTask

__all__ = [
    "McpManager",
    "McpServerConfig",
    "StdioConfig",
    "HttpConfig",
    "SseConfig",
    "ToolAdapter",
    "McpServerTask",
]
