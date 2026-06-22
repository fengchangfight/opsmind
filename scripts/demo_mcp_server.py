"""
Demo MCP server: echo + sysinfo for testing MCP connectivity.
Run: python scripts/demo_mcp_server.py
"""
import asyncio
import platform
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("opsmind-demo-mcp")


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="echo",
            description="Echoes back the input message",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to echo"}
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="sysinfo",
            description="Returns system information",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "echo":
        msg = arguments.get("message", "")
        return [TextContent(type="text", text=f"Echo: {msg}")]
    elif name == "sysinfo":
        info = f"OS: {platform.system()} {platform.release()}\nPython: {sys.version}\nPlatform: {platform.platform()}"
        return [TextContent(type="text", text=info)]
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
