"""
MCP server configuration model.
Supports three transport types: stdio, http (streamable), sse.
Reference: Hermes config.yaml MCP section, OpenCode MCP config, Claude Code .mcp.json.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class StdioConfig(BaseModel):
    transport: Literal["stdio"] = "stdio"
    command: str = Field(..., description="Executable path or command (npx, python, etc.)")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Extra environment variables")


class HttpConfig(BaseModel):
    transport: Literal["http"] = "http"
    url: str = Field(..., description="Streamable HTTP endpoint URL")
    headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers (auth, etc.)")
    oauth: Optional["OAuthConfig"] = None


class SseConfig(BaseModel):
    transport: Literal["sse"] = "sse"
    url: str = Field(..., description="SSE endpoint URL")
    headers: dict[str, str] = Field(default_factory=dict)


class OAuthConfig(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    auth_server_url: str = ""
    scope: str = ""


McpTransportConfig = StdioConfig | HttpConfig | SseConfig


class McpServerConfig(BaseModel):
    """Single MCP server configuration."""
    name: str = Field(..., description="Server name (unique identifier)")
    description: str = Field(default="", description="Human-readable description")
    enabled: bool = Field(default=True, description="Auto-connect on startup")
    transport: McpTransportConfig = Field(..., discriminator="transport")
    tool_timeout: int = Field(default=30, description="Per-tool-call timeout in seconds")
    connect_timeout: int = Field(default=15, description="Initial connection timeout in seconds")
    tool_include: list[str] = Field(default_factory=list, description="Allowlist tools (empty = all)")
    tool_exclude: list[str] = Field(default_factory=list, description="Denylist tools")
    supports_parallel: bool = Field(default=True, description="Allow parallel tool calls")
