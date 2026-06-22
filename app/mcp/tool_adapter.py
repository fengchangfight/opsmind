"""
Converts MCP tools into OpsMind-compatible tool descriptions.
Used by McpManager to register MCP tools with the ReasonAgent's function calling.
"""
import json
from typing import Any


class ToolAdapter:
    """Adapt MCP tool definitions to OpenAI function calling format."""

    @staticmethod
    def to_openai_function(server_name: str, tool: dict) -> dict:
        """
        Convert an MCP tool to OpenAI function calling schema.
        
        Returns: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        name = ToolAdapter.sanitize_name(server_name, tool["name"])
        schema = tool.get("inputSchema", {"type": "object", "properties": {}})
        
        # Ensure schema is valid for OpenAI function calling
        params = ToolAdapter._normalize_schema(schema)
        params.setdefault("type", "object")
        params["additionalProperties"] = False

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", f"MCP tool: {tool['name']}"),
                "parameters": params,
            },
        }

    @staticmethod
    def sanitize_name(server_name: str, tool_name: str) -> str:
        """
        Generate safe tool name: mcp_{server}_{tool}.
        Reference: Hermes mcp_{server}_{tool}, OpenCode {server}_{tool}.
        """
        safe_server = server_name.replace("-", "_").replace(".", "_")
        safe_tool = tool_name.replace("-", "_").replace(".", "_")
        return f"mcp_{safe_server}_{safe_tool}"

    @staticmethod
    def _normalize_schema(schema: dict) -> dict:
        """Normalize JSON Schema for OpenAI compatibility."""
        result = {}
        if "type" in schema:
            result["type"] = schema["type"]
        if "properties" in schema:
            result["properties"] = {}
            for key, prop in schema["properties"].items():
                result["properties"][key] = ToolAdapter._normalize_property(prop)
        if "required" in schema:
            result["required"] = list(schema["required"])
        return result

    @staticmethod
    def _normalize_property(prop: dict) -> dict:
        """Recursively normalize a single property."""
        result = {}
        if "type" in prop:
            result["type"] = prop["type"]
        if "description" in prop:
            result["description"] = prop["description"]
        if "enum" in prop:
            result["enum"] = list(prop["enum"])
        if "properties" in prop:
            result["properties"] = {}
            for k, v in prop["properties"].items():
                result["properties"][k] = ToolAdapter._normalize_property(v)
        if "items" in prop:
            result["items"] = ToolAdapter._normalize_property(prop["items"])
        return result
