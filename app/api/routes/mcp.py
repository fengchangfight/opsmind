from fastapi import APIRouter, Request

router = APIRouter(tags=["mcp"])


@router.get("/mcp/status")
async def mcp_status(request: Request):
    """Get MCP server connection status."""
    mcp_manager = request.app.state.runtime.get("mcp")
    if not mcp_manager:
        return {"servers": {}, "message": "MCP not initialized"}
    return {"servers": mcp_manager.status()}
