"""
OpsMind native tool system.

Architecture:
    BaseTool (abstract)
    ├── DateTimeTool    — get_current_time
    ├── CalculatorTool  — arithmetic
    └── RandomTool      — random numbers
    ToolRegistry (singleton)
    └── create_default_registry()  — factory with built-in tools

Adding a new tool:
    1. Create my_tool.py: class MyTool(BaseTool): ...
    2. Register: registry.register(MyTool())

See: docs/LLD_OpsMind_RAG_04_Agent层.md §5 ExecuteAgent for production design (circuit breaker, rate limiting).
"""
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry, create_default_registry

__all__ = ["BaseTool", "ToolRegistry", "create_default_registry"]
