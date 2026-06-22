"""
BaseTool: abstract interface for OpsMind native tools.
Mimics OpenAI function calling schema for registration.

New tools: subclass BaseTool, implement execute(), register with ToolRegistry.

Production features (Demo defaults):
- permissions: list of roles allowed to call this tool (empty = all)
- timeout: per-tool execution timeout (default 30s)
- parallel_safe: can be executed in parallel with other tools (default True)
"""
from abc import ABC, abstractmethod
from typing import Any


class PermissionChecker(ABC):
    """Abstract permission checker. Swap implementations for production."""

    @abstractmethod
    def check(self, tool_name: str, user_context: dict | None) -> bool:
        """Return True if user is allowed to use this tool."""
        ...


class AllowAllPermissionChecker(PermissionChecker):
    """Demo default: allow every tool for every user."""

    def check(self, tool_name: str, user_context: dict | None) -> bool:
        return True


class CircuitBreaker:
    """
    Production circuit breaker. Tracks consecutive failures.
    After failure_threshold consecutive failures, opens circuit.
    After recovery_timeout seconds, moves to half-open.
    Pattern: closed → open → half_open → closed.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0

    @property
    def is_open(self) -> bool:
        import time
        if self.state == self.CLOSED:
            return False
        if self.state == self.OPEN:
            if self.last_failure_time and (time.monotonic() - self.last_failure_time) >= self.recovery_timeout:
                self.state = self.HALF_OPEN
                return False
            return True
        return False  # HALF_OPEN → allow probe

    def on_success(self):
        self.failure_count = 0
        self.state = self.CLOSED

    def on_failure(self):
        import time
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN


class BaseTool(ABC):
    """Abstract native tool with production-ready guards."""

    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}, "required": []}
    timeout: int = 30
    parallel_safe: bool = True
    permissions: list[str] = []  # Empty = all roles allowed

    # Per-instance circuit breaker (shared across calls to same tool)
    circuit_breaker: CircuitBreaker | None = None

    def _get_circuit_breaker(self) -> CircuitBreaker:
        if self.circuit_breaker is None:
            self.circuit_breaker = CircuitBreaker()
        return self.circuit_breaker

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool and return a text result."""
        ...

    def to_openai_function(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.get("properties", {}),
                    "required": self.parameters.get("required", []),
                    "additionalProperties": False,
                },
            },
        }
