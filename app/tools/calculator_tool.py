"""Simple arithmetic calculator."""
from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a simple arithmetic expression. Supports +, -, *, /, **, parentheses."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to evaluate, e.g. '2 + 3 * 4'",
            }
        },
        "required": ["expression"],
    }

    async def execute(self, arguments: dict) -> str:
        expr = arguments.get("expression", "")
        if not expr:
            return "Error: no expression provided"
        # Safe eval: only allow numbers, operators, parens, whitespace
        allowed = set("0123456789+-*/(). **")
        if not all(c in allowed for c in expr.replace(" ", "")):
            return f"Error: expression contains invalid characters: {expr}"
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Error: {e}"
