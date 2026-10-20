"""Tool error type. Subclasses the MCP SDK's ToolError so the call returns is_error=True
with our JSON payload as the text the model reads."""
import json
from typing import Any, Literal

from mcp.server.mcpserver.exceptions import ToolError

ErrorCode = Literal["INVALID_INPUT", "NOT_FOUND", "INSUFFICIENT_HISTORY", "NO_OVERLAP", "DATA_QUALITY",
                    "MODEL_UNAVAILABLE", "INTERNAL"]


class QuantError(ToolError):
    def __init__(self, code: ErrorCode, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(json.dumps({"code": code, "message": message, "details": self.details}, default=str))
