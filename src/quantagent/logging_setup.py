"""JSON-lines logging to stderr (stdout carries the MCP protocol over stdio)."""
import json
import logging
import sys
import time

LOGGER_NAME = "quantagent"
_MAX_STR = 200
_MAX_ITEMS = 20


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["traceback"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_quantagent", False) for h in log.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        h._quantagent = True  # type: ignore[attr-defined]
        log.addHandler(h)
    log.setLevel(level.upper())
    log.propagate = False
    return log


def sanitise(value):
    """Shorten argument values for logs; never logs settings or environment."""
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "…"
    if isinstance(value, dict):
        return {str(k)[:50]: sanitise(v) for k, v in list(value.items())[:_MAX_ITEMS]}
    if isinstance(value, list | tuple):
        return [sanitise(v) for v in list(value)[:_MAX_ITEMS]]
    return value
