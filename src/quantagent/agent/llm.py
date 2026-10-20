"""LLM adapters. The agent loop talks to `LLM.create`, which takes and returns Anthropic
Messages-API shaped dicts, so a scripted fake can stand in for tests and CI."""
import os
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLM(Protocol):
    model: str

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        """-> {"content": [blocks], "stop_reason": str, "usage": {"input_tokens", "output_tokens"}}"""


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        import anthropic  # optional dependency ([agent] extra)

        from ..config import get_settings

        key = get_settings().anthropic_api_key
        if key is None or not key.get_secret_value():
            raise RuntimeError("ANTHROPIC_API_KEY is not set (environment or .env)")
        self.client = anthropic.Anthropic(api_key=key.get_secret_value())
        self.model = model or os.environ.get("QUANTAGENT_AGENT_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=messages, tools=tools)
        return {"content": [b.model_dump(exclude_none=True) for b in r.content], "stop_reason": r.stop_reason,
                "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}}


class ScriptedLLM:
    """Replays a fixed list of responses (for tests). Each item is a list of content blocks or a
    callable(messages) -> list of blocks."""

    def __init__(self, script: list, model: str = "scripted"):
        self.script, self.model, self.calls = list(script), model, []

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.script:
            raise RuntimeError("ScriptedLLM ran out of responses")
        step = self.script.pop(0)
        blocks = step(messages) if callable(step) else step
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return {"content": blocks, "stop_reason": stop, "usage": {"input_tokens": 0, "output_tokens": 0}}
