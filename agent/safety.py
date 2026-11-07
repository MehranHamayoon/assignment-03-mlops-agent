import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any


def redact(value: Any, secret_values: tuple[str, ...] = ()) -> Any:
    """Redact credential keys and configured secrets, including untrusted service logs."""
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if re.search(r"password|secret|token|authorization|api.?key", str(k), re.I)
            else redact(v, secret_values)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(x, secret_values) for x in value]
    if isinstance(value, str):
        for secret in secret_values:
            if len(secret) >= 4:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)Bearer\s+[\w.\-/+=]+", "Bearer [REDACTED]", value)
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
        value = re.sub(
            r"(?i)((?:password|token|api_key|secret)\s*[=:]\s*)[^\s,;]+",
            r"\1[REDACTED]",
            value,
        )
    return value


@dataclass
class PendingAction:
    tool: str
    arguments: dict
    expires_at: float


class ConfirmationStore:
    """Server-side, one-use confirmations. The LLM cannot provide confirmation flags."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._actions: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def propose(self, tool: str, arguments: dict) -> dict:
        with self._lock:
            now = time.monotonic()
            self._actions = {k: v for k, v in self._actions.items() if v.expires_at > now}
            if len(self._actions) >= 100:
                raise PermissionError("Too many pending actions")
            action_id = secrets.token_hex(16)
            self._actions[action_id] = PendingAction(tool, arguments, now + self.ttl)
        return {
            "action_id": action_id,
            "tool": tool,
            "arguments": arguments,
            "expires_in_seconds": self.ttl,
            "required_confirmation": f"CONFIRM {action_id}",
            "instruction": "Review this exact action; submit it to POST /actions/confirm.",
        }

    def consume(self, action_id: str, text: str) -> PendingAction:
        with self._lock:
            action = self._actions.get(action_id)
            if not action or action.expires_at <= time.monotonic():
                raise PermissionError("Unknown or expired action")
            if not secrets.compare_digest(text, f"CONFIRM {action_id}"):
                raise PermissionError("Explicit confirmation does not match")
            return self._actions.pop(action_id)
