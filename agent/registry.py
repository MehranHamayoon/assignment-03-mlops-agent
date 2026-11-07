import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from agent.safety import ConfirmationStore, redact
from agent.schemas import Evidence

logger = logging.getLogger("mlops.tools")


@dataclass
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    handler: Callable[..., Any]
    mutation: bool = False
    authorize: Callable[..., None] | None = None


class Registry:
    def __init__(self, secrets: tuple[str, ...] = ()):
        self.tools: dict[str, Tool] = {}
        self.confirmations = ConfirmationStore()
        self.secrets = secrets

    def add(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError("Duplicate tool")
        self.tools[tool.name] = tool

    def definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema.model_json_schema(),
                },
            }
            for t in self.tools.values()
        ]

    def execute(self, name: str, arguments: dict, request_id: str, *, confirmed=False) -> Evidence:
        started = time.monotonic()
        status, data, error = "ok", None, None
        try:
            if name not in self.tools:
                raise ValueError("Unknown tool")
            tool = self.tools[name]
            args = tool.schema.model_validate(arguments).model_dump()
            if tool.authorize:
                tool.authorize(**args)
            if tool.mutation and not confirmed:
                status = "confirmation_required"
                data = self.confirmations.propose(name, args)
            else:
                data = redact(tool.handler(**args), self.secrets)
        except (ValidationError, ValueError, KeyError):
            status, error = "error", "invalid_request"
        except PermissionError:
            status, error = "error", "permission_denied"
        except (TimeoutError, subprocess.TimeoutExpired, httpx.TimeoutException):
            status, error = "error", "tool_timeout"
        except Exception:
            # Exception bodies may contain credentials or raw infrastructure responses.
            status, error = "error", "service_unavailable"
        elapsed = round((time.monotonic() - started) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "tool": name if name in self.tools else "unknown",
                    "status": status,
                    "duration_ms": elapsed,
                    "request_id": request_id,
                }
            )
        )
        return Evidence(
            tool=name,
            status=status,
            data=data,
            error=error,
            request_id=request_id,
            duration_ms=elapsed,
        )

    def confirm(self, action_id: str, text: str, request_id: str) -> Evidence:
        action = self.confirmations.consume(action_id, text)
        return self.execute(action.tool, action.arguments, request_id, confirmed=True)
