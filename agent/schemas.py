from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmptyArgs(StrictModel):
    pass


class Evidence(BaseModel):
    tool: str
    status: Literal["ok", "error", "confirmation_required"]
    data: Any = None
    error: str | None = None
    request_id: str
    duration_ms: float


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=6000)


class ChatResponse(BaseModel):
    request_id: str
    facts: list[dict[str, Any]]
    uncertainties: list[str]
    recommendations: list[str]
    evidence: list[Evidence]


class ConfirmRequest(StrictModel):
    action_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    confirmation_text: str = Field(max_length=100)
