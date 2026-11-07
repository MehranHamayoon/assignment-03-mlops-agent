import argparse
import json
import logging
import secrets
from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException

from agent.config import Settings
from agent.orchestrator import Orchestrator
from agent.schemas import ChatRequest, ChatResponse, ConfirmRequest, Evidence

app = FastAPI(title="Evidence-based MLOps Agent", version="1.0.0")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@lru_cache
def engine() -> Orchestrator:
    return Orchestrator(Settings())


def authenticate(x_agent_key: Annotated[str | None, Header()] = None):
    expected = engine().settings.agent_api_key.get_secret_value()
    if not expected:
        raise HTTPException(503, "Set AGENT_API_KEY before using the agent API")
    if not x_agent_key or not secrets.compare_digest(x_agent_key, expected):
        raise HTTPException(401, "Invalid agent credential")


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/tools", dependencies=[Depends(authenticate)])
def tools():
    return engine().registry.definitions()


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(authenticate)])
def chat(request: ChatRequest):
    return engine().chat(request.message)


@app.post("/actions/confirm", response_model=Evidence, dependencies=[Depends(authenticate)])
def confirm(request: ConfirmRequest):
    import uuid

    try:
        return engine().registry.confirm(
            request.action_id, request.confirmation_text, uuid.uuid4().hex
        )
    except PermissionError as exc:
        raise HTTPException(403, "Confirmation missing, expired or incorrect") from exc


def main():
    parser = argparse.ArgumentParser(description="Query the running MLOps agent")
    parser.add_argument("message", nargs="?")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--confirm", metavar="ACTION_ID")
    args = parser.parse_args()
    settings = Settings()
    headers = {"X-Agent-Key": settings.agent_api_key.get_secret_value()}
    with httpx.Client(timeout=600, headers=headers) as client:
        if args.confirm:
            print(f"Confirm the reviewed action {args.confirm}.")
            text = input(f"Type CONFIRM {args.confirm}: ")
            response = client.post(
                args.url + "/actions/confirm",
                json={"action_id": args.confirm, "confirmation_text": text},
            )
        elif args.message:
            response = client.post(args.url + "/chat", json={"message": args.message})
        else:
            parser.error("Provide a message or --confirm ACTION_ID")
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
