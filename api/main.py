import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from training.train import predict_json


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[list[float]] = Field(min_length=1, max_length=100)

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, instances):
        import math

        if any(
            len(row) != 4 or any(not math.isfinite(x) or x <= 0 or x > 100 for x in row)
            for row in instances
        ):
            raise ValueError("Each instance requires four finite measurements in (0, 100]")
        return instances


@asynccontextmanager
async def lifespan(application):
    path = Path(os.environ.get("MODEL_PATH", "artifacts/approved/model.json"))
    application.state.model = None
    if path.exists():
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("format_version") != 1:
            raise ValueError("Unsupported model format")
        application.state.model = model
    yield


app = FastAPI(title="Iris prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    if app.state.model is None:
        raise HTTPException(503, "Approved model is not loaded")
    return {"status": "ready", "model_run_id": app.state.model.get("run_id")}


@app.get("/version")
def version():
    model = app.state.model
    return {
        "app_version": os.environ.get("APP_VERSION", "1.0.0"),
        "run_id": model.get("run_id") if model else None,
        "provenance": model.get("provenance") if model else None,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    ready()
    labels, probabilities = predict_json(app.state.model, request.instances)
    return {
        "predictions": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "model_run_id": app.state.model.get("run_id"),
    }
