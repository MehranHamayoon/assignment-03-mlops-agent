import math
import os
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field

from agent.config import Settings
from agent.registry import Registry, Tool
from agent.schemas import EmptyArgs, StrictModel


class RunsArgs(StrictModel):
    experiment_ids: list[str] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=5, ge=1, le=100)
    days: int = Field(default=30, ge=1, le=3650)


class CompareArgs(RunsArgs):
    metric: str = Field(default="f1", pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    direction: Literal["maximize", "minimize"] = "maximize"


class RunArgs(StrictModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class RegistryArgs(StrictModel):
    name: str = Field(default="iris-classifier", pattern=r"^[A-Za-z0-9_.-]{1,100}$")


class MlflowTools:
    def __init__(self, settings: Settings, client=None):
        self.settings, self._client = settings, client

    @property
    def client(self):
        if self._client is None:
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(int(self.settings.tool_timeout) or 1)
            os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
            from mlflow import MlflowClient

            self._client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        return self._client

    @staticmethod
    def serialize(run) -> dict:
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time_ms": run.info.start_time,
            "metrics": {k: v for k, v in run.data.metrics.items() if math.isfinite(v)},
            "parameters": dict(run.data.params),
            "tags": dict(run.data.tags),
        }

    def experiments(self) -> dict:
        items = self.client.search_experiments(max_results=100)
        return {
            "experiments": [{"id": x.experiment_id, "name": x.name} for x in items],
            "truncated": bool(getattr(items, "token", None)),
        }

    def runs(self, experiment_ids=None, limit=5, days=30) -> dict:
        ids = experiment_ids or []
        if not ids:
            exp = self.client.get_experiment_by_name(self.settings.mlflow_experiment)
            if exp is None:
                return {"runs": [], "scope": self.settings.mlflow_experiment}
            ids = [exp.experiment_id]
        if any(not x.isdigit() for x in ids):
            raise ValueError("Invalid experiment identifier")
        after = int((datetime.now(UTC) - timedelta(days=days)).timestamp() * 1000)
        items = self.client.search_runs(
            ids,
            filter_string=f"attributes.start_time >= {after}",
            max_results=limit,
            order_by=["attributes.start_time DESC"],
        )
        return {
            "runs": [self.serialize(x) for x in items],
            "days": days,
            "experiment_ids": ids,
            "selection_scope": f"latest {limit} runs",
        }

    def details(self, run_id: str) -> dict:
        result = self.serialize(self.client.get_run(run_id))
        result["artifacts"] = [
            {"path": x.path, "is_dir": x.is_dir, "size": x.file_size}
            for x in self.client.list_artifacts(run_id)
        ]
        return result

    def compare(self, metric="f1", direction="maximize", **kwargs) -> dict:
        result = self.runs(**kwargs)
        eligible = [
            r for r in result["runs"] if r["status"] == "FINISHED" and metric in r["metrics"]
        ]
        eligible.sort(key=lambda r: r["metrics"][metric], reverse=direction == "maximize")
        result.update(
            {
                "metric": metric,
                "direction": direction,
                "best_run": eligible[0] if eligible else None,
                "ranking": [r["run_id"] for r in eligible],
                "rationale": f"{direction} recorded {metric} among finished runs in this scope.",
                "production_readiness": "Not established by metric ranking; inspect quality gate, "
                "lineage, deployment health, and monitoring separately.",
            }
        )
        return result

    def model_registry(self, name="iris-classifier") -> dict:
        versions = self.client.search_model_versions(f"name = '{name}'", max_results=100)
        return {
            "name": name,
            "versions": [
                {
                    "version": x.version,
                    "run_id": x.run_id,
                    "status": x.status,
                    "aliases": list(x.aliases or []),
                }
                for x in versions
            ],
        }

    def register(self, registry: Registry):
        for name, description, schema, handler in [
            (
                "mlflow_experiments",
                "List experiments using MlflowClient.",
                EmptyArgs,
                self.experiments,
            ),
            (
                "mlflow_runs",
                "List recent training runs with metrics, parameters and DVC tags.",
                RunsArgs,
                self.runs,
            ),
            (
                "mlflow_run",
                "Retrieve run details and list its recorded artifacts.",
                RunArgs,
                self.details,
            ),
            (
                "mlflow_compare",
                "Compare latest runs (including multiple experiment IDs) and "
                "select a metric candidate; does not establish production readiness.",
                CompareArgs,
                self.compare,
            ),
            (
                "mlflow_registry",
                "Read model versions, aliases and source run IDs.",
                RegistryArgs,
                self.model_registry,
            ),
        ]:
            registry.add(Tool(name, description, schema, handler))
