from datetime import UTC, datetime
from urllib.parse import quote

import httpx
from pydantic import Field

from agent.config import Settings
from agent.registry import Registry, Tool
from agent.schemas import EmptyArgs, StrictModel


class DagArgs(StrictModel):
    dag_id: str = Field(default="iris_training", pattern=r"^[A-Za-z0-9_.-]{1,100}$")


class DagRunArgs(DagArgs):
    run_id: str = Field(min_length=1, max_length=200)


class TaskLogArgs(DagRunArgs):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    try_number: int = Field(default=1, ge=1, le=20)


class TrainingConf(StrictModel):
    c: float = Field(default=1.0, ge=0.0001, le=100)
    max_iter: int = Field(default=300, ge=1, le=1000)
    quality_threshold: float = Field(default=0.85, ge=0.5, le=1.0)


class TriggerArgs(DagArgs):
    conf: TrainingConf = Field(default_factory=TrainingConf)


class AirflowTools:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=settings.airflow_url.rstrip("/"),
            timeout=settings.tool_timeout,
            follow_redirects=False,
        )
        self._token = settings.airflow_token.get_secret_value()

    def authenticate(self):
        response = self.client.post(
            "/auth/token",
            json={
                "username": self.settings.airflow_username,
                "password": self.settings.airflow_password.get_secret_value(),
            },
        )
        response.raise_for_status()
        self._token = response.json()["access_token"]

    def request(self, method: str, path: str, **kwargs):
        if not self._token:
            self.authenticate()
        response = self.client.request(
            method, f"/api/v2{path}", headers={"Authorization": f"Bearer {self._token}"}, **kwargs
        )
        if response.status_code == 401:
            self.authenticate()
            response = self.client.request(
                method,
                f"/api/v2{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                **kwargs,
            )
        response.raise_for_status()
        return response.json()

    def authorize(self, dag_id: str, **_):
        if dag_id != self.settings.approved_dag:
            raise PermissionError("DAG is not allowlisted")

    def dags(self) -> dict:
        result = self.request("GET", "/dags", params={"limit": 100})
        return {
            "dags": [
                {k: x.get(k) for k in ("dag_id", "is_paused", "description")}
                for x in result.get("dags", [])
            ]
        }

    def runs(self, dag_id="iris_training") -> dict:
        return self.request(
            "GET",
            f"/dags/{quote(dag_id, safe='')}/dagRuns",
            params={"limit": 10, "order_by": "-logical_date"},
        )

    def tasks(self, run_id: str, dag_id="iris_training") -> dict:
        path = f"/dags/{quote(dag_id, safe='')}/dagRuns/{quote(run_id, safe='')}/taskInstances"
        result = self.request("GET", path, params={"limit": 100})
        tasks = result.get("task_instances", [])
        return {
            "task_instances": tasks,
            "failed_tasks": [t for t in tasks if t.get("state") in ("failed", "upstream_failed")],
            "total_entries": result.get("total_entries", len(tasks)),
        }

    def logs(self, run_id: str, task_id: str, try_number=1, dag_id="iris_training"):
        path = (
            f"/dags/{quote(dag_id, safe='')}/dagRuns/{quote(run_id, safe='')}"
            f"/taskInstances/{quote(task_id, safe='')}/logs/{try_number}"
        )
        return self.request("GET", path, params={"map_index": -1})

    def trigger(self, dag_id="iris_training", conf=None):
        self.authorize(dag_id)
        return self.request(
            "POST",
            f"/dags/{quote(dag_id, safe='')}/dagRuns",
            json={
                "logical_date": datetime.now(UTC).isoformat(),
                "conf": conf or {},
            },
        )

    def register(self, registry: Registry):
        for name, description, schema, handler, mutation in [
            ("airflow_dags", "List actual Airflow 3 DAGs.", EmptyArgs, self.dags, False),
            (
                "airflow_runs",
                "Get recent DAG run statuses, without starting training.",
                DagArgs,
                self.runs,
                False,
            ),
            (
                "airflow_tasks",
                "Get task execution states and identify failed tasks.",
                DagRunArgs,
                self.tasks,
                False,
            ),
            (
                "airflow_logs",
                "Read available task logs for failure diagnosis.",
                TaskLogArgs,
                self.logs,
                False,
            ),
            (
                "airflow_trigger",
                "Propose an approved training DAG run. Requires explicit "
                "confirmation in the authenticated confirmation endpoint.",
                TriggerArgs,
                self.trigger,
                True,
            ),
        ]:
            registry.add(
                Tool(
                    name,
                    description,
                    schema,
                    handler,
                    mutation,
                    None if schema is EmptyArgs else self.authorize,
                )
            )
