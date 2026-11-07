"""Two deliberately injected failures, evidence and recovery in an isolated demo stack."""

import json
import subprocess
import time
from pathlib import Path

from agent.config import Settings
from agent.orchestrator import Orchestrator
from agent.tools.airflow_tools import AirflowTools
from scripts.collect_demo import save, wait_run
from training.train import validate


def main():
    engine = Orchestrator(Settings())
    directory = Path("evidence/live/failures")
    directory.mkdir(parents=True, exist_ok=True)
    # Failure 1: corrupted input causes validation to stop before any model promotion.
    data = Path("data/iris.csv")
    original = data.read_bytes()
    try:
        data.write_text("wrong_schema\n1\n", encoding="utf-8")
        adapter = AirflowTools(engine.settings)
        run = adapter.trigger()
        terminal = wait_run(engine, run["dag_run_id"])
        tasks = adapter.tasks(run["dag_run_id"])
        logs = adapter.logs(run["dag_run_id"], "validate_dataset", try_number=3)
        if terminal["state"] != "failed":
            raise RuntimeError("Injected corrupt-data run should have failed")
        save(
            directory / "01-invalid-dataset.json",
            {
                "injection": "Replaced dataset with a one-column invalid schema",
                "run": terminal,
                "tasks": tasks,
                "logs": logs,
                "root_cause": "Dataset schema failed validation before preprocessing or training",
                "resolution": "Restore the original DVC-tracked data bytes and revalidate",
            },
        )
    finally:
        data.write_bytes(original)
    validate(output="artifacts/recovered-validation.json")
    # Failure 2: a bad model path keeps the application alive but fails its readiness probe.
    subprocess.run(
        [
            "kubectl",
            "-n",
            "mlops",
            "set",
            "env",
            "deployment/iris-api",
            "MODEL_PATH=/missing/model.json",
        ],
        check=True,
        timeout=30,
    )
    try:
        deadline = time.monotonic() + 120
        health = None
        while time.monotonic() < deadline:
            result = engine.registry.execute("k8s_health", {}, "failure-health")
            if result.status == "ok" and result.data.get("unhealthy_pods"):
                health = result
                break
            time.sleep(3)
        if health is None:
            raise RuntimeError("Injected readiness failure was not observed")
        logs = [
            engine.registry.execute("k8s_logs", {"pod": p["name"]}, "failure-logs").model_dump()
            for p in health.data["unhealthy_pods"][:2]
        ]
        save(
            directory / "02-readiness-failure.json",
            {
                "injection": "Set MODEL_PATH=/missing/model.json in isolated demo namespace",
                "health": health.model_dump(),
                "logs": logs,
                "root_cause": "No model loaded; readiness returns 503 and rollout cannot complete",
                "resolution": "Restore MODEL_PATH=/opt/model/model.json and verify rollout",
            },
        )
    finally:
        subprocess.run(
            [
                "kubectl",
                "-n",
                "mlops",
                "set",
                "env",
                "deployment/iris-api",
                "MODEL_PATH=/opt/model/model.json",
            ],
            check=True,
            timeout=30,
        )
        subprocess.run(
            [
                "kubectl",
                "-n",
                "mlops",
                "rollout",
                "status",
                "deployment/iris-api",
                "--timeout=180s",
            ],
            check=True,
            timeout=190,
        )
    restored = engine.registry.execute("k8s_health", {}, "recovery-health")
    save(
        directory / "recovery.json", {"dataset_restored": True, "kubernetes": restored.model_dump()}
    )
    print(json.dumps({"two_failures_injected_and_recovered": True}))


if __name__ == "__main__":
    main()
