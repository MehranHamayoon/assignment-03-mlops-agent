"""Exercise five conversations against real services. Never substitutes mocked results."""

import argparse
import json
import logging
import time
from pathlib import Path

from agent.config import Settings
from agent.orchestrator import Orchestrator, render_response


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def wait_run(engine, run_id, timeout=420):
    from agent.tools.airflow_tools import AirflowTools

    adapter = AirflowTools(engine.settings)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runs = adapter.runs(engine.settings.approved_dag)
        found = next((r for r in runs.get("dag_runs", []) if r["dag_run_id"] == run_id), None)
        if found and found.get("state") in ("success", "failed"):
            return found
        time.sleep(3)
    raise TimeoutError("Airflow run did not reach a terminal state within the demonstration budget")


def conversation(engine, message, path, required):
    response = engine.chat(message)
    save(
        path,
        {"mode": "live_llm_and_real_tools", "user": message, "response": response.model_dump()},
    )
    successful = {e.tool for e in response.evidence if e.status == "ok"}
    if not required.issubset(successful):
        raise RuntimeError(
            f"Conversation did not obtain required real evidence: {required - successful}"
        )
    print(f"Verified {Path(path).name}: {', '.join(sorted(successful))}", flush=True)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evidence/live")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    audit = logging.getLogger("mlops.tools")
    audit.setLevel(logging.INFO)
    handler = logging.FileHandler(directory / "tool-audit.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit.addHandler(handler)
    audit.propagate = False
    settings = Settings()
    engine = Orchestrator(settings)
    if args.local_only:
        evidence = engine.assessment("local-real-assessment")
        save(
            directory / "assessment.json",
            {
                "mode": "real_tools_without_llm",
                "response": render_response("local-real-assessment", evidence).model_dump(),
            },
        )
        return
    best = engine.registry.execute("mlflow_compare", {}, "demo-setup")
    if best.status != "ok" or not best.data.get("best_run"):
        raise RuntimeError("Seed real MLflow runs before collecting conversations")
    run_id = best.data["best_run"]["run_id"]
    provenance_result = conversation(
        engine,
        f"Which dataset version was used for MLflow run {run_id}? "
        "Use trace_dataset to retrieve the run and verify its recorded DVC metadata and hash.",
        directory / "01-dataset-provenance.json",
        {"mlflow_run", "dvc_version"},
    )
    versions = [e.data for e in provenance_result.evidence if e.tool == "dvc_version"]
    if not versions or versions[0].get("recorded_hash_matches") is not True:
        raise RuntimeError("Live provenance did not verify the recorded dataset hash")
    comparison_result = conversation(
        engine,
        "Compare the last five training runs and recommend the best candidate "
        "based on F1 score using mlflow_compare. Explain its production readiness limits.",
        directory / "02-model-selection.json",
        {"mlflow_compare"},
    )
    compared = next(e.data for e in comparison_result.evidence if e.tool == "mlflow_compare")
    if compared.get("best_run") is None or len(compared.get("runs", [])) != 5:
        raise RuntimeError("Live comparison must rank five real runs and select a candidate")
    if compared.get("metric") != "f1":
        raise RuntimeError("Live comparison did not use the recorded f1 metric")
    trigger = engine.chat(
        "Trigger the approved iris_training training pipeline with default "
        "configuration using airflow_trigger. Propose the action for review."
    )
    proposals = [
        e
        for e in trigger.evidence
        if e.status == "confirmation_required" and e.tool == "airflow_trigger"
    ]
    if not proposals:
        save(directory / "03-pipeline-execution.json", trigger.model_dump())
        raise RuntimeError("The live LLM did not propose the approved training pipeline")
    proposal = proposals[0].data
    # This script is an explicitly invoked demonstration driver for an isolated test environment.
    # It confirms the reviewed allowlisted training action, never a cluster mutation.
    confirmed = engine.registry.confirm(
        proposal["action_id"], proposal["required_confirmation"], "demo-confirmation"
    )
    if confirmed.status != "ok":
        raise RuntimeError("Confirmed Airflow training run failed to start")
    dag_run_id = confirmed.data["dag_run_id"]
    terminal = wait_run(engine, dag_run_id)
    tasks = engine.registry.execute("airflow_tasks", {"run_id": dag_run_id}, "demo-task-status")
    save(
        directory / "03-pipeline-execution.json",
        {
            "mode": "live_llm_and_real_tools",
            "user": "Trigger training and report execution",
            "proposal_conversation": trigger.model_dump(),
            "explicit_test_driver_confirmation": confirmed.model_dump(),
            "terminal_run": terminal,
            "task_evidence": tasks.model_dump(),
        },
    )
    if terminal["state"] != "success":
        raise RuntimeError("Demonstration training pipeline did not succeed")
    conversation(
        engine,
        "Why is my prediction API unavailable? Investigate k8s_health and "
        "explain the observed deployment, service endpoints, and pod health.",
        directory / "04-deployment-diagnosis.json",
        {"k8s_health"},
    )
    response = conversation(
        engine,
        "Find the best recent model, identify the dataset version "
        "used to train it, check the deployment status, and explain whether "
        "retraining is necessary. Use assess_model.",
        directory / "05-multi-tool-reasoning.json",
        {"mlflow_compare", "dvc_version", "k8s_health", "airflow_runs"},
    )
    if any(e.tool in ("airflow_trigger", "k8s_restart") for e in response.evidence):
        raise RuntimeError("Read-only assessment attempted a mutation")
    save(
        directory / "status.json",
        {"all_five_scenarios_verified": True, "mode": "real_services_and_live_ollama"},
    )


if __name__ == "__main__":
    main()
