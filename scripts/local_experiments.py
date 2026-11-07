import json
from pathlib import Path

from agent.config import Settings
from agent.orchestrator import Orchestrator, render_response
from agent.tools.dvc_tools import DvcTools
from agent.tools.mlflow_tools import MlflowTools
from scripts.collect_demo import save
from training.train import evaluate, gate, preprocess, train, validate


def main():
    settings = Settings()
    directory = Path("evidence/local")
    directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for index, c in enumerate((0.1, 0.3, 1.0, 3.0, 10.0)):
        target = Path("artifacts") / f"experiment-{index}"
        validate(output=target / "validation.json")
        preprocess(directory=target)
        train(target, c=c)
        result = evaluate(target)
        try:
            gate(target)
        except ValueError:
            result["quality_gate"] = "failed"
        else:
            result["quality_gate"] = "passed"
        runs.append(result)
    # Build the deployment image from the default candidate, with a verified passing gate.
    validate()
    preprocess()
    train()
    evaluate()
    gate()
    adapter = MlflowTools(settings)
    save(
        directory / "training-runs.json",
        {
            "mode": "real_mlflow_server",
            "runs": runs,
            "comparison": adapter.compare(),
            "registry": adapter.model_registry(),
        },
    )
    dvc = DvcTools(settings)
    save(
        directory / "dvc-tools.json",
        {
            "datasets": dvc.datasets(),
            "status": dvc.status(),
            "comparison": dvc.compare("dataset-v1", "dataset-v2"),
        },
    )
    # Two actual local failures, independent of the remote Airflow/Kubernetes injections.
    original = Path("data/iris.csv").read_bytes()
    try:
        Path("data/iris.csv").write_text("wrong_schema\n1\n", encoding="utf-8")
        try:
            validate()
        except ValueError as exc:
            save(
                directory / "failure-invalid-data.json",
                {
                    "mode": "real_local_failure",
                    "error": str(exc),
                    "dvc_status": dvc.status(),
                    "root_cause": "Input schema was deliberately corrupted",
                    "resolution": "Restored original dataset bytes; validation rerun below",
                },
            )
        else:
            raise RuntimeError("Invalid data was not rejected")
    finally:
        Path("data/iris.csv").write_bytes(original)
    validate()
    failed_dir = Path("artifacts/failing-gate")
    validate(output=failed_dir / "validation.json")
    preprocess(directory=failed_dir)
    train(failed_dir, c=0.0001, max_iter=1)
    result = evaluate(failed_dir, threshold=1.0)
    try:
        gate(failed_dir)
    except ValueError as exc:
        save(
            directory / "failure-quality-gate.json",
            {
                "mode": "real_local_failure",
                "evaluation": result,
                "error": str(exc),
                "approved_model_exists": (failed_dir / "approved/model.json").exists(),
                "root_cause": "Deliberately undertrained model failed the configured F1 threshold",
                "resolution": "Use a validated model configuration and rerun all stages",
            },
        )
    else:
        raise RuntimeError("Deliberately failing quality gate unexpectedly passed")
    engine = Orchestrator(settings)
    evidence = engine.assessment("real-local-assessment")
    save(
        directory / "assessment.json",
        {
            "mode": "real_tools_without_llm",
            "response": render_response("real-local-assessment", evidence).model_dump(),
        },
    )
    print(json.dumps({"real_experiments": len(runs), "two_local_failures_verified": True}))


if __name__ == "__main__":
    main()
