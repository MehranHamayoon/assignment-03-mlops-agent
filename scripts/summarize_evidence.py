"""Render a readable index from actual exported conversation evidence."""

import json
from pathlib import Path


def load(name):
    return json.loads((Path("evidence/live") / name).read_text(encoding="utf-8"))


def observed(conversation, tool):
    return next(
        e["data"]
        for e in conversation["response"]["evidence"]
        if e["tool"] == tool and e["status"] == "ok"
    )


def main():
    first = load("01-dataset-provenance.json")
    comparison = load("02-model-selection.json")
    pipeline = load("03-pipeline-execution.json")
    deployment = load("04-deployment-diagnosis.json")
    multi = load("05-multi-tool-reasoning.json")
    manifest = load("manifest.json")
    provenance = observed(first, "dvc_version")
    best = observed(comparison, "mlflow_compare")["best_run"]
    if best is None:
        raise RuntimeError("The live comparison has no candidate; inspect and rerun it first")
    health = observed(deployment, "k8s_health")
    rows = [
        "# Five tested agent conversations",
        "",
        "These are actual Ollama selections and real service observations exported by the",
        f"[successful hosted run]({manifest['workflow_url']}) at source commit",
        f"`{manifest['code_commit']}`. The response fields below are rendered from the JSON",
        "evidence; they are not invented illustrative dialogue.",
        "",
        "## 1. Dataset provenance",
        "",
        f"> {first['user']}",
        "",
        f"The run used dataset MD5 `{provenance['outputs'][0]['md5']}` from Git commit",
        f"`{provenance['git_commit']}`. Recorded MLflow hash matches committed DVC metadata:",
        f"**{provenance['recorded_hash_matches']}**.",
        "",
        "[Complete tool evidence](../evidence/live/01-dataset-provenance.json)",
        "",
        "## 2. Model selection",
        "",
        f"> {comparison['user']}",
        "",
        f"Selected candidate: `{best['run_id']}`; macro F1 **{best['metrics']['f1']:.4f}**.",
        "This is a ranking within the requested recent-run scope. The response explicitly",
        "requires separate quality-gate, lineage and deployment checks before promotion.",
        "",
        "[Complete comparison](../evidence/live/02-model-selection.json)",
        "",
        "## 3. Pipeline execution",
        "",
        "> Trigger the approved training pipeline and report execution.",
        "",
        "The agent proposed the approved action. The explicitly invoked test driver confirmed",
        "that exact proposal; Airflow executed the real DAG and the driver waited for completion.",
        f"Run `{pipeline['terminal_run']['dag_run_id']}` finished "
        f"**{pipeline['terminal_run']['state']}**.",
        "",
        "| Task | Recorded state |",
        "|---|---|",
    ]
    rows += [
        f"| {t['task_id']} | {t['state']} |"
        for t in pipeline["task_evidence"]["data"]["task_instances"]
    ]
    rows += [
        "",
        "[Run and task evidence](../evidence/live/03-pipeline-execution.json)",
        "",
        "## 4. Deployment diagnosis",
        "",
        f"> {deployment['user']}",
        "",
        f"Observed healthy: **{health['healthy']}**. The real deployment reported:",
        "",
    ]
    for item in health["deployments"]:
        rows.append(
            f"* `{item['name']}`: {item['ready_replicas']}/{item['desired_replicas']} "
            f"ready replicas, image `{', '.join(item['images'])}`."
        )
    rows += [
        "",
        "The supplied question does not itself prove an outage. The tool observations show",
        "the state at inspection. The separate failure injection captures a real readiness 503.",
        "",
        "[Health, Services, endpoints and events](../evidence/live/04-deployment-diagnosis.json)",
        "",
        "## 5. Multi-tool investigation",
        "",
        f"> {multi['user']}",
        "",
        "Successful sources: "
        + ", ".join(f"`{e['tool']}`" for e in multi["response"]["evidence"] if e["status"] == "ok")
        + ".",
        "",
        "Recorded uncertainties:",
        "",
    ]
    rows += [f"* {u}" for u in multi["response"]["uncertainties"]]
    rows += ["", "Recorded recommendations:", ""]
    rows += [f"* {r}" for r in multi["response"]["recommendations"]]
    rows += [
        "",
        "No training trigger or cluster mutation was invoked by this assessment.",
        "",
        "[Complete correlated evidence](../evidence/live/05-multi-tool-reasoning.json)",
        "",
        "[Two real failure investigations and recovery](../evidence/live/failures/)",
        "",
    ]
    Path("docs/demo-conversations.md").write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
