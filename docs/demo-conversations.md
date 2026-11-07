# Five tested agent conversations

These are actual Ollama selections and real service observations exported by the
[successful hosted run](https://github.com/MehranHamayoon/assignment-03-mlops-agent/actions/runs/42392077317) at source commit
`e3e1339e1d05c301de566fe63a19e20989ef6ea4`. The response fields below are rendered from the JSON
evidence; they are not invented illustrative dialogue.

## 1. Dataset provenance

> Which dataset version was used for MLflow run fca7b07a516e49f5bd2b48982c69b8b1? Use trace_dataset to retrieve the run and verify its recorded DVC metadata and hash.

The run used dataset MD5 `4224576f0267bf88902f87f0f6200967` from Git commit
`e3e1339e1d05c301de566fe63a19e20989ef6ea4`. Recorded MLflow hash matches committed DVC metadata:
**True**.

[Complete tool evidence](../evidence/live/01-dataset-provenance.json)

## 2. Model selection

> Compare the last five training runs and recommend the best candidate based on F1 score using mlflow_compare. Explain its production readiness limits.

Selected candidate: `fca7b07a516e49f5bd2b48982c69b8b1`; macro F1 **0.9487**.
This is a ranking within the requested recent-run scope. The response explicitly
requires separate quality-gate, lineage and deployment checks before promotion.

[Complete comparison](../evidence/live/02-model-selection.json)

## 3. Pipeline execution

> Trigger the approved training pipeline and report execution.

The agent proposed the approved action. The explicitly invoked test driver confirmed
that exact proposal; Airflow executed the real DAG and the driver waited for completion.
Run `manual__2026-11-07T16:41:14.626318+00:00` finished **success**.

| Task | Recorded state |
|---|---|
| validate_dataset | success |
| preprocess_data | success |
| train_model | success |
| evaluate_and_log | success |
| quality_gate | success |

[Run and task evidence](../evidence/live/03-pipeline-execution.json)

## 4. Deployment diagnosis

> Why is my prediction API unavailable? Investigate k8s_health and explain the observed deployment, service endpoints, and pod health.

Observed healthy: **True**. The real deployment reported:

* `iris-api`: 2/2 ready replicas, image `iris-api:1.0.0`.

The supplied question does not itself prove an outage. The tool observations show
the state at inspection. The separate failure injection captures a real readiness 503.

[Health, Services, endpoints and events](../evidence/live/04-deployment-diagnosis.json)

## 5. Multi-tool investigation

> Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary. Use assess_model.

Successful sources: `mlflow_compare`, `mlflow_run`, `dvc_version`, `dvc_status`, `k8s_health`, `airflow_runs`.

Recorded uncertainties:

* The selected run is not verified as the deployed run.

Recorded recommendations:

* Metric ranking selects a candidate; production readiness also requires a passed gate, verified lineage and health.
* Available evidence does not establish a need to retrain. Obtain a freshness policy or degradation measurements if needed.

No training trigger or cluster mutation was invoked by this assessment.

[Complete correlated evidence](../evidence/live/05-multi-tool-reasoning.json)

[Two real failure investigations and recovery](../evidence/live/failures/)
