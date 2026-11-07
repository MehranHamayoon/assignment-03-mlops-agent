# Technical report: Evidence-based MLOps AI Agent

**Course:** MLOps & Production AI Engineering, Fall 2026 · **Assignment:** 03

## 1. Problem, scope and design

The system gives an engineer one conversational interface for dataset provenance, experiment
comparison, workflow execution and deployment diagnosis. It implements four independently
callable tool modules and a supported LLM tool-calling planning step. The architecture follows the
assignment's first figure; its Airflow DAG follows the required five-stage second figure.
The complete flow and training diagram are in [architecture.md](../architecture.md).

FastAPI validates and authenticates requests. Ollama chooses JSON-schema tools in one bounded
`/api/chat` request. Typed Python workflows resolve dependent references from actual tool results;
large operational payloads are never fed through repeated LLM turns. A central registry
validates each call again, applies allowlists, proposes changes,
measures duration and records a request ID. DVC subprocesses, MlflowClient, Airflow 3 REST API
v2 and the Kubernetes Python client return actual observations. A deterministic response
builder exposes those observations as facts, with explicit unknowns and rule-supported
recommendations. Free-form model narration never supplies operational facts.

## 2. Data lineage and model selection

The Iris workload is deterministic and runs without a dataset download. Version 1 has 120 rows
and version 2 has 150; both are DVC-tracked and tagged in Git. The content hashes are:

| Version | DVC MD5 | Evidence |
|---|---|---|
| v1 | `4a3beb7462eb695a87c685647c3245c2` | Fresh-clone DVC restoration, 120 rows |
| v2 | `4224576f0267bf88902f87f0f6200967` | Fresh-clone DVC restoration, 150 rows |

See [dataset revisions](../evidence/local/dataset-versions.json) and
[actual restoration output](../evidence/local/reproduction.json). DVC metadata is read at a
resolved full Git commit; a recorded MLflow hash is checked against that committed metadata.
Validation also compares actual file bytes and the committed metadata before training.

The training split is stratified, 112 training / 38 evaluation rows for v2, seed 42. Scaling is
fit on training data only. Five actual local candidates produced these held-out results:

| LogisticRegression C | Macro F1 | Accuracy | Gate at F1 ≥ 0.85 |
|---|---:|---:|---|
| 0.1 | 0.8452 | 0.8421 | Failed |
| 0.3 | 0.8974 | 0.8947 | Passed |
| 1.0 | 0.9230 | 0.9211 | Passed |
| 3.0 | 0.9487 | 0.9474 | Passed |
| 10.0 | 0.9487 | 0.9474 | Passed |

These are real measurements exported from the local MLflow server, not illustrative values;
[training-runs.json](../evidence/local/training-runs.json) records the exact run IDs and model
registry versions. The tool compares the latest requested count within a declared day window,
includes only finished runs with finite requested metrics, and supports metric minimization.
Equal scores retain the recency ordering. Ranking selects a candidate, not proof of readiness.

The mandatory four-system investigation reads MLflow first, checks that run's DVC lineage,
inspects deployment readiness and model identity, queries recent Airflow workflows, and
separates known facts from missing observations. It never triggers training. Retraining
recommendations require a failed gate, recorded freshness policy or supplied degradation result.

## 3. Workflow, deployment and permissions

The Airflow 3 DAG implements `validate_dataset → preprocess_data → train_model →
evaluate_and_log → quality_gate`. Tasks use retry=2 with a ten-second delay; a failed quality
gate is non-retryable because repeating identical evidence cannot improve the metric.
Airflow task directories are isolated by run ID and concurrent runs are limited to one.
Validation failures block all following tasks. A failed gate never writes an approved model.
DVC invokes the same stages and records dependency/output hashes in `dvc.lock`.

The inference image uses an approved portable JSON model. The Kubernetes deployment desires
two replicas behind a ClusterIP Service and sets startup, liveness and readiness probes, CPU
and memory requests/limits, a ConfigMap, non-root identity and a read-only root filesystem.
`/version` reports loaded model lineage. The operations adapter checks Pods, restart counts,
Deployment generations/replicas/images, Services, ready endpoints, relevant events and bounded logs.

[Tool definitions and permissions](tools.md) document all 22 callable operations. Read-only
Kubernetes RBAC is supplied by default. A separately installed optional Role permits patching
only `iris-api`. Namespace and deployment checks run before proposals or SDK operations.
Airflow triggers are restricted to the approved training DAG and typed bounded parameters.
Every write proposal requires an exact, authenticated, one-use confirmation within five minutes;
the model cannot manufacture confirmation flags or change the stored arguments.

## 4. Tests and real-service demonstrations

The local suite has **29 passing tests** and approximately **77% aggregate statement coverage**
across agent, API and training. The tests cover successful tool selection/execution, invalid
arguments, missing services, timeouts, absence of invented results, forbidden cluster changes,
forged/replayed/expired confirmations, credential redaction, MLflow ranking, DVC path/revision
validation, Airflow authentication, failed task detection and prediction readiness/validation.
External calls are mocked in ordinary CI, which runs lint, formatting and pytest on pull requests.

Local evidence includes real DVC restoration, actual MLflow experiments/registry, the five-stage
DVC pipeline and two injected local failures. Windows had no Docker/WSL or Kubernetes backend;
Airflow/Kubernetes failures in the local assessment are explicitly unavailable, not simulated
deployment results. The separate hosted demonstration builds the real Linux stack.

| Tested conversation | Required actual evidence export |
|---|---|
| Which dataset trained this MLflow run? | `evidence/live/01-dataset-provenance.json` |
| Compare the last five runs by F1 | `evidence/live/02-model-selection.json` |
| Trigger training and report execution | `evidence/live/03-pipeline-execution.json` |
| Investigate prediction API availability | `evidence/live/04-deployment-diagnosis.json` |
| Best model + DVC lineage + health + retraining | `evidence/live/05-multi-tool-reasoning.json` |

The [evidence index](../evidence/README.md) distinguishes local runs, mocks and live hosted
observations. `evidence/live/status.json` is created only if all five conversations obtain the
required successful tool results. `manifest.json` links the exact source commit and Actions run
and records SHA-256 checksums of the exports. Read the actual workflow outcome when assessing
live completion; configuration alone is not execution evidence.

## 5. Failure investigations and resolutions

**Invalid dataset.** Replacing the CSV with a one-column `wrong_schema` file raised
`Dataset columns do not match the required schema`; DVC simultaneously identified changed data.
The root cause was deliberate schema corruption. Restoring the original versioned bytes and
rerunning validation resolved it. The hosted injection repeats this failure through Airflow,
captures the failed validation task and blocked downstream tasks, and exports the task log.

**Failed quality gate.** An intentionally undertrained model (`C=0.0001`, one iteration) scored
F1 **0.7833**, below the deliberately strict threshold **1.0**. The gate raised an exception,
set the MLflow `quality_gate=failed` tag and did not create an approved model. The resolution is
to investigate training configuration and retrain with validated settings, then reevaluate;
lowering the quality policy merely to hide a failure is not a resolution. The default C=1 model
passed the normal 0.85 gate. Exact evidence is in
[failure-quality-gate.json](../evidence/local/failure-quality-gate.json).

**Kubernetes readiness failure.** The hosted test sets `MODEL_PATH=/missing/model.json` in the
isolated `mlops` Deployment. The process stays alive but readiness returns 503, new Pods remain
unready and rollout cannot complete. The agent collects real readiness, rollout, events and
logs. The harness restores `/opt/model/model.json` in a `finally` block and verifies rollout
recovery. This diagnoses a configuration/model-loading issue; it does not imply retraining.
Exports are under `evidence/live/failures/`.

An additional integration issue discovered during implementation was DVC rejecting a `.dvc`
file as a pipeline dependency. The pipeline now depends on data bytes and training code while
validation independently checks metadata against Git. A Windows console encoding failure
after MLflow model logging was resolved by UTF-8 output; the already-recorded run was not
reported as a completed end-to-end training demonstration until the stages were rerun.

## 6. Limitations and proposed improvements

* The Iris hold-out set is small; production needs broader validation, monitoring and drift tests.
* F1 ranking, observed Kubernetes readiness and model identity are separate signals. No drift
  monitoring service is implemented; missing degradation/freshness evidence remains unknown.
* The small local LLM can select tools inconsistently. One bounded planning call and a response
  token limit prevent repetitive tool loops. Built-in provenance and assessment workflows
  resolve dependent references. New kinds of dependent investigations require another typed
  workflow; the planner does not perform open-ended iterative reasoning over service payloads.
* Confirmations are in-memory in one agent process. Multiple workers need an atomic shared
  store, user-specific authorization, persistent auditing and an identity provider.
* The Airflow standalone/SimpleAuthManager stack and local DVC remote are development choices.
  Production should add a production auth manager, TLS, durable remote storage, backups,
  network policies, image digests/signing and independent infrastructure credentials.
* Tool requests have transport timeouts; aggregate multi-call diagnosis may take longer.
  Kubernetes/DVC read responses and MLflow searches are bounded, but very large installations
  need pagination and finer per-resource policies.
* Inference uses JSON rather than pickle; the internally created MLflow pyfunc artifact uses
  trusted project serialization and should not be used to load arbitrary third-party models.

## References

* [DVC diff and revision comparison](https://doc.dvc.org/command-reference/diff)
* [MLflow Python client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
* [Airflow 3.1.6 REST API](https://airflow.apache.org/docs/apache-airflow/3.1.6/stable-rest-api-ref.html)
* [Ollama structured tool calling](https://docs.ollama.com/capabilities/tool-calling)
* [Kubernetes health probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
* [scikit-learn Iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
