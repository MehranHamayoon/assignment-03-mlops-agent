# Assignment requirement mapping

The assignment PDF was reviewed in full: seven pages, architecture figure on page 2, five-stage
DAG figure on page 4, demonstration table on page 6 and 100-mark rubric on page 7.

| Requirement | Implementation | Verification/evidence |
|---|---|---|
| DVC datasets, versions, commits, diff, changed data, reproduction | `agent/tools/dvc_tools.py`, dataset tags, seed/reproduction scripts | `evidence/local/dataset-versions.json`, `reproduction.json` |
| MLflow experiments, runs, parameters, metrics, artifacts, comparison, registry | `agent/tools/mlflow_tools.py`, `training/train.py` | live MLflow server records, local/live evidence |
| Airflow 3 authentication, DAG listing/trigger/status/failures/logs | `agent/tools/airflow_tools.py` | live integration workflow exports |
| Five logical training stages, retries, dependencies, quality gate | `dags/training_pipeline.py` | completed DAG tasks and injected validation failure |
| Kubernetes pods/deployments/services/logs/events/version | `agent/tools/kubernetes_tools.py` | real kind cluster evidence |
| Two replicas, ClusterIP, probes, resource limits, ConfigMap | `kubernetes/` | deployment spec and readiness diagnosis |
| Explicit confirmation and namespace restriction | registry, confirmation store, API, RBAC | permission/forged-confirmation/single-use tests |
| Supported LLM structured tool calling and multi-tool reasoning | Ollama planner and bounded orchestrator | live conversations and orchestration tests |
| No unsupported retraining or invented infrastructure claims | deterministic evidence response builder | unavailable-tool test and read-only assessment test |
| At least eight pytest tests, mocks, invalid input, failures, timeout | `tests/` | local pytest XML and CI |
| Structured logs with names/status/duration/request IDs, no secrets | `agent/registry.py`, `agent/safety.py` | log-field and redaction tests |
| Two deliberately injected failures, root cause and recovery | `scripts/inject_failures.py`, report | real Airflow validation and K8s readiness evidence |
| GitHub repository, reproducible environment, lint/test PR workflow | `requirements.lock`, `pyproject.toml`, `.github/workflows/ci.yml` | GitHub Actions status |
| Five tested conversations with actual tools | `scripts/collect_demo.py` | `evidence/live/01` through `05` |
| Technical report, architecture, permissions, limitations, defense | `docs/`, `architecture.md` | repository documentation |

`evidence/live/status.json` is written only after all five real-service conversations pass.
The mere existence of configuration files or a unit-test fixture does not establish that a
live system was deployed. Check the evidence index and Actions execution before assessing completion.

