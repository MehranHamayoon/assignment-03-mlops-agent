# Tool definitions and permissions

Every tool has a Pydantic JSON input schema and a standard evidence envelope:
`tool`, `status`, `data`, `error`, `request_id`, `duration_ms`. Discover the precise machine
schemas at authenticated `GET /tools`. Operational tool data is not executed as instructions.

| Module / callable | Inputs | Operation and permission |
|---|---|---|
| `dvc_datasets` | none | Read actual Git-tracked dataset metadata |
| `dvc_version` | revision, metadata path, optional expected hash | Read committed DVC metadata; verify recorded provenance; return reproduction commands |
| `dvc_status` | none | Read changed/missing dataset status |
| `dvc_compare` | old/new Git revisions | Read actual DVC diff and hashes |
| `mlflow_experiments` | none | List experiments via MlflowClient |
| `mlflow_runs` | experiment IDs, count, recent-day window | Read recent runs, parameters, metrics and tags |
| `mlflow_run` | run ID | Read details and artifact inventory |
| `mlflow_compare` | run scope, metric, maximize/minimize | Compare completed finite-metric runs and select candidate |
| `mlflow_registry` | allowlisted-format model name | Read model versions, aliases and source runs |
| `airflow_dags` | none | Authenticated list of DAGs |
| `airflow_runs` | approved DAG | Read recent run status |
| `airflow_tasks` | approved DAG, run ID | Read task states; identify failed/upstream-failed tasks |
| `airflow_logs` | DAG/run/task/attempt | Read available task-execution diagnostics |
| `airflow_trigger` | approved DAG, bounded training configuration | Propose a run; exact human confirmation executes POST |
| `k8s_pods` | approved namespace | List app-labelled pods and restart counts |
| `k8s_deployments` | approved namespace | Read desired/updated/ready replicas, images and run identity |
| `k8s_services` | approved namespace | Read Services and ready endpoints |
| `k8s_logs` | namespace, app pod, previous flag | Read last 80 lines / maximum 12 KB from app pod |
| `k8s_health` | approved namespace | Correlate pods, rollout, Services, endpoints and pod events |
| `k8s_restart` | approved namespace/deployment | Propose restart; exact confirmation and optional patch RBAC required |
| `assess_model` | none | Read-only dependency-ordered investigation across all four systems |

Namespaced operations use `APPROVED_NAMESPACE` (default `mlops`), application selector
`app=iris-api`, and `APPROVED_DEPLOYMENT` (default `iris-api`). Airflow tools use
`APPROVED_DAG` (default `iris_training`). Dataset metadata paths must resolve under `data/`
with a `.dvc` suffix. Git revisions cannot start with an option marker; subprocesses receive
argument arrays with `shell=False`. MLflow metric names and experiment IDs are validated
before query construction. No arbitrary kubectl, shell, Python eval or manifest application
tool is exposed to the LLM.

