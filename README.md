# Evidence-based MLOps AI Agent

[![CI](https://github.com/MehranHamayoon/assignment-03-mlops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/MehranHamayoon/assignment-03-mlops-agent/actions/workflows/ci.yml)
[![Live integration](https://github.com/MehranHamayoon/assignment-03-mlops-agent/actions/workflows/demo.yml/badge.svg)](https://github.com/MehranHamayoon/assignment-03-mlops-agent/actions/workflows/demo.yml)

A Python agent that chooses structured tools through Ollama, inspects **real DVC, MLflow,
Airflow 3 and Kubernetes systems**, and returns observed facts, uncertainties and recommended
actions with the underlying tool evidence. The example ML workload is an Iris classifier.

**Submission entry points:** [technical report](docs/technical-report.md),
[architecture](architecture.md), [requirements checklist](docs/requirements.md),
[tool catalog](docs/tools.md), [evidence index](evidence/README.md), and
[engineering defense](docs/defense.md). Read the [five tested conversations](docs/demo-conversations.md)
for a concise walkthrough of the actual results. No account credentials are needed to inspect this repository.

**Verified:** 31 passing automated tests; five real-service conversation scenarios; all five
Airflow training tasks succeeded; two Kubernetes replicas; injected validation/readiness
failures and healthy recovery. The [successful integration run](https://github.com/MehranHamayoon/assignment-03-mlops-agent/actions/runs/42392077317)
and committed JSON exports provide the execution evidence.

## Quick verification

Use Python **3.12 or 3.13** and Git. The locked dependencies are shared by local runs and CI;
Airflow is isolated in its own Linux image to avoid dependency and Windows compatibility issues.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Ordinary CI uses mocked external calls and never needs a cluster, LLM, cloud subscription or API key.
The separate **Live integration demonstration** workflow launches real disposable services on a
GitHub-hosted Ubuntu runner, exercises five conversations, injects two failures, and exports
sanitized evidence artifacts. Reviewed exports are included under `evidence/live/` in this submission.
Its Actions logs and artifacts provide an independent
execution record. Re-run it from **Actions → Live integration demonstration → Run workflow**.

## Configure and launch the services

For the full stack, install Docker with Compose, kubectl and kind. On Windows, Docker requires
a working Linux backend (usually WSL 2); alternatively run the supplied integration workflow.
Allow approximately 12 GB RAM and 15 GB free disk for the development stack and model download.

```bash
python -m scripts.configure
python -m scripts.seed_remote
python -m scripts.verify_reproduction
docker compose up -d --build
docker compose exec ollama ollama pull qwen3:1.7b
python -m scripts.wait_services --url http://127.0.0.1:5000/health
python -m scripts.wait_services --airflow --seconds 360
```

`scripts.configure` creates random credentials in ignored `.env`; it never prints them.
On Linux it also sets the Airflow container UID to the host UID so run artifacts are writable.
Review `.env.example` for every setting. In PowerShell, import `.env` into the current process
with the following command before running training or the demo (Bash: `set -a; source .env; set +a`):

```powershell
Get-Content .env | Where-Object { $_ -match '^[A-Z_]+=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
```

Open MLflow at [localhost:5000](http://localhost:5000), Airflow at
[localhost:8080](http://localhost:8080) (username `agent`, password from local `.env`).
Airflow's SimpleAuthManager and single-node standalone launcher are for this development
demonstration. A production deployment should use a production auth manager, TLS, a durable
artifact store and separate Airflow components. PostgreSQL backs Airflow; MLflow uses SQLite
in the development environment. No paid external LLM provider is required.

## Train, track and reproduce

```bash
python -m training.train all
# Or the same five stages through DVC:
dvc repro
dvc status
dvc diff --json --show-hash dataset-v1 dataset-v2
```

The first command logs parameters, macro F1, accuracy, portable model, MLflow model/registry
version and dataset provenance. A failed quality gate does not create an approved model.
The Airflow DAG implements validation → preprocessing → training → evaluation/MLflow →
quality gate, with dependencies, two retries and isolated run directories. DAG failures are
reported honestly; a queued trigger is never described as a successful completed run.

Dataset tags `dataset-v1` and `dataset-v2` identify 120-row and 150-row revisions. Their DVC
content hashes are committed; the data is reproducibly generated from scikit-learn's bundled
Iris dataset by `scripts.seed_remote`. The generated local DVC remote contains both versions.
The remote path is local-only configuration, never a credential embedded in Git. See
[data documentation](data/README.md) for source and attribution.

## Deploy the inference service

Run these commands only against your disposable demonstration cluster:

```bash
kind create cluster --name mlops-agent
docker build -f Dockerfile.inference -t iris-api:1.0.0 .
kind load docker-image iris-api:1.0.0 --name mlops-agent
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl apply -f kubernetes/rbac.yaml
python -m scripts.stamp_deployment
kubectl -n mlops rollout status deployment/iris-api --timeout=180s
kubectl -n mlops port-forward service/iris-api 8000:80
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/version
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"instances":[[5.1,3.5,1.4,0.2]]}'
```

The manifests specify two replicas, a ClusterIP Service, startup/readiness/liveness probes,
resource requests/limits, non-sensitive ConfigMap, non-root process and a read-only root
filesystem. A missing model returns readiness HTTP 503. `/version` reports the actual model
run and dataset lineage. The portable inference model is JSON; no untrusted pickle is loaded.

## Talk to the agent

```bash
uvicorn agent.main:app --host 127.0.0.1 --port 8001
# In another activated terminal:
mlops-agent "Compare the last five training runs by F1 score"
mlops-agent "Which datasets are tracked by DVC?"
mlops-agent "Why is my prediction API unavailable?"
mlops-agent "Find the best recent model, identify its dataset version, check the deployment status, and explain whether retraining is necessary"
mlops-agent "Trigger the approved iris_training pipeline"
```

The API requires `X-Agent-Key` on `/chat`, `/tools`, and `/actions/confirm`; the CLI reads the
key from `.env`. Interactive API documentation is at [localhost:8001/docs](http://localhost:8001/docs).
Responses contain `facts`, `uncertainties`, `recommendations`, and a request-correlated
`evidence` array. Facts are constructed only from successful tool results. Model-generated
free text cannot become an operational claim. Tool logs are untrusted data and credentials
are redacted before results return to the LLM or client.

Cluster changes and Airflow triggers return a **proposal**, without changing state. Review its
tool and arguments; then run `mlops-agent --confirm ACTION_ID` and type the exact confirmation.
Confirmations expire after five minutes and are single-use, bound to the proposed action.
The LLM cannot supply `confirmed=true` or invoke the confirmation endpoint. Only the approved
namespace/deployment and training DAG are accepted. Read-only Kubernetes RBAC is the default;
install `kubernetes/restart-role.optional.yaml` only when intentionally demonstrating a
confirmed restart. The host demo driver uses the disposable kind context; deployed agents
should use the supplied restricted service account.

## Evidence and failure demonstrations

```bash
python -m scripts.collect_demo
# Deliberately corrupts data and changes MODEL_PATH in the disposable mlops namespace:
python -m scripts.inject_failures
```

The failure script is an explicit test harness, not an LLM tool. It restores the original data
and model path in `finally` blocks and verifies recovery. Do not run it against shared workloads.
See the [report](docs/technical-report.md) for root causes, evidence interpretation and limitations.

