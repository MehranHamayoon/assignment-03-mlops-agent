#!/usr/bin/env bash
set -euo pipefail
set -a; source .env; set +a
mkdir -p runtime artifacts evidence/live

echo 'Rebuild DVC remote and verify historical checkout'
python -m scripts.seed_remote
python -m scripts.verify_reproduction --output evidence/live/reproduction.json

echo 'Start real MLflow and train candidates'
mlflow server --host 0.0.0.0 --port 5000 --workers 1 \
  --backend-store-uri sqlite:///runtime/mlflow.db \
  --artifacts-destination ./runtime/mlartifacts --serve-artifacts \
  --allowed-hosts 'localhost:*,127.0.0.1:*' > runtime/mlflow.log 2>&1 &
python -m scripts.wait_services --url http://127.0.0.1:5000/health
for c in 0.3 0.5 1.0 3.0 10.0; do
  python -m training.train all --c "$c" --directory "artifacts/seed-$c"
done
python -m training.train all

echo 'Start real Airflow 3 with PostgreSQL'
set -a; source .env; set +a
docker run -d --name airflow-postgres --network host \
  -e POSTGRES_USER=airflow -e POSTGRES_DB=airflow \
  -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" postgres:16.11
docker build -f Dockerfile.airflow -t assignment-airflow:3.1.6 .
chmod -R a+rwX artifacts
docker run -d --name assignment-airflow --network host \
  -v "$PWD:/opt/project" -e AIRFLOW_PASSWORD="$AIRFLOW_PASSWORD" \
  -e AIRFLOW__CORE__LOAD_EXAMPLES=false \
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/project/dags \
  -e AIRFLOW__CORE__EXECUTOR=LocalExecutor \
  -e AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS=agent:user \
  -e AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE=/opt/airflow/passwords.json \
  -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflow:$POSTGRES_PASSWORD@127.0.0.1:5432/airflow" \
  -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  assignment-airflow:3.1.6
python -m scripts.wait_services --airflow --seconds 360

echo 'Deploy two-replica prediction application to kind'
curl -fsSL -o runtime/kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
chmod +x runtime/kind
runtime/kind create cluster --name mlops-agent --wait 120s
docker build -f Dockerfile.inference -t iris-api:1.0.0 .
runtime/kind load docker-image iris-api:1.0.0 --name mlops-agent
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl apply -f kubernetes/rbac.yaml
python -m scripts.stamp_deployment
kubectl -n mlops rollout status deployment/iris-api --timeout=180s
kubectl -n mlops get deployment,pods,service,endpoints -o json > evidence/live/cluster-resources.json
kubectl -n mlops port-forward service/iris-api 8000:80 > runtime/port-forward.log 2>&1 &
python -m scripts.wait_services --url http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/version > evidence/live/inference-version.json
curl -fsS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"instances":[[5.1,3.5,1.4,0.2]]}' > evidence/live/prediction.json

echo 'Start real Ollama and download tool-calling model'
docker run -d --name assignment-ollama --network host ollama/ollama:0.13.5
python -m scripts.wait_services --url http://127.0.0.1:11434/api/tags
docker exec assignment-ollama ollama pull qwen3:1.7b

echo 'Execute all five live conversations'
python -m scripts.collect_demo

echo 'Inject two failures and verify recovery'
python -m scripts.inject_failures
