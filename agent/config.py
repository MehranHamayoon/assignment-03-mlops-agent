from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    repo_root: Path = Path(".")
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "iris-classifier"
    airflow_url: str = "http://127.0.0.1:8080"
    airflow_username: str = "agent"
    airflow_password: SecretStr = SecretStr("")
    airflow_token: SecretStr = SecretStr("")
    approved_dag: str = "iris_training"
    approved_namespace: str = "mlops"
    approved_deployment: str = "iris-api"
    kube_context: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:1.7b"
    agent_api_key: SecretStr = SecretStr("")
    tool_timeout: float = Field(15, ge=0.1, le=120)
    model_path: Path = Path("artifacts/model.json")
    app_version: str = "1.0.0"
