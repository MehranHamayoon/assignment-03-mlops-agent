"""Airflow 3 TaskFlow DAG with five required stages and an enforced quality gate."""

from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, get_current_context, task

from training import train as pipeline


@dag(
    dag_id="iris_training",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=10)},
    tags=["mlops", "assignment-03"],
)
def iris_training():
    def config():
        from agent.tools.airflow_tools import TrainingConf

        ctx = get_current_context()
        conf = TrainingConf.model_validate(ctx["dag_run"].conf or {})
        # Run-isolated paths avoid races; shared volume supports local executor task processes.
        import hashlib

        suffix = hashlib.sha256(ctx["run_id"].encode()).hexdigest()[:16]
        directory = str(Path("/opt/project/artifacts/airflow") / suffix)
        return conf, directory

    @task(task_id="validate_dataset")
    def validate_dataset():
        _, directory = config()
        return pipeline.validate(output=Path(directory) / "validation.json")

    @task(task_id="preprocess_data")
    def preprocess_data():
        _, directory = config()
        return pipeline.preprocess(directory=directory)

    @task(task_id="train_model")
    def train_model():
        conf, directory = config()
        pipeline.train(directory, conf.c, conf.max_iter)

    @task(task_id="evaluate_and_log")
    def evaluate_and_log():
        conf, directory = config()
        return pipeline.evaluate(directory, conf.quality_threshold)

    @task(task_id="quality_gate", retries=0)
    def quality_gate():
        _, directory = config()
        # Failure raises an exception and never writes an approved model.
        return pipeline.gate(directory)

    validation = validate_dataset()
    preprocessing = preprocess_data()
    training = train_model()
    evaluation = evaluate_and_log()
    quality = quality_gate()
    validation >> preprocessing >> training >> evaluation >> quality


iris_training()
