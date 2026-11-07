"""Five reproducible stages; no network is needed to generate the Iris dataset."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
log = logging.getLogger(__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate(version=2, path="data/iris.csv"):
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURES)
    frame["target"] = iris.target
    if version == 1:
        frame = frame.groupby("target", sort=True).head(40)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return {"version": version, "rows": len(frame), "path": str(path)}


def validate(data="data/iris.csv", output="artifacts/validation.json"):
    frame = pd.read_csv(data)
    if list(frame.columns) != FEATURES + ["target"]:
        raise ValueError("Dataset columns do not match the required schema")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Dataset is empty or contains missing values")
    numeric = frame.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame[FEATURES] <= 0).any().any():
        raise ValueError("Dataset contains invalid or nonpositive measurements")
    if set(frame["target"].unique()) != {0, 1, 2}:
        raise ValueError("Dataset requires all three Iris classes")
    md5 = hashlib.md5(Path(data).read_bytes()).hexdigest()
    dvc_file = f"{data}.dvc"
    metadata = yaml.safe_load(Path(dvc_file).read_text(encoding="utf-8"))
    if md5 != metadata["outs"][0]["md5"]:
        raise ValueError("Dataset bytes differ from DVC metadata")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()
    committed = subprocess.run(
        ["git", "show", f"{commit}:{dvc_file}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout
    if yaml.safe_load(committed) != metadata:
        raise ValueError("Commit the DVC metadata before training")
    result = {
        "rows": len(frame),
        "dataset.md5": md5,
        "dataset.dvc_file": dvc_file,
        "dataset.git_commit": commit,
        "dataset.version": f"md5:{md5}",
        "valid": True,
    }
    write_json(output, result)
    log.info("validation_passed rows=%s dataset_md5=%s", len(frame), md5)
    return result


def preprocess(data="data/iris.csv", directory="artifacts"):
    frame = pd.read_csv(data)
    train, test = train_test_split(frame, test_size=0.25, random_state=42, stratify=frame["target"])
    Path(directory).mkdir(parents=True, exist_ok=True)
    train.to_csv(Path(directory) / "train.csv", index=False)
    test.to_csv(Path(directory) / "test.csv", index=False)
    log.info("preprocessing_complete train_rows=%s test_rows=%s", len(train), len(test))
    return {"train_rows": len(train), "test_rows": len(test)}


def train(directory="artifacts", c=1.0, max_iter=300):
    frame = pd.read_csv(Path(directory) / "train.csv")
    scaler = StandardScaler().fit(frame[FEATURES])
    model = LogisticRegression(C=c, max_iter=max_iter, random_state=42)
    model.fit(scaler.transform(frame[FEATURES]), frame["target"])
    result = {
        "format_version": 1,
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
        "class_names": list(load_iris().target_names),
        "parameters": {"c": c, "max_iter": max_iter, "random_state": 42},
        "provenance": read_json(Path(directory) / "validation.json"),
    }
    write_json(Path(directory) / "model.json", result)
    log.info("training_complete c=%s max_iter=%s", c, max_iter)
    return result


def predict_json(model: dict, features) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    normalized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    scores = normalized @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    classes = np.asarray(model["classes"])
    return classes[probs.argmax(axis=1)], probs


def evaluate(directory="artifacts", threshold=0.85):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import mlflow
    from mlflow import MlflowClient

    directory = Path(directory)
    model = read_json(directory / "model.json")
    frame = pd.read_csv(directory / "test.csv")
    predictions, _ = predict_json(model, frame[FEATURES])
    metrics = {
        "f1": float(f1_score(frame["target"], predictions, average="macro")),
        "accuracy": float(accuracy_score(frame["target"], predictions)),
    }
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "iris-classifier"))

    class JsonClassifier(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            values = model_input[FEATURES] if isinstance(model_input, pd.DataFrame) else model_input
            return predict_json(model, values)[0]

    with mlflow.start_run() as run:
        mlflow.log_params(model["parameters"])
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                **{k: v for k, v in model["provenance"].items() if k.startswith("dataset.")},
                "quality_gate": "pending",
                "quality_threshold": str(threshold),
            }
        )
        mlflow.log_dict(model["provenance"], "provenance.json")
        mlflow.log_artifact(str(directory / "model.json"), "portable_model")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=JsonClassifier(),
            input_example=frame[FEATURES].head(2),
            registered_model_name="iris-classifier",
            pip_requirements=[f"mlflow=={mlflow.__version__}", "numpy", "pandas", "scikit-learn"],
        )
        result = {"run_id": run.info.run_id, "metrics": metrics, "threshold": threshold}
    model["run_id"] = result["run_id"]
    write_json(directory / "evaluated_model.json", model)
    write_json(directory / "evaluation.json", result)
    MlflowClient().set_tag(result["run_id"], "model.portable_format", "json-logistic-regression")
    log.info("evaluation_logged run_id=%s f1=%.4f", result["run_id"], metrics["f1"])
    return result


def gate(directory="artifacts"):
    import mlflow
    from mlflow import MlflowClient

    result = read_json(Path(directory) / "evaluation.json")
    passed = result["metrics"]["f1"] >= result["threshold"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    MlflowClient().set_tag(result["run_id"], "quality_gate", "passed" if passed else "failed")
    write_json(Path(directory) / "quality_gate.json", {**result, "passed": passed})
    if not passed:
        raise ValueError(
            f"Quality gate failed: f1={result['metrics']['f1']:.4f} < {result['threshold']}"
        )
    # Only a passed candidate is written to the deployment directory.
    write_json(
        Path(directory) / "approved" / "model.json",
        read_json(Path(directory) / "evaluated_model.json"),
    )
    log.info("quality_gate_passed run_id=%s", result["run_id"])
    return {**result, "passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["generate", "validate", "preprocess", "train", "evaluate", "gate", "all"]
    )
    parser.add_argument("--version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--directory", default="artifacts")
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["train"]
    c = args.c if args.c is not None else params["c"]
    max_iter = args.max_iter if args.max_iter is not None else params["max_iter"]
    threshold = args.threshold if args.threshold is not None else params["quality_threshold"]
    operations = {
        "generate": lambda: generate(args.version),
        "validate": lambda: validate(output=Path(args.directory) / "validation.json"),
        "preprocess": lambda: preprocess(directory=args.directory),
        "train": lambda: train(args.directory, c, max_iter),
        "evaluate": lambda: evaluate(args.directory, threshold),
        "gate": lambda: gate(args.directory),
    }
    if args.stage == "all":
        for name in ("validate", "preprocess", "train", "evaluate", "gate"):
            operations[name]()
    else:
        print(json.dumps(operations[args.stage](), indent=2))


if __name__ == "__main__":
    main()
