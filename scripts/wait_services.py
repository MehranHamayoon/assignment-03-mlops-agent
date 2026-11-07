import argparse
import time

import httpx

from agent.config import Settings
from agent.tools.airflow_tools import AirflowTools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airflow", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            if args.airflow:
                adapter = AirflowTools(Settings())
                dags = adapter.dags()
                if not any(d["dag_id"] == "iris_training" for d in dags["dags"]):
                    raise RuntimeError("Waiting for DAG discovery")
            else:
                response = httpx.get(args.url, timeout=5)
                response.raise_for_status()
            print("Service is ready.")
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError("Service readiness deadline exceeded")


if __name__ == "__main__":
    main()
