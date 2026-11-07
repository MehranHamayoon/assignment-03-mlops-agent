"""Record the actual packaged model run identity on the disposable app Deployment."""

import json
import subprocess
from pathlib import Path


def main():
    model = json.loads(Path("artifacts/approved/model.json").read_text(encoding="utf-8"))
    subprocess.run(
        [
            "kubectl",
            "-n",
            "mlops",
            "set",
            "env",
            "deployment/iris-api",
            f"MODEL_RUN_ID={model['run_id']}",
        ],
        check=True,
        timeout=30,
    )


if __name__ == "__main__":
    main()
