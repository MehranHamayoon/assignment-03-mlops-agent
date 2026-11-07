"""Rebuild public deterministic data into a local DVC cache and remote after cloning."""

import subprocess
import sys
from pathlib import Path

from training.train import generate


def main():
    remote = Path("dvc-remote").resolve()
    remote.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dvc",
            "remote",
            "add",
            "--force",
            "--local",
            "-d",
            "storage",
            str(remote),
        ],
        check=True,
    )
    original = Path("data/iris.csv.dvc").read_bytes()
    try:
        for version in (1, 2):
            generate(version)
            for args in (("add", "data/iris.csv"), ("push", "data/iris.csv.dvc")):
                subprocess.run([sys.executable, "-m", "dvc", *args], check=True, timeout=90)
    finally:
        generate(2)
        Path("data/iris.csv.dvc").write_bytes(original)
    print("Rebuilt both deterministic dataset versions and populated the local DVC remote.")


if __name__ == "__main__":
    main()
