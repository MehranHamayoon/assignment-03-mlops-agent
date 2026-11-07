"""Create two real DVC/Git revisions and a local content-addressed remote."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from training.train import generate


def command(*args, cwd=None):
    return subprocess.run(
        list(args), cwd=cwd, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="dvc-remote")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if not (root / ".git").exists():
        raise RuntimeError("Initialize or clone the Git repository first")
    if (root / "data/iris.csv.dvc").exists():
        generate(2)
        print("Dataset metadata exists; regenerated deterministic v2 content.")
        return
    if not (root / ".dvc").exists():
        command(sys.executable, "-m", "dvc", "init")
    remote = Path(args.remote).resolve()
    remote.mkdir(parents=True, exist_ok=True)
    command(
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
    )
    records = []
    for version in (1, 2):
        generate(version)
        command(sys.executable, "-m", "dvc", "add", "data/iris.csv")
        command(
            "git",
            "add",
            ".dvc/.gitignore",
            ".dvc/config",
            ".dvcignore",
            "data/.gitignore",
            "data/iris.csv.dvc",
        )
        command("git", "commit", "-m", f"data: record deterministic Iris dataset v{version}")
        command("git", "tag", f"dataset-v{version}")
        command(sys.executable, "-m", "dvc", "push", "data/iris.csv.dvc")
        records.append(
            {
                "version": version,
                "git_commit": command("git", "rev-parse", "HEAD"),
                "md5": hashlib.md5(Path("data/iris.csv").read_bytes()).hexdigest(),
            }
        )
    Path("evidence/local").mkdir(parents=True, exist_ok=True)
    Path("evidence/local/dataset-versions.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
