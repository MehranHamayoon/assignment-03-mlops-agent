import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evidence/local/reproduction.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    parent = root / "tmp" / "reproduction" / uuid.uuid4().hex[:8]
    parent.mkdir(parents=True, exist_ok=True)
    records = []
    for version in (1, 2):
        destination = parent / f"version-{version}"
        if destination.exists():
            raise RuntimeError(f"Use a fresh reproduction destination: {destination}")
        subprocess.run(
            ["git", "clone", "--no-hardlinks", str(root), str(destination)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "checkout", f"dataset-v{version}"],
            cwd=destination,
            check=True,
            capture_output=True,
            timeout=30,
        )
        for args in (
            ("remote", "add", "--local", "-d", "storage", str(root / "dvc-remote")),
            ("pull", "data/iris.csv.dvc"),
            ("checkout", "data/iris.csv.dvc"),
        ):
            subprocess.run(
                [sys.executable, "-m", "dvc", *args],
                cwd=destination,
                check=True,
                capture_output=True,
                timeout=90,
            )
        data = (destination / "data/iris.csv").read_bytes()
        records.append(
            {
                "version": version,
                "md5": hashlib.md5(data).hexdigest(),
                "rows": len(data.splitlines()) - 1,
                "result": "Restored from DVC remote in a fresh Git clone",
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
