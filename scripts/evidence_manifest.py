import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    directory = Path("evidence/live")
    directory.mkdir(parents=True, exist_ok=True)
    entries = {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.suffix in (".json", ".jsonl") and p.name != "manifest.json"
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    run_id = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY", "MehranHamayoon/assignment-03-mlops-agent")
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "workflow_url": f"https://github.com/{repo}/actions/runs/{run_id}" if run_id else None,
        "sha256": entries,
    }
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
