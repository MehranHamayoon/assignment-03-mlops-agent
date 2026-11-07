import json
import re
import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import Field

from agent.config import Settings
from agent.registry import Registry, Tool
from agent.schemas import EmptyArgs, StrictModel


class VersionArgs(StrictModel):
    rev: str = Field(default="HEAD", min_length=1, max_length=100)
    dvc_file: str = "data/iris.csv.dvc"
    expected_md5: str | None = None


class DiffArgs(StrictModel):
    old_rev: str = Field(min_length=1, max_length=100)
    new_rev: str = Field(default="HEAD", min_length=1, max_length=100)


class DvcTools:
    def __init__(self, settings: Settings):
        self.root = settings.repo_root.resolve()
        self.timeout = settings.tool_timeout

    def command(self, *args: str) -> str:
        return subprocess.run(
            list(args),
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=self.timeout,
            shell=False,
        ).stdout.strip()

    def revision(self, rev: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./~^{}-]{0,99}", rev):
            raise ValueError("Invalid revision")
        sha = self.command("git", "rev-parse", "--verify", f"{rev}^{{commit}}")
        if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
            raise ValueError("Invalid commit")
        return sha

    def metadata_path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root / "data") or path.suffix != ".dvc":
            raise PermissionError("Only dataset .dvc metadata is allowed")
        return path

    def datasets(self) -> dict:
        files = self.command("git", "ls-files", "--", "data/*.dvc").splitlines()
        return {"datasets": [self.version(dvc_file=f) for f in files]}

    def version(self, rev="HEAD", dvc_file="data/iris.csv.dvc", expected_md5=None) -> dict:
        path = self.metadata_path(dvc_file).relative_to(self.root).as_posix()
        sha = self.revision(rev)
        doc = yaml.safe_load(self.command("git", "show", f"{sha}:{path}"))
        outs = doc.get("outs", [])
        if not outs:
            raise ValueError("No DVC outputs")
        hashes = [x.get("md5") for x in outs]
        return {
            "git_commit": sha,
            "dvc_file": path,
            "outputs": outs,
            "recorded_hash_matches": expected_md5 in hashes if expected_md5 else None,
            "reproduce_in_new_clone": [
                "git clone <repository-url> reproduced-dataset",
                "cd reproduced-dataset",
                f"git checkout {sha}",
                "dvc remote add --local -d storage <authorized-remote-path>",
                f"dvc pull {path}",
                f"dvc checkout {path}",
            ],
            "remote_note": "Remote availability is not implied by committed metadata.",
        }

    def status(self) -> dict:
        result = json.loads(
            self.command(sys.executable, "-m", "dvc", "status", "--json", "data/iris.csv.dvc")
        )
        return {"changed": bool(result), "details": result}

    def compare(self, old_rev: str, new_rev="HEAD") -> dict:
        old, new = self.revision(old_rev), self.revision(new_rev)
        return {
            "old_commit": old,
            "new_commit": new,
            "diff": json.loads(
                self.command(sys.executable, "-m", "dvc", "diff", "--json", "--show-hash", old, new)
            ),
        }

    def register(self, registry: Registry):
        for name, description, schema, handler in [
            (
                "dvc_datasets",
                "List actual Git-tracked DVC datasets and committed hashes.",
                EmptyArgs,
                self.datasets,
            ),
            (
                "dvc_version",
                "Read DVC metadata at a Git revision; verify MLflow provenance hash.",
                VersionArgs,
                self.version,
            ),
            (
                "dvc_status",
                "Detect changed or missing dataset content with real dvc status.",
                EmptyArgs,
                self.status,
            ),
            (
                "dvc_compare",
                "Compare dataset versions using actual dvc diff.",
                DiffArgs,
                self.compare,
            ),
        ]:
            registry.add(Tool(name, description, schema, handler))
