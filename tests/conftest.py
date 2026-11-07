from types import SimpleNamespace as NS

import pytest

from agent.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        repo_root=tmp_path,
        agent_api_key="test-agent-key",
        airflow_token="test-airflow-token",
        tool_timeout=0.2,
    )


def run(run_id="a" * 32, f1=0.9, status="FINISHED", start=1000, tags=None):
    return NS(
        info=NS(run_id=run_id, experiment_id="1", status=status, start_time=start),
        data=NS(metrics={"f1": f1}, params={"c": "1"}, tags=tags or {}),
    )
