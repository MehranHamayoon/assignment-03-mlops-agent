from unittest.mock import Mock

from fastapi.testclient import TestClient

from agent.orchestrator import Orchestrator, render_response
from agent.registry import Registry, Tool
from agent.schemas import EmptyArgs, Evidence


class Planner:
    def __init__(self, name, args=None):
        self.name, self.args, self.called = name, args or {}, False

    def next_message(self, messages, tools):
        if self.called:
            return {"role": "assistant", "content": "Made up 100% healthy result"}
        self.called = True
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": self.name, "arguments": self.args}}],
        }


def test_tool_selection_and_execution(settings):
    registry = Registry()
    handler = Mock(return_value={"datasets": ["iris.csv"]})
    registry.add(Tool("dvc_datasets", "datasets", EmptyArgs, handler))
    result = Orchestrator(settings, registry, Planner("dvc_datasets")).chat("List datasets")
    handler.assert_called_once()
    assert result.facts[0]["observed"] == {"datasets": ["iris.csv"]}
    assert "Made up" not in result.model_dump_json()


def test_planner_runs_once_and_does_not_reprocess_operational_payloads(settings):
    registry = Registry()
    registry.add(Tool("dvc_datasets", "datasets", EmptyArgs, lambda: {"datasets": ["iris"]}))
    planner = Mock()
    planner.next_message.return_value = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "dvc_datasets", "arguments": {}}},
            {"function": {"name": "dvc_datasets", "arguments": {}}},
        ],
    }
    result = Orchestrator(settings, registry, planner).chat("List datasets")
    planner.next_message.assert_called_once()
    assert len(result.evidence) == 1


def test_unavailable_tool_never_becomes_an_operational_fact(settings):
    registry = Registry()
    registry.add(
        Tool("k8s_health", "health", EmptyArgs, Mock(side_effect=ConnectionError("offline")))
    )
    result = Orchestrator(settings, registry, Planner("k8s_health")).chat("Is the API healthy?")
    assert result.facts == []
    assert any("unknown" in u for u in result.uncertainties)
    assert "100% healthy" not in result.model_dump_json()


def test_assessment_uses_four_tools_and_never_triggers_training(settings):
    engine = Orchestrator(settings, Registry(), Planner("assess_model"))
    calls = []

    def execute(name, args, request_id):
        calls.append((name, args))
        data = {"runs": [], "dag_runs": []}
        if name == "mlflow_compare":
            data = {
                "best_run": {
                    "run_id": "a" * 32,
                    "tags": {
                        "dataset.git_commit": "b" * 40,
                        "dataset.md5": "hash",
                        "dataset.dvc_file": "data/iris.csv.dvc",
                    },
                }
            }
        if name == "mlflow_run":
            data = {
                "tags": {
                    "dataset.git_commit": "b" * 40,
                    "dataset.md5": "hash",
                    "dataset.dvc_file": "data/iris.csv.dvc",
                }
            }
        return Evidence(tool=name, status="ok", data=data, request_id=request_id, duration_ms=1)

    engine.registry.execute = execute
    result = engine.chat("Find best model, data, deployment, and whether retraining is necessary")
    names = [n for n, _ in calls]
    assert all(n in names for n in ("mlflow_compare", "dvc_version", "k8s_health", "airflow_runs"))
    assert "airflow_trigger" not in names and "k8s_restart" not in names
    assert next(a for n, a in calls if n == "dvc_version")["rev"] == "b" * 40
    assert any("does not establish a need" in r for r in result.recommendations)


def test_failed_quality_gate_supports_retraining_recommendation():
    evidence = [
        Evidence(
            tool="mlflow_compare",
            status="ok",
            request_id="r",
            duration_ms=1,
            data={"best_run": {"run_id": "a" * 32, "tags": {"quality_gate": "failed"}}},
        )
    ]
    result = render_response("r", evidence)
    assert any("retraining run is warranted" in r for r in result.recommendations)


def test_agent_api_requires_authentication_and_validates_payload(settings, monkeypatch):
    from agent import main

    engine = Orchestrator(settings, Registry(), Planner("unknown"))
    monkeypatch.setattr(main, "engine", lambda: engine)
    with TestClient(main.app) as client:
        assert client.post("/chat", json={"message": "Hi"}).status_code == 401
        assert (
            client.post(
                "/chat", json={"message": ""}, headers={"X-Agent-Key": "test-agent-key"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/actions/confirm",
                json={"action_id": "f" * 32, "confirmation_text": "yes"},
                headers={"X-Agent-Key": "test-agent-key"},
            ).status_code
            == 403
        )
