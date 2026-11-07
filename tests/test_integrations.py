import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest

from agent.registry import Registry
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DvcTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MlflowTools
from tests.conftest import run


def test_mlflow_ranking_excludes_failed_and_nan_runs(settings):
    client = Mock()
    client.search_runs.return_value = [
        run("a" * 32, 0.8),
        run("b" * 32, 0.99, "FAILED"),
        run("c" * 32, float("nan")),
        run("d" * 32, 0.92),
    ]
    result = MlflowTools(settings, client).compare(experiment_ids=["1", "2"], limit=5)
    assert result["best_run"]["run_id"] == "d" * 32
    assert len(result["ranking"]) == 2
    assert "Not established" in result["production_readiness"]
    assert client.search_runs.call_args.args[0] == ["1", "2"]


def test_mlflow_missing_metric_does_not_invent_candidate(settings):
    client = Mock()
    client.search_runs.return_value = [run()]
    result = MlflowTools(settings, client).compare(metric="precision", experiment_ids=["1"])
    assert result["best_run"] is None


def test_metric_capitalization_resolves_only_to_a_unique_recorded_key(settings):
    client = Mock()
    client.search_runs.return_value = [run("a" * 32, 0.8), run("b" * 32, 0.95)]
    result = MlflowTools(settings, client).compare(metric="F1", experiment_ids=["1"])
    assert result["metric"] == "f1"
    assert result["requested_metric"] == "F1"
    assert result["best_run"]["run_id"] == "b" * 32


def test_airflow_uses_v2_authentication_and_required_logical_date(settings):
    seen = []

    def transport(request):
        seen.append(request)
        if request.url.path == "/auth/token":
            return httpx.Response(200, json={"access_token": "new-token"})
        assert request.headers["Authorization"] == "Bearer new-token"
        return httpx.Response(200, json={"dag_run_id": "manual-1", "state": "queued"})

    settings.airflow_token = ""  # Revalidate the modified setting below.
    fresh = type(settings)(**{**settings.model_dump(), "airflow_token": ""}, _env_file=None)
    client = httpx.Client(base_url="http://airflow", transport=httpx.MockTransport(transport))
    adapter = AirflowTools(fresh, client)
    result = adapter.trigger(conf={})
    assert result["state"] == "queued"
    assert seen[-1].url.path == "/api/v2/dags/iris_training/dagRuns"
    assert "logical_date" in json.loads(seen[-1].content)


def test_airflow_trigger_is_allowlisted_and_proposed_not_executed(settings):
    client = Mock()
    registry = Registry()
    AirflowTools(settings, client).register(registry)
    assert (
        registry.execute("airflow_trigger", {"dag_id": "delete_everything"}, "r").error
        == "permission_denied"
    )
    assert registry.execute("airflow_trigger", {}, "r").status == "confirmation_required"
    client.request.assert_not_called()


def test_failed_airflow_tasks_include_upstream_failures(settings):
    def transport(request):
        return httpx.Response(
            200,
            json={
                "task_instances": [
                    {"task_id": "validate", "state": "failed"},
                    {"task_id": "train", "state": "upstream_failed"},
                    {"task_id": "other", "state": "success"},
                ]
            },
        )

    client = httpx.Client(base_url="http://airflow", transport=httpx.MockTransport(transport))
    result = AirflowTools(settings, client).tasks("manual/test")
    assert len(result["failed_tasks"]) == 2


def test_dvc_rejects_path_traversal_and_option_injection(settings):
    adapter = DvcTools(settings)
    with pytest.raises(PermissionError):
        adapter.metadata_path("../secrets.dvc")
    with pytest.raises(ValueError):
        adapter.revision("--output=/tmp/pwn")


def test_dvc_version_checks_committed_hash(settings):
    adapter = DvcTools(settings)
    adapter.command = Mock(side_effect=["a" * 40, "outs:\n- md5: actual\n  path: iris.csv\n"])
    result = adapter.version(expected_md5="wrong")
    assert result["recorded_hash_matches"] is False
    assert result["git_commit"] == "a" * 40


def test_kubernetes_crash_diagnosis_records_restarts_and_events(settings):
    pod = NS(
        metadata=NS(name="iris-api-123"),
        spec=NS(containers=[NS(image="iris-api:1")]),
        status=NS(
            phase="Running",
            conditions=[NS(type="Ready", status="False")],
            container_statuses=[
                NS(
                    name="api",
                    ready=False,
                    restart_count=4,
                    state=NS(waiting=NS(reason="CrashLoopBackOff"), terminated=None),
                )
            ],
        ),
    )
    core, apps = Mock(), Mock()
    core.list_namespaced_pod.return_value.items = [pod]
    core.list_namespaced_service.return_value.items = []
    core.list_namespaced_endpoints.return_value.items = []
    core.list_namespaced_event.return_value.items = [
        NS(reason="BackOff", count=4, message="Restarting failed container")
    ]
    apps.list_namespaced_deployment.return_value.items = []
    result = KubernetesTools(settings, core, apps).health()
    assert result["healthy"] is False
    assert result["unhealthy_pods"][0]["containers"][0]["restart_count"] == 4
    assert result["events"][0]["reason"] == "BackOff"
