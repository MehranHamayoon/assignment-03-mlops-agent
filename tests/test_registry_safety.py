import json
import logging
from unittest.mock import Mock

import httpx
import pytest

from agent.registry import Registry, Tool
from agent.safety import ConfirmationStore, redact
from agent.schemas import EmptyArgs
from agent.tools.kubernetes_tools import KubernetesTools


def test_successful_execution_logs_identifiers_without_arguments(caplog):
    registry = Registry()
    registry.add(Tool("inspect", "Inspect", EmptyArgs, lambda: {"healthy": True}))
    with caplog.at_level(logging.INFO, logger="mlops.tools"):
        result = registry.execute("inspect", {}, "request-123")
    assert result.status == "ok" and result.data["healthy"]
    record = json.loads(caplog.records[-1].message)
    assert set(record) == {"tool", "status", "duration_ms", "request_id"}
    assert record["request_id"] == "request-123"


def test_invalid_arguments_never_call_handler():
    registry = Registry()
    handler = Mock()
    registry.add(Tool("inspect", "Inspect", EmptyArgs, handler))
    result = registry.execute("inspect", {"shell": "rm -rf /"}, "r")
    assert result.error == "invalid_request"
    handler.assert_not_called()


@pytest.mark.parametrize(
    "failure,code",
    [
        (httpx.ReadTimeout("private-token"), "tool_timeout"),
        (ConnectionError("password=private-token"), "service_unavailable"),
    ],
)
def test_failures_are_sanitized(failure, code, caplog):
    registry = Registry()
    registry.add(Tool("inspect", "Inspect", EmptyArgs, Mock(side_effect=failure)))
    with caplog.at_level(logging.INFO):
        result = registry.execute("inspect", {}, "r")
    assert result.status == "error" and result.error == code and result.data is None
    assert "private-token" not in result.model_dump_json() + caplog.text


def test_namespace_and_deployment_escape_are_rejected(settings):
    core, apps = Mock(), Mock()
    registry = Registry()
    KubernetesTools(settings, core, apps).register(registry)
    for args in ({"namespace": "kube-system"}, {"deployment": "other-app"}):
        result = registry.execute("k8s_restart", args, "r")
        assert result.error == "permission_denied"
    apps.patch_namespaced_deployment.assert_not_called()


def test_llm_cannot_forge_confirmation(settings):
    core, apps = Mock(), Mock()
    registry = Registry()
    KubernetesTools(settings, core, apps).register(registry)
    result = registry.execute("k8s_restart", {"confirmed": True}, "r")
    assert result.error == "invalid_request"
    apps.patch_namespaced_deployment.assert_not_called()


def test_confirmation_is_exact_bound_and_single_use(settings):
    core, apps = Mock(), Mock()
    registry = Registry()
    KubernetesTools(settings, core, apps).register(registry)
    proposal = registry.execute("k8s_restart", {}, "r")
    assert proposal.status == "confirmation_required"
    apps.patch_namespaced_deployment.assert_not_called()
    action_id = proposal.data["action_id"]
    with pytest.raises(PermissionError):
        registry.confirm(action_id, "yes", "r")
    result = registry.confirm(action_id, f"CONFIRM {action_id}", "r")
    assert result.status == "ok"
    assert apps.patch_namespaced_deployment.call_count == 1
    with pytest.raises(PermissionError):
        registry.confirm(action_id, f"CONFIRM {action_id}", "r")


def test_expired_confirmation_is_rejected():
    store = ConfirmationStore(ttl_seconds=-1)
    proposal = store.propose("restart", {})
    with pytest.raises(PermissionError):
        store.consume(proposal["action_id"], proposal["required_confirmation"])


def test_service_logs_and_credential_keys_are_redacted():
    result = redact(
        {
            "authorization": "secret",
            "logs": "Bearer abc123 token=xyz password=qwerty",
            "nested": ["configured-secret"],
        },
        ("configured-secret",),
    )
    serialized = json.dumps(result)
    for secret in ("abc123", "xyz", "qwerty", "configured-secret"):
        assert secret not in serialized
