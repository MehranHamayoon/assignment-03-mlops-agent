from datetime import UTC, datetime

from pydantic import Field

from agent.config import Settings
from agent.registry import Registry, Tool
from agent.schemas import StrictModel


class NamespaceArgs(StrictModel):
    namespace: str = Field(default="mlops", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")


class PodArgs(NamespaceArgs):
    pod: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,252}$")
    previous: bool = False


class RestartArgs(NamespaceArgs):
    deployment: str = Field(default="iris-api", pattern=r"^[a-z0-9][a-z0-9.-]{0,252}$")


class KubernetesTools:
    def __init__(self, settings: Settings, core=None, apps=None):
        self.settings, self._core, self._apps = settings, core, apps
        self.selector = "app=iris-api"
        self.timeout = (min(settings.tool_timeout, 5), settings.tool_timeout)

    def connect(self):
        if self._core is None or self._apps is None:
            from kubernetes.config.config_exception import ConfigException

            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except ConfigException:
                config.load_kube_config(context=self.settings.kube_context or None)
            self._core, self._apps = client.CoreV1Api(), client.AppsV1Api()

    def authorize(self, namespace="mlops", deployment=None, **_):
        if namespace != self.settings.approved_namespace:
            raise PermissionError("Namespace is not allowlisted")
        if deployment and deployment != self.settings.approved_deployment:
            raise PermissionError("Deployment is not allowlisted")

    def pod_items(self, namespace):
        self.authorize(namespace)
        self.connect()
        return self._core.list_namespaced_pod(
            namespace,
            label_selector=self.selector,
            _request_timeout=self.timeout,
        ).items

    @staticmethod
    def pod_summary(p):
        states = []
        for container in p.status.container_statuses or []:
            waiting = container.state.waiting
            terminated = container.state.terminated
            states.append(
                {
                    "name": container.name,
                    "ready": container.ready,
                    "restart_count": container.restart_count,
                    "reason": waiting.reason
                    if waiting
                    else terminated.reason
                    if terminated
                    else None,
                }
            )
        ready = next(
            (c.status == "True" for c in p.status.conditions or [] if c.type == "Ready"), False
        )
        return {
            "name": p.metadata.name,
            "phase": p.status.phase,
            "ready": ready,
            "containers": states,
            "images": [c.image for c in p.spec.containers],
        }

    def pods(self, namespace="mlops"):
        return {
            "namespace": namespace,
            "pods": [self.pod_summary(p) for p in self.pod_items(namespace)],
        }

    def deployments(self, namespace="mlops"):
        self.authorize(namespace)
        self.connect()
        items = self._apps.list_namespaced_deployment(
            namespace,
            label_selector=self.selector,
            _request_timeout=self.timeout,
        ).items
        return {
            "namespace": namespace,
            "deployments": [
                {
                    "name": d.metadata.name,
                    "desired_replicas": d.spec.replicas,
                    "ready_replicas": d.status.ready_replicas or 0,
                    "updated_replicas": d.status.updated_replicas or 0,
                    "generation": d.metadata.generation,
                    "observed_generation": d.status.observed_generation,
                    "images": [c.image for c in d.spec.template.spec.containers],
                    "model_run_ids": [
                        e.value
                        for c in d.spec.template.spec.containers
                        for e in c.env or []
                        if e.name == "MODEL_RUN_ID"
                    ],
                }
                for d in items
            ],
        }

    def services(self, namespace="mlops"):
        self.authorize(namespace)
        self.connect()
        items = self._core.list_namespaced_service(
            namespace,
            label_selector=self.selector,
            _request_timeout=self.timeout,
        ).items
        slices = self._core.list_namespaced_endpoints(
            namespace,
            label_selector=self.selector,
            _request_timeout=self.timeout,
        ).items
        return {
            "services": [
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "selector": s.spec.selector,
                    "ports": [p.port for p in s.spec.ports],
                }
                for s in items
            ],
            "endpoints": [
                {
                    "name": e.metadata.name,
                    "ready_addresses": [
                        a.ip for subset in e.subsets or [] for a in subset.addresses or []
                    ],
                }
                for e in slices
            ],
        }

    def logs(self, pod: str, namespace="mlops", previous=False):
        if pod not in [p.metadata.name for p in self.pod_items(namespace)]:
            raise PermissionError("Pod does not belong to the application")
        return {
            "pod": pod,
            "previous": previous,
            "logs": self._core.read_namespaced_pod_log(
                pod,
                namespace,
                tail_lines=80,
                limit_bytes=12000,
                previous=previous,
                _request_timeout=self.timeout,
            ),
        }

    def health(self, namespace="mlops"):
        pods = self.pods(namespace)
        deployments = self.deployments(namespace)
        services = self.services(namespace)
        events = []
        for pod in pods["pods"][:10]:
            items = self._core.list_namespaced_event(
                namespace,
                field_selector=f"involvedObject.name={pod['name']}",
                _request_timeout=self.timeout,
            ).items
            events.extend(
                {"pod": pod["name"], "reason": e.reason, "message": e.message, "count": e.count}
                for e in items[-10:]
            )
        healthy = bool(pods["pods"]) and all(p["ready"] for p in pods["pods"])
        healthy = (
            healthy
            and bool(deployments["deployments"])
            and all(
                d["desired_replicas"] >= 2
                and d["ready_replicas"] >= d["desired_replicas"]
                and d["updated_replicas"] >= d["desired_replicas"]
                and (d["observed_generation"] or 0) >= d["generation"]
                for d in deployments["deployments"]
            )
        )
        healthy = (
            healthy
            and bool(services["services"])
            and any(e["ready_addresses"] for e in services["endpoints"])
        )
        return {
            "healthy": healthy,
            **pods,
            **deployments,
            **services,
            "events": events,
            "unhealthy_pods": [p for p in pods["pods"] if not p["ready"]],
            "note": "Readiness and endpoints are observed; external reachability is separate.",
        }

    def restart(self, deployment="iris-api", namespace="mlops"):
        self.authorize(namespace, deployment)
        self.connect()
        self._apps.patch_namespaced_deployment(
            deployment,
            namespace,
            {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": datetime.now(UTC).isoformat(),
                            }
                        }
                    }
                }
            },
            _request_timeout=self.timeout,
        )
        return {
            "deployment": deployment,
            "namespace": namespace,
            "restart_requested": True,
            "rollout_succeeded": "Not yet verified",
        }

    def register(self, registry: Registry):
        for name, description, schema, handler, mutation in [
            (
                "k8s_pods",
                "List application pods, readiness, images and restart counts.",
                NamespaceArgs,
                self.pods,
                False,
            ),
            (
                "k8s_deployments",
                "List actual desired/ready replicas and deployed image versions.",
                NamespaceArgs,
                self.deployments,
                False,
            ),
            (
                "k8s_services",
                "Inspect application Services and ready endpoints.",
                NamespaceArgs,
                self.services,
                False,
            ),
            (
                "k8s_logs",
                "Read bounded application pod logs, optionally the previous container.",
                PodArgs,
                self.logs,
                False,
            ),
            (
                "k8s_health",
                "Diagnose readiness, deployment rollout, endpoints and pod events.",
                NamespaceArgs,
                self.health,
                False,
            ),
            (
                "k8s_restart",
                "Propose a namespace-scoped application restart. Requires explicit "
                "user confirmation outside the LLM.",
                RestartArgs,
                self.restart,
                True,
            ),
        ]:
            registry.add(Tool(name, description, schema, handler, mutation, self.authorize))
