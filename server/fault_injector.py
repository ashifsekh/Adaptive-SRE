import httpx
import random
from typing import Tuple, Optional
from .service_graph import ServiceGraph


class FaultInjector:
    FAULT_TYPES = ["oom_kill", "crash_loop", "network_partition", "connection_exhaustion"]

    SERVICE_PORTS = {
        "db": 15432,
        "auth": 8102,
        "payment": 8101,
        "cache": 6379,
        "notification": 8103
    }

    SERVICE_DISPLAY = {
        "db": "db-service",
        "auth": "auth-service",
        "payment": "payment-service",
        "cache": "cache-service",
        "notification": "notification-service"
    }

    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url

    def _post_crash(self, service_name: str) -> bool:
        port = self.SERVICE_PORTS.get(service_name)
        if not port:
            return False
        try:
            url = f"{self.base_url}:{port}/crash"
            with httpx.Client(timeout=5) as client:
                resp = client.post(url)
                resp.raise_for_status()
                return True
        except:
            return False

    def _get_health(self, service_name: str) -> Optional[dict]:
        port = self.SERVICE_PORTS.get(service_name)
        if not port:
            return {"error_rate": 0.9}
        try:
            url = f"{self.base_url}:{port}/health"
            with httpx.Client(timeout=3) as client:
                resp = client.get(url)
                resp.raise_for_status()
                import json
                return json.loads(resp.text)
        except Exception:
            # Synthetic fallback — training works without live services
            return {"error_rate": 0.9, "health": 0.1, "latency_ms": 2000}

    def inject_cascade(self, service_graph: ServiceGraph, root_service: str, fault_type: str) -> str:
        self._post_crash(root_service)
        service_graph.apply_fault(root_service, fault_type)

        health_data = self._get_health(root_service) or {}
        error_rate = health_data.get("error_rate", 0.9)
        error_pct = int(error_rate * 100)

        downstream_affected = []
        from .service_graph import DEPENDENCY_GRAPH
        if root_service in DEPENDENCY_GRAPH:
            for downstream in DEPENDENCY_GRAPH[root_service].keys():
                if downstream in service_graph.services:
                    downstream_affected.append(downstream)

        alert_templates = {
            "db": f"[CRITICAL] P1 Incident — {root_service} connection pool exhausted.\nAlert: 847 failed connections in last 60s. Error rate: {error_pct}%.\nDownstream: auth-service showing elevated latency.",
            "auth": f"[CRITICAL] P1 Incident — {root_service} service authentication failures.\nAlert: Token validation failing at {error_pct}% rate.\nDownstream: payment-service experiencing timeouts.",
            "payment": f"[CRITICAL] P1 Incident — {root_service} transaction processing degraded.\nAlert: {error_pct}% of transactions failing.\nDownstream: api-gateway reporting 503 errors.",
            "cache": f"[CRITICAL] P1 Incident — {root_service} memory pressure critical.\nAlert: Hit rate dropped to {max(5, int((1-error_rate)*100))}%. OOM killer invoked.\nDownstream: notification-service queue backing up.",
            "notification": f"[CRITICAL] P1 Incident — {root_service} queue overflow.\nAlert: {error_pct}% message delivery failure. Queue depth: 8471.\nDownstream: None (leaf service)."
        }

        return alert_templates.get(root_service, f"[CRITICAL] P1 Incident — {root_service} service degraded.\nError rate: {error_pct}%. Immediate action required.")

    def inject_coincident(self, service_graph: ServiceGraph, service1: str, service2: str,
                          fault1: str, fault2: str) -> str:
        self._post_crash(service1)
        self._post_crash(service2)
        service_graph.apply_fault(service1, fault1)
        service_graph.apply_fault(service2, fault2)

        health1 = self._get_health(service1) or {}
        health2 = self._get_health(service2) or {}
        error1 = int(health1.get("error_rate", 0.9) * 100)
        error2 = int(health2.get("error_rate", 0.9) * 100)

        alert = f"""[CRITICAL] P1 Incident — Multiple simultaneous failures detected.

Alert 1: {service1} service degraded
  - Error rate: {error1}%
  - Fault type: {fault1.replace('_', ' ')}

Alert 2: {service2} service degraded  
  - Error rate: {error2}%
  - Fault type: {fault2.replace('_', ' ')}

NOTE: These appear to be independent failures. Investigate both root causes."""

        return alert

    def inject_random_fault(self, service_graph: ServiceGraph) -> str:
        root_service = random.choice(list(service_graph.services.keys()))
        fault_type = random.choice(self.FAULT_TYPES)
        return self.inject_cascade(service_graph, root_service, fault_type)
