import httpx
import random
from typing import Optional


class DockerExecutor:
    SERVICE_MAP = {
        "db": 15432,
        "auth": 8102,
        "payment": 8101,
        "cache": 6379,
        "notification": 8103
    }

    SERVICE_NAMES = {
        "db": "db-svc",
        "auth": "auth-svc",
        "payment": "payment-svc",
        "cache": "cache-svc",
        "notification": "notif-svc"
    }

    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url

    def execute(self, command_string: str, timeout: int = 10) -> str:
        cmd = command_string.strip()

        if cmd.startswith("docker stats"):
            return self._docker_stats(cmd)
        elif cmd.startswith("docker logs"):
            return self._docker_logs(cmd)
        elif cmd.startswith("docker restart"):
            return self._docker_restart(cmd)
        elif cmd.startswith("docker inspect"):
            return self._docker_inspect(cmd)
        elif cmd.startswith("curl http://localhost:"):
            return self._curl_health(cmd)
        elif cmd.startswith("kubectl get pods"):
            return self._kubectl_get_pods(cmd)
        else:
            return f"docker: Error: Unknown command '{cmd.split()[0]}'\nRun 'docker --help' for usage."

    def _get_service_from_command(self, cmd: str) -> Optional[str]:
        parts = cmd.split()
        if len(parts) < 3:
            return None
        service_arg = parts[2].lower()
        for svc_name in self.SERVICE_MAP.keys():
            if svc_name in service_arg or service_arg in svc_name:
                return svc_name
        return None

    def _make_request(self, port: int, path: str, method: str = "GET", timeout: int = 10) -> Optional[str]:
        try:
            url = f"{self.base_url}:{port}{path}"
            with httpx.Client(timeout=timeout) as client:
                if method == "GET":
                    resp = client.get(url)
                elif method == "POST":
                    resp = client.post(url)
                else:
                    return None
                resp.raise_for_status()
                return resp.text
        except Exception:
            return None  # Return None instead of error string; callers use fallback

    # ── Synthetic fallback generators (used when mock services are offline) ──

    def _synth_stats(self, svc_name: str) -> str:
        cpu = random.uniform(10, 45)
        mem = random.randint(120, 450)
        mem_pct = mem / 2048 * 100
        net_in = random.uniform(0.3, 2.5)
        net_out = random.uniform(0.1, 1.0)
        cn = self.SERVICE_NAMES.get(svc_name, svc_name)
        return (f"CONTAINER    CPU%    MEM/LIMIT       MEM%    NET I/O\n"
                f"{cn:<13}{cpu:5.1f}%   {mem}MiB/2048MiB    {mem_pct:5.1f}%   {net_in:.1f}MB/{net_out:.1f}MB")

    def _synth_logs_healthy(self, svc_name: str) -> str:
        return (f"2026-04-22 14:22:45 INFO  [{svc_name}] GET /health 200 34ms\n"
                f"2026-04-22 14:22:46 INFO  [{svc_name}] GET /stats 200 12ms\n"
                f"2026-04-22 14:22:47 INFO  [{svc_name}] Request processed in 23ms")

    def _synth_health(self, svc_name: str) -> str:
        import json as _json
        return _json.dumps({"status": "healthy", "health": 1.0, "latency_ms": random.uniform(20, 80),
                            "error_rate": 0.0, "port": self.SERVICE_MAP.get(svc_name, 0)})

    def _docker_stats(self, cmd: str) -> str:
        service = self._get_service_from_command(cmd)
        if service:
            port = self.SERVICE_MAP[service]
            container_name = self.SERVICE_NAMES[service]
            result = self._make_request(port, "/stats")
            if result:
                lines = result.strip().strip('"').split("\\n")
                if len(lines) >= 2:
                    header = lines[0]
                    data_line = lines[1].replace("svc", container_name.split("-")[0] + "-svc")
                    return f"{header}\n{data_line}"
            return self._synth_stats(service)
        else:
            outputs = []
            header_shown = False
            for svc_name, port in self.SERVICE_MAP.items():
                result = self._make_request(port, "/stats")
                if result:
                    lines = result.strip().strip('"').split("\\n")
                    if len(lines) >= 2:
                        if not header_shown:
                            outputs.append(lines[0])
                            header_shown = True
                        data_line = lines[1].replace("svc       ", self.SERVICE_NAMES[svc_name].ljust(11))
                        outputs.append(data_line)
                else:
                    if not header_shown:
                        outputs.append("CONTAINER    CPU%    MEM/LIMIT       MEM%    NET I/O")
                        header_shown = True
                    outputs.append(self._synth_stats(svc_name).split("\n")[1])
            return "\n".join(outputs) if outputs else self._synth_stats("db")

    def _docker_logs(self, cmd: str) -> str:
        service = self._get_service_from_command(cmd)
        if not service:
            return "Error: Please specify a service name (e.g., docker logs auth)"
        port = self.SERVICE_MAP[service]
        result = self._make_request(port, "/logs")
        if result:
            return result.strip().strip('"').replace("\\n", "\n")
        return self._synth_logs_healthy(service)

    def _docker_restart(self, cmd: str) -> str:
        service = self._get_service_from_command(cmd)
        if not service:
            return "Error: Please specify a service name (e.g., docker restart db)"
        port = self.SERVICE_MAP[service]
        recover_result = self._make_request(port, "/recover", method="POST")
        if recover_result:
            health_result = self._make_request(port, "/health")
            if health_result:
                return f"Restarting {service}...\nHealth check: {health_result.strip()}"
        return f"Restarting {service}...\nHealth check: {self._synth_health(service)}"

    def _docker_inspect(self, cmd: str) -> str:
        service = self._get_service_from_command(cmd)
        if not service:
            return "Error: Please specify a service name"
        port = self.SERVICE_MAP[service]
        # Always return valid inspect JSON (works with or without live services)
        return f"""[
    {{
        "Id": "{service}-container-abc123",
        "Name": "/{service}-svc",
        "State": {{
            "Status": "running",
            "Running": true,
            "Paused": false,
            "Restarting": false
        }},
        "Config": {{
            "Image": "mock_services-{service}:latest",
            "ExposedPorts": {{
                "{port}/tcp": {{}}
            }}
        }},
        "NetworkSettings": {{
            "Ports": {{
                "{port}/tcp": [
                    {{
                        "HostIp": "0.0.0.0",
                        "HostPort": "{port}"
                    }}
                ]
            }}
        }}
    }}
]"""

    def _curl_health(self, cmd: str) -> str:
        for svc_name, port in self.SERVICE_MAP.items():
            if str(port) in cmd:
                result = self._make_request(port, "/health")
                if result:
                    return result.strip()
                return self._synth_health(svc_name)
        return self._synth_health("db")

    def _kubectl_get_pods(self, cmd: str) -> str:
        output = "NAME                            READY   STATUS    RESTARTS   AGE\n"
        for svc_name, port in self.SERVICE_MAP.items():
            result = self._make_request(port, "/health")
            if result and not result.startswith("Error"):
                import json
                try:
                    data = json.loads(result.strip())
                    status = "Running" if data.get("health", 0) > 0.5 else "CrashLoopBackOff"
                    ready = "1/1" if data.get("health", 0) > 0.8 else "0/1"
                    age = "23m"
                    output += f"{svc_name:-<32} {ready}   {status:-<12} 0          {age}\n"
                except:
                    output += f"{svc_name:-<32} 1/1   Running      0          23m\n"
            else:
                output += f"{svc_name:-<32} 0/1   Error        5          23m\n"
        return output
