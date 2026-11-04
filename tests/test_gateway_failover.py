"""Opt-in loopback failover and circuit-breaker test."""

import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class GatewayFailoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker_bin = os.environ.get("BENCH_WORKER_BIN")
        cls.gateway_bin = os.environ.get("BENCH_GATEWAY_BIN")
        cls.model = os.environ.get("BENCH_MODEL")
        if not cls.worker_bin or not cls.gateway_bin or not cls.model:
            raise unittest.SkipTest("set BENCH_WORKER_BIN, BENCH_GATEWAY_BIN and BENCH_MODEL")
        cls.tmp = tempfile.TemporaryDirectory(prefix="gateway-failover-")
        cls.logs = []
        cls.worker_a_port = free_port()
        cls.worker_b_port = free_port()
        cls.gateway_port = free_port()
        cls.worker_a = cls.start_worker(cls.worker_a_port, "worker_a")
        cls.worker_b = cls.start_worker(cls.worker_b_port, "worker_b")
        env = os.environ.copy()
        env["GATEWAY_PORT"] = str(cls.gateway_port)
        env["ROUTING_POLICY"] = "hash"
        gateway_log = open(Path(cls.tmp.name) / "gateway.log", "w")
        cls.logs.append(gateway_log)
        cls.gateway = subprocess.Popen(
            [
                cls.gateway_bin,
                f"localhost:{cls.worker_a_port}",
                f"localhost:{cls.worker_b_port}",
            ],
            stdout=gateway_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            cls.wait_url(f"http://127.0.0.1:{cls.worker_a_port}/health")
            cls.wait_url(f"http://127.0.0.1:{cls.worker_b_port}/health")
            cls.wait_url(f"http://127.0.0.1:{cls.gateway_port}/stats")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def start_worker(cls, port, node_id):
        log = open(Path(cls.tmp.name) / f"{node_id}.log", "w")
        cls.logs.append(log)
        return subprocess.Popen(
            [cls.worker_bin, str(port), node_id, cls.model],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    @classmethod
    def tearDownClass(cls):
        for process in (
            getattr(cls, "gateway", None),
            getattr(cls, "worker_b", None),
            getattr(cls, "worker_a", None),
        ):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
        for log in getattr(cls, "logs", []):
            log.close()
        if getattr(cls, "tmp", None) is not None:
            cls.tmp.cleanup()

    @staticmethod
    def wait_url(url, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError(f"timeout waiting for {url}")

    def post(self, payload):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.gateway_port}/infer",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.load(exc)
            finally:
                exc.close()

    def get_json(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.gateway_port}{path}", timeout=5
        ) as response:
            return json.load(response)

    def test_request_retries_on_healthy_worker(self):
        target = None
        for index in range(20):
            payload = {
                "request_id": f"route-{index}",
                "input_data": [1000.0 + index, 0.5, 0.25],
            }
            status, body = self.post(payload)
            self.assertEqual(status, 200)
            if body["node_id"] == "worker_a":
                target = payload
                break
        self.assertIsNotNone(target, "could not find an input routed to worker_a")

        self.worker_a.terminate()
        self.worker_a.wait(timeout=5)
        status, body = self.post(target)
        self.assertEqual(status, 200)
        self.assertIsInstance(body["output_data"], list)
        self.assertEqual(body["node_id"], "worker_b")

        status, _ = self.post(target)
        self.assertEqual(status, 200)

        stats = self.get_json("/stats")
        breaker = next(
            item
            for item in stats["circuit_breakers"]
            if item["node"].endswith(f":{self.worker_a_port}")
        )
        self.assertGreaterEqual(breaker["failures"], 1)


if __name__ == "__main__":
    unittest.main()
