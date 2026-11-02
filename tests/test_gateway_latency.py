"""Opt-in gateway/worker regression check for keep-alive cache-hit latency.

The check guards the delayed-ACK failure mode: before TCP_NODELAY was enabled,
sequential cache hits through the gateway clustered near 80 ms on loopback.
The ceiling is intentionally generous so it catches that regression without
turning ordinary scheduler noise into a failure. It is not a performance claim.
"""

import json
import math
import os
import socket
import statistics
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


class GatewayLatencyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker_bin = os.environ.get("BENCH_WORKER_BIN")
        cls.gateway_bin = os.environ.get("BENCH_GATEWAY_BIN")
        cls.model = os.environ.get("BENCH_MODEL")
        if not cls.worker_bin or not cls.gateway_bin or not cls.model:
            raise unittest.SkipTest(
                "set BENCH_WORKER_BIN, BENCH_GATEWAY_BIN and BENCH_MODEL to run gateway tests"
            )

    @staticmethod
    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @staticmethod
    def wait_url(url, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    return
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        raise RuntimeError(f"timed out waiting for {url}")

    def test_cache_hits_do_not_cluster_at_delayed_ack_latency(self):
        worker_port = self.free_port()
        gateway_port = self.free_port()
        with tempfile.TemporaryDirectory(prefix="gateway-latency-test-") as tmp:
            worker_log = Path(tmp) / "worker.log"
            gateway_log = Path(tmp) / "gateway.log"
            worker = gateway = None
            try:
                with worker_log.open("w") as worker_handle:
                    worker = subprocess.Popen(
                        [self.worker_bin, str(worker_port), "gateway-test", self.model],
                        stdout=worker_handle,
                        stderr=subprocess.STDOUT,
                    )
                self.wait_url(f"http://127.0.0.1:{worker_port}/health")
                env = os.environ.copy()
                env["GATEWAY_PORT"] = str(gateway_port)
                with gateway_log.open("w") as gateway_handle:
                    gateway = subprocess.Popen(
                        [self.gateway_bin, f"localhost:{worker_port}"],
                        stdout=gateway_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                self.wait_url(f"http://127.0.0.1:{gateway_port}/stats")

                payload = json.dumps(
                    {"request_id": "cache-hit", "input_data": [1.0, 0.5, 0.25]}
                ).encode()
                latencies = []
                responses = []
                for _ in range(30):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{gateway_port}/infer",
                        payload,
                        {"Content-Type": "application/json"},
                        method="POST",
                    )
                    started = time.monotonic()
                    with urllib.request.urlopen(request, timeout=30) as response:
                        responses.append(json.load(response))
                    latencies.append((time.monotonic() - started) * 1000)

                self.assertTrue(all(item["request_id"] == "cache-hit" for item in responses))
                self.assertFalse(responses[0]["cached"])
                self.assertTrue(all(item["cached"] for item in responses[1:]))
                self.assertEqual(responses[0]["output_data"], responses[1]["output_data"])

                p50 = statistics.median(latencies)
                p95 = percentile(latencies, 0.95)
                p50_limit = float(os.environ.get("BENCH_CACHE_HIT_P50_MS", "30"))
                p95_limit = float(os.environ.get("BENCH_CACHE_HIT_P95_MS", "60"))
                self.assertLessEqual(
                    p50,
                    p50_limit,
                    f"cache-hit p50 {p50:.1f} ms exceeds {p50_limit:.1f} ms; "
                    "check keep-alive TCP_NODELAY settings",
                )
                self.assertLessEqual(
                    p95, p95_limit, f"cache-hit p95 {p95:.1f} ms exceeds {p95_limit:.1f} ms"
                )
            finally:
                for process in (gateway, worker):
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()


if __name__ == "__main__":
    unittest.main()
