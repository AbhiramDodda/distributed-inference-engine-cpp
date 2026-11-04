"""Opt-in worker and gateway graceful-shutdown tests."""

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class GracefulShutdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker_bin = os.environ.get("BENCH_WORKER_BIN")
        cls.gateway_bin = os.environ.get("BENCH_GATEWAY_BIN")
        cls.model = os.environ.get("BENCH_MODEL")
        if not cls.worker_bin or not cls.gateway_bin or not cls.model:
            raise unittest.SkipTest("set BENCH_WORKER_BIN, BENCH_GATEWAY_BIN and BENCH_MODEL")

    def wait_url(self, url):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError(f"timeout waiting for {url}")

    def infer(self, port):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/infer",
            json.dumps({"request_id": "shutdown", "input_data": [1, 2, 3]}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def test_worker_exits_after_sigterm(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as tmp:
            with Path(tmp, "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [self.worker_bin, str(port), "shutdown-worker", self.model],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            try:
                self.wait_url(f"http://127.0.0.1:{port}/health")
                self.infer(port)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_gateway_exits_after_sigterm(self):
        worker_port = free_port()
        gateway_port = free_port()
        with tempfile.TemporaryDirectory() as tmp:
            with Path(tmp, "worker.log").open("w") as worker_log:
                worker = subprocess.Popen(
                    [self.worker_bin, str(worker_port), "shutdown-worker", self.model],
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                )
            env = os.environ.copy()
            env["GATEWAY_PORT"] = str(gateway_port)
            with Path(tmp, "gateway.log").open("w") as gateway_log:
                gateway = subprocess.Popen(
                    [self.gateway_bin, f"localhost:{worker_port}"],
                    stdout=gateway_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            try:
                self.wait_url(f"http://127.0.0.1:{worker_port}/health")
                self.wait_url(f"http://127.0.0.1:{gateway_port}/stats")
                gateway.send_signal(signal.SIGTERM)
                self.assertEqual(gateway.wait(timeout=5), 0)
            finally:
                for process in (gateway, worker):
                    if process.poll() is None:
                        process.kill()
                        process.wait()


if __name__ == "__main__":
    unittest.main()
