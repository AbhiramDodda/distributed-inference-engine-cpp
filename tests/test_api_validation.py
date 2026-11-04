"""Opt-in request-validation tests against a real worker and gateway."""

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


class ApiValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker_bin = os.environ.get("BENCH_WORKER_BIN")
        cls.gateway_bin = os.environ.get("BENCH_GATEWAY_BIN")
        cls.model = os.environ.get("BENCH_MODEL")
        if not cls.worker_bin or not cls.gateway_bin or not cls.model:
            raise unittest.SkipTest("set BENCH_WORKER_BIN, BENCH_GATEWAY_BIN and BENCH_MODEL")
        cls.worker_port = free_port()
        cls.gateway_port = free_port()
        cls.tmp = tempfile.TemporaryDirectory(prefix="api-validation-")
        cls.logs = []
        worker_log = open(Path(cls.tmp.name) / "worker.log", "w")
        cls.logs.append(worker_log)
        cls.worker = subprocess.Popen(
            [cls.worker_bin, str(cls.worker_port), "validation-worker", cls.model],
            stdout=worker_log,
            stderr=subprocess.STDOUT,
        )
        env = os.environ.copy()
        env["GATEWAY_PORT"] = str(cls.gateway_port)
        gateway_log = open(Path(cls.tmp.name) / "gateway.log", "w")
        cls.logs.append(gateway_log)
        cls.gateway = subprocess.Popen(
            [cls.gateway_bin, f"localhost:{cls.worker_port}"],
            stdout=gateway_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            cls.wait_url(f"http://127.0.0.1:{cls.worker_port}/health")
            cls.wait_url(f"http://127.0.0.1:{cls.gateway_port}/stats")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        for process in (getattr(cls, "gateway", None), getattr(cls, "worker", None)):
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
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

    def post(self, port, payload):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/infer",
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

    def get_json(self, port, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
            return json.load(response)

    def test_gateway_and_worker_reject_invalid_fields_with_400(self):
        for port in (self.gateway_port, self.worker_port):
            status, body = self.post(port, {"request_id": "x"})
            self.assertEqual(status, 400)
            self.assertIn("error", body)

            status, body = self.post(port, {"request_id": "x", "input_data": ["bad"]})
            self.assertEqual(status, 400)

            status, body = self.post(port, {"request_id": "", "input_data": [1.0]})
            self.assertEqual(status, 400)

    def test_valid_request_remains_unchanged_and_breakers_are_clean(self):
        status, body = self.post(
            self.gateway_port,
            {"request_id": "ok", "input_data": [1.0, 2.0, 3.0]},
        )
        self.assertEqual(status, 200)
        self.assertIsInstance(body["output_data"], list)
        self.assertEqual(body["request_id"], "ok")

        stats = self.get_json(self.gateway_port, "/stats")
        self.assertTrue(all(item["failures"] == 0 for item in stats["circuit_breakers"]))


if __name__ == "__main__":
    unittest.main()
