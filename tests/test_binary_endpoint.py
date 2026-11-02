"""Opt-in binary tensor transport test against a real worker and model."""

import json
import os
import socket
import struct
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


class BinaryEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker_bin = os.environ.get("BENCH_WORKER_BIN")
        cls.gateway_bin = os.environ.get("BENCH_GATEWAY_BIN")
        cls.model = os.environ.get("BENCH_MODEL")
        if not cls.worker_bin or not cls.gateway_bin or not cls.model:
            raise unittest.SkipTest("set BENCH_WORKER_BIN, BENCH_GATEWAY_BIN and BENCH_MODEL")
        cls.worker_port = free_port()
        cls.gateway_port = free_port()
        cls.tmp = tempfile.TemporaryDirectory(prefix="binary-endpoint-test-")
        cls.logs = []
        worker_log = open(Path(cls.tmp.name) / "worker.log", "w")
        cls.logs.append(worker_log)
        cls.worker = subprocess.Popen(
            [cls.worker_bin, str(cls.worker_port), "binary-test", cls.model],
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

    def json_infer(self, request_id, values):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.gateway_port}/infer",
            json.dumps({"request_id": request_id, "input_data": values}).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def tensor_infer(self, request_id, values):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.gateway_port}/infer_tensor",
            struct.pack(f"<{len(values)}f", *values),
            {"Content-Type": "application/octet-stream", "X-Request-Id": request_id},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return response.headers, list(struct.unpack(f"<{len(body) // 4}f", body))

    def test_binary_and_json_outputs_match(self):
        json_response = self.json_infer("json-1", [1.0, 2.0, 3.0])
        headers, binary_output = self.tensor_infer("tensor-1", [1.0, 2.0, 3.0])
        self.assertEqual(headers.get("X-Cached"), "true")
        self.assertEqual(headers.get("X-Request-Id"), "tensor-1")
        self.assertEqual(len(binary_output), len(json_response["output_data"]))
        max_difference = max(
            abs(a - b) for a, b in zip(binary_output, json_response["output_data"])
        )
        self.assertLessEqual(max_difference, 1e-5)

        headers, binary_output = self.tensor_infer("tensor-2", [4.0, 5.0, 6.0])
        self.assertEqual(headers.get("X-Cached"), "false")
        json_response = self.json_infer("json-2", [4.0, 5.0, 6.0])
        self.assertEqual(len(binary_output), len(json_response["output_data"]))
        max_difference = max(
            abs(a - b) for a, b in zip(binary_output, json_response["output_data"])
        )
        self.assertLessEqual(max_difference, 1e-5)

    def test_invalid_tensor_request_is_rejected(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.gateway_port}/infer_tensor",
            b"bad",
            {"Content-Type": "application/octet-stream", "X-Request-Id": "bad-1"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()

        request = urllib.request.Request(
            f"http://127.0.0.1:{self.gateway_port}/infer_tensor",
            struct.pack("<3f", 1.0, 2.0, 3.0),
            {"Content-Type": "application/octet-stream"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()


if __name__ == "__main__":
    unittest.main()
