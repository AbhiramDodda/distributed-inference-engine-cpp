"""Opt-in integration test against a locally built worker and ONNX model."""

import importlib.util
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


class WorkerTimingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("BENCH_WORKER_BIN")
        model = os.environ.get("BENCH_MODEL")
        if not binary or not model:
            raise unittest.SkipTest("set BENCH_WORKER_BIN and BENCH_MODEL to run real worker tests")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.tmp = tempfile.TemporaryDirectory(prefix="inference-worker-test-")
        cls.log = open(Path(cls.tmp.name) / "worker.log", "w+")
        cls.process = subprocess.Popen(
            [binary, str(cls.port), "timing-test", model],
            stdout=cls.log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if cls.process.poll() is not None:
                    raise RuntimeError("worker exited before becoming healthy")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/health", timeout=1):
                        return
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            raise RuntimeError("worker health check timed out")
        except Exception:
            cls.log.flush()
            cls.log.seek(0)
            output = cls.log.read()[-3000:]
            cls.tearDownClass()
            raise RuntimeError(f"worker could not start: {output}")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait()
        cls.log.close()
        cls.tmp.cleanup()

    def infer(self, request_id):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/infer",
            data=json.dumps({"request_id": request_id, "input_data": [1, 2, 3]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    def test_miss_then_hit_have_explicit_timing_semantics(self):
        miss = self.infer("miss")
        hit = self.infer("hit")
        self.assertFalse(miss["cached"])
        self.assertTrue(hit["cached"])
        self.assertEqual(miss["request_id"], "miss")
        self.assertEqual(hit["request_id"], "hit")
        self.assertEqual(miss["output_data"], hit["output_data"])
        for field in ("inference_time_us", "queue_wait_us", "batch_execution_us"):
            self.assertIsInstance(miss[field], int)
            self.assertGreaterEqual(miss[field], 0)
        self.assertEqual(miss["batch_size"], 1)
        self.assertIsInstance(hit["inference_time_us"], int)
        for field in ("queue_wait_us", "batch_execution_us", "batch_size"):
            self.assertIsNone(hit[field])

    def test_benchmark_client_records_real_worker_timings(self):
        path = Path(__file__).resolve().parents[1] / "benchmark-aws" / "load.py"
        spec = importlib.util.spec_from_file_location("integration_load", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        summary, rows = module.run_level(f"http://127.0.0.1:{self.port}", 3, 2, False, 1000)
        self.assertEqual(summary["successes"], 3)
        self.assertEqual(summary["worker_timings_us"]["queue_wait_us"]["samples"], 3)
        self.assertEqual(summary["worker_timings_us"]["batch_execution_us"]["samples"], 3)
        self.assertTrue(all(row["batch_size"] >= 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
