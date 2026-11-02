"""Opt-in integration test against a locally built worker and ONNX model."""

import contextlib
import importlib.util
import json
import os
import re
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


class ThreadSelectionIntegrationTests(unittest.TestCase):
    """The backend thread count must be visible and must not change outputs."""

    @classmethod
    def setUpClass(cls):
        cls.binary = os.environ.get("BENCH_WORKER_BIN")
        cls.model = os.environ.get("BENCH_MODEL")
        if not cls.binary or not cls.model:
            raise unittest.SkipTest("set BENCH_WORKER_BIN and BENCH_MODEL to run real worker tests")

    @contextlib.contextmanager
    def running_worker(self, threads=None):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ.copy()
        if threads is not None:
            env["ORT_INTRA_THREADS"] = str(threads)
        else:
            env.pop("ORT_INTRA_THREADS", None)
        with tempfile.TemporaryDirectory(prefix="inference-thread-test-") as tmp:
            log_path = Path(tmp) / "worker.log"
            with log_path.open("w+") as log:
                process = subprocess.Popen(
                    [self.binary, str(port), "thread-test", self.model],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                try:
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            log.flush()
                            raise RuntimeError(
                                f"worker exited early: {log_path.read_text()[-2000:]}"
                            )
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{port}/health", timeout=1
                            ):
                                break
                        except (OSError, urllib.error.URLError):
                            time.sleep(0.1)
                    else:
                        log.flush()
                        raise RuntimeError(
                            f"worker health check timed out: {log_path.read_text()[-2000:]}"
                        )
                    yield port, log_path
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    @staticmethod
    def infer(port, request_id, values):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/infer",
            data=json.dumps({"request_id": request_id, "input_data": values}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def logged_threads(self, log_path):
        text = log_path.read_text()
        match = re.search(r"ORT intra-op threads: (\d+)", text)
        self.assertIsNotNone(match, f"missing thread log line in:\n{text[-2000:]}")
        return int(match.group(1))

    def test_default_thread_count_matches_cpu_capacity(self):
        expected = min(4, os.cpu_count() or 1)
        with self.running_worker() as (_, log_path):
            self.assertEqual(self.logged_threads(log_path), expected)

    def test_two_and_four_threads_produce_equivalent_outputs(self):
        inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        outputs = {}
        for threads in (2, 4):
            with self.running_worker(threads) as (port, log_path):
                self.assertEqual(self.logged_threads(log_path), threads)
                outputs[threads] = [
                    self.infer(port, f"{threads}-{index}", values)["output_data"]
                    for index, values in enumerate(inputs)
                ]
        for two, four in zip(outputs[2], outputs[4]):
            self.assertEqual(len(two), len(four))
            max_difference = max(abs(a - b) for a, b in zip(two, four))
            self.assertLessEqual(max_difference, 1e-4)


if __name__ == "__main__":
    unittest.main()
