"""Opt-in gateway admission-control test with a blocking fake worker."""

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class BlockingWorker:
    def __init__(self):
        self.count = 0
        self.received = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                with worker.lock:
                    worker.count += 1
                worker.received.set()
                worker.release.wait(timeout=10)
                body = json.dumps(
                    {
                        "request_id": self.headers.get("X-Request-Id", ""),
                        "output_data": [1.0],
                        "node_id": "blocking-worker",
                        "cached": False,
                        "inference_time_us": 1,
                        "queue_wait_us": 0,
                        "batch_execution_us": 1,
                        "batch_size": 1,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        self.port = free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self


class AdmissionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gateway_bin = os.environ.get("BENCH_GATEWAY_BIN")
        if not cls.gateway_bin:
            raise unittest.SkipTest("set BENCH_GATEWAY_BIN to run admission tests")
        cls.worker = BlockingWorker().start()
        cls.port = free_port()
        cls.tmp = tempfile.TemporaryDirectory(prefix="admission-test-")
        cls.log = open(Path(cls.tmp.name) / "gateway.log", "w")
        env = os.environ.copy()
        env["GATEWAY_PORT"] = str(cls.port)
        env["ROUTING_POLICY"] = "hash"
        env["ADMISSION_MAX_INFLIGHT"] = "1"
        cls.gateway = subprocess.Popen(
            [cls.gateway_bin, f"127.0.0.1:{cls.worker.port}"],
            stdout=cls.log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            cls.wait_url(f"http://127.0.0.1:{cls.port}/stats")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.worker.release.set()
        if getattr(cls, "gateway", None) is not None:
            cls.gateway.terminate()
            try:
                cls.gateway.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.gateway.kill()
                cls.gateway.wait()
        if getattr(cls, "worker", None) is not None:
            cls.worker.server.shutdown()
            cls.worker.server.server_close()
            cls.worker.thread.join(timeout=5)
        if getattr(cls, "log", None) is not None:
            cls.log.close()
        if getattr(cls, "tmp", None) is not None:
            cls.tmp.cleanup()

    @staticmethod
    def wait_url(url, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError(f"timeout waiting for {url}")

    def post(self, request_id, timeout=5):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/infer",
            json.dumps({"request_id": request_id, "input_data": [1.0, 2.0, 3.0]}).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, dict(response.headers), json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, dict(exc.headers), json.load(exc)
            finally:
                exc.close()

    def test_second_concurrent_request_is_rejected_before_the_worker(self):
        result = {}
        first = threading.Thread(
            target=lambda: result.update(first=self.post("first", timeout=10)),
            daemon=True,
        )
        first.start()
        self.assertTrue(self.worker.received.wait(timeout=5))
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/stats", timeout=5) as response:
            live_stats = json.load(response)
        self.assertEqual(live_stats["admission"]["in_flight"], 1)
        self.assertEqual(live_stats["admission"]["accepted"], 1)
        self.assertEqual(live_stats["admission"]["rejected"], 0)

        status, headers, body = self.post("second", timeout=2)
        self.assertEqual(status, 503)
        self.assertEqual(headers.get("Retry-After"), "1")
        self.assertIn("error", body)
        self.assertEqual(self.worker.count, 1)

        self.worker.release.set()
        first.join(timeout=10)
        self.assertFalse(first.is_alive())
        self.assertEqual(result["first"][0], 200)

        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/stats", timeout=5) as response:
            stats = json.load(response)
        self.assertEqual(stats["admission"]["max_in_flight"], 1)
        self.assertEqual(stats["admission"]["in_flight"], 0)
        self.assertEqual(stats["admission"]["accepted"], 1)
        self.assertEqual(stats["admission"]["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
