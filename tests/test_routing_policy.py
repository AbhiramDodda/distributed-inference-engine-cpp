"""Opt-in gateway routing test with two fake workers."""

import collections
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeWorker:
    def __init__(self, node_id):
        self.node_id = node_id
        self.count = 0
        self.lock = threading.Lock()
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                with worker.lock:
                    worker.count += 1
                body = struct.pack("<f", 1.0)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Request-Id", self.headers.get("X-Request-Id", ""))
                self.send_header("X-Node-Id", worker.node_id)
                self.send_header("X-Cached", "false")
                self.send_header("X-Output-Elements", "1")
                self.send_header("X-Batch-Size", "1")
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

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class RoutingPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gateway_bin = os.environ.get("BENCH_GATEWAY_BIN")
        if not cls.gateway_bin:
            raise unittest.SkipTest("set BENCH_GATEWAY_BIN to run routing tests")
        cls.workers = [FakeWorker("worker_1").start(), FakeWorker("worker_2").start()]
        cls.port = free_port()
        cls.tmp = tempfile.TemporaryDirectory(prefix="routing-test-")
        cls.log = open(Path(cls.tmp.name) / "gateway.log", "w")
        env = os.environ.copy()
        env["GATEWAY_PORT"] = str(cls.port)
        env["ROUTING_POLICY"] = "least_loaded"
        cls.gateway = subprocess.Popen(
            [
                cls.gateway_bin,
                f"127.0.0.1:{cls.workers[0].port}",
                f"127.0.0.1:{cls.workers[1].port}",
            ],
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
        for process in (getattr(cls, "gateway", None),):
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for worker in getattr(cls, "workers", []):
            worker.stop()
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

    def test_stats_reports_the_selected_policy(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/stats", timeout=5) as response:
            stats = json.load(response)
        self.assertEqual(stats.get("routing_policy"), "least_loaded")

    def test_least_loaded_spreads_sequential_requests(self):
        for index in range(40):
            body = struct.pack("<2f", float(index), float(index + 1))
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/infer_tensor",
                body,
                {
                    "Content-Type": "application/octet-stream",
                    "X-Request-Id": f"routing-{index}",
                },
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
        counts = collections.Counter({worker.node_id: worker.count for worker in self.workers})
        self.assertGreaterEqual(min(counts.values()), 16)
        self.assertLessEqual(max(counts.values()), 24)


if __name__ == "__main__":
    unittest.main()
