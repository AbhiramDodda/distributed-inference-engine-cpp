# Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the inference service for safe operation without changing valid inference behavior.

**Architecture:** Add small shared headers for shutdown signals and request validation, make the batch processor fail queued work instead of leaving handlers blocked, add loopback failover and API tests, and run the real worker/gateway in CI with the pinned ONNX Runtime. The existing gateway, cache, batching, and ORT backend remain the architecture.

**Tech Stack:** C++17, cpp-httplib 0.14.3, nlohmann/json 3.11.3, ONNX Runtime 1.23.2, Python 3.12 `unittest`, GitHub Actions.

**Spec:** `docs/investigation/CONTINUOUS-OPTIMIZATION.md` (safety and evidence rules), `docs/investigation/EXPERIMENTS.md` (accepted baseline and stop condition), and `docs/investigation/FINDINGS.md` (verified state).

## Current state review

The repository is clean at `7602e31`, and `origin/main` matches. Five accepted changes are already committed:

1. `eb7ba33` — capacity-aware ORT intra-op threads.
2. `0e6bf8c` — `TCP_NODELAY` on worker/gateway keep-alive paths.
3. `b35501d` — oversized-input batch alignment fix.
4. `d2a49ed` — 10 ms batch deadline with `BATCH_DEADLINE_MS`.
5. `e058d9f` — gateway response passthrough.

Verified evidence:

- Default Python suite: 33 passed, 3 intended skips.
- Six opt-in worker/gateway tests pass with the real model.
- `ctest` passes `batch_timings` and `batch_shapes` with the real model.
- Thirteen AWS runs: estimated **$0.044703** of the $5 target; all cleanup records verified; no active tagged AWS resources.
- The resume is one page; the IPhO version has two pages; links, extracted text, and bullet-tail checks pass.

The performance stop condition in `EXPERIMENTS.md` is still valid: no remaining single-host micro-optimization has a demonstrated material gain. The remaining material gaps are:

- Shutdown can leave queued requests blocked because `BatchProcessor::stop()` abandons the queue.
- Malformed requests become worker exceptions and gateway retries, so clients can receive 500 and circuit breakers can record client errors.
- No loopback test covers worker failure, gateway retry, or circuit-breaker state.
- CI does not build the worker/gateway or run the real-model tests.
- The gateway passthrough trust tradeoff is documented but not tested as an explicit behavior.

## Global Constraints

- Preserve upstream attribution: the gateway, cache, batching, and ORT backend remain upstream work.
- Do not add a public service, GPU, Kubernetes, multi-host experiment, or new paid service.
- All new integration tests use loopback and ephemeral ports; do not bind fixed port 8000 or 8001.
- Valid response bodies, `request_id`, `output_data`, cache flags, and timing fields must remain unchanged.
- Keep the one-page resume rules and the IPhO certificate attachment if the resume is edited.
- Commit after each task. Push only after the full suite and owner review, unless the owner confirms immediate push.
- Keep raw logs under the ignored `benchmark-aws/runs/` directory; do not add secrets, keys, or account identifiers.

## Review Focus

1. **Shutdown with queued requests:** every handler must finish or fail within a bounded time; no `future.get()` may hang after `stop()`.
2. **Malformed or missing fields:** the gateway must return 400 before routing, must not retry across workers, and must not increment circuit-breaker failures.
3. **Worker failure during a request:** the gateway must retry a healthy worker, return a valid response, and record the failed worker without returning a duplicate or partial body.
4. **Port and CI environment:** tests must not depend on fixed ports or local-only `/tmp` paths; CI must fetch the pinned submodules and ONNX Runtime tarball.
5. **Response compatibility:** valid requests must still return the same JSON body and timing semantics after the hardening changes.

---

### Task 1: Make batch-processor shutdown bounded and explicit

**Files:**
- Modify: `include/batch_processor.h`
- Modify: `tests/test_batch_timings.cpp`

**Interfaces:**
- Consumes: existing `BatchProcessor<Request, Response>` template.
- Produces: `stop()` remains `void`; `process(const Request&)` throws `std::runtime_error` when the processor is stopped; queued promises receive `std::runtime_error("batch processor stopped")`.

- [ ] **Step 1: Add the failing shutdown test**

Append this case to `tests/test_batch_timings.cpp`:

```cpp
// Shutdown fails queued requests and rejects new work instead of hanging.
{
    std::promise<void> started;
    std::promise<void> release;
    auto release_future = release.get_future().share();
    BatchProcessor<int, int> processor(1, 2s, [&](const std::vector<int>& values) {
        started.set_value();
        release_future.wait();
        return values;
    });
    processor.start();
    auto first = std::async(std::launch::async, [&] { return processor.process(1); });
    require(started.get_future().wait_for(2s) == std::future_status::ready);
    auto second = std::async(std::launch::async, [&] { return processor.process(2); });
    const auto enqueue_deadline = std::chrono::steady_clock::now() + 2s;
    while (processor.getMetrics().total_requests < 2 &&
           std::chrono::steady_clock::now() < enqueue_deadline) {
        std::this_thread::yield();
    }
    require(processor.getMetrics().total_requests == 2);
    auto stopping = std::async(std::launch::async, [&] { processor.stop(); });
    release.set_value();
    require(first.wait_for(2s) == std::future_status::ready);
    require(stopping.wait_for(2s) == std::future_status::ready);
    bool second_failed = false;
    try {
        second.get();
    } catch (const std::runtime_error&) {
        second_failed = true;
    }
    require(second_failed);
    bool new_work_rejected = false;
    try {
        processor.process(3);
    } catch (const std::runtime_error&) {
        new_work_rejected = true;
    }
    require(new_work_rejected);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```sh
g++ -std=c++17 -O2 -pthread -Iinclude tests/test_batch_timings.cpp \
  -o /tmp/test_batch_timings
/tmp/test_batch_timings
```

Expected: FAIL or hang because the queued second request is abandoned and `process(3)` does not reject.

- [ ] **Step 3: Implement bounded shutdown**

In `include/batch_processor.h`:

- In `process()`, after locking `queue_mutex_`, throw `std::runtime_error("batch processor is stopped")` when `running_` is false. Do not enqueue.
- In `stop()`, set `running_ = false` while holding `queue_mutex_`, call `queue_cv_.notify_all()`, join the worker thread, then move every remaining `QueuedRequest` out of `request_queue_` and set each promise's exception to `std::runtime_error("batch processor stopped")`.
- Make `stop()` idempotent. A second call must not join an already joined thread or fail.
- Keep `processingLoop()` exiting on `!running_`; the post-join queue cleanup owns the remaining promises.

- [ ] **Step 4: Run the test and the default suite**

Run:

```sh
g++ -std=c++17 -O2 -pthread -Iinclude tests/test_batch_timings.cpp \
  -o /tmp/test_batch_timings
/tmp/test_batch_timings
python3 -m unittest discover -s tests -v
```

Expected: the C++ test passes; Python suite remains 33 passed with 3 intended skips.

- [ ] **Step 5: Commit**

```sh
git add include/batch_processor.h tests/test_batch_timings.cpp
git commit -m "Fail queued work cleanly on batch-processor shutdown"
```

---

### Task 2: Add bounded graceful shutdown to worker and gateway

**Files:**
- Create: `include/shutdown_signal.h`
- Modify: `src/worker_node.cpp`
- Modify: `src/gateway.cpp`
- Modify: `benchmark-aws/controller.py`
- Modify: `benchmark-aws/verify_source.py`
- Modify: `tests/test_source_delivery.py`
- Create: `tests/test_graceful_shutdown.py`

**Interfaces:**
- Consumes: `httplib::Server::stop()`, `httplib::Server::wait_until_ready()`, `httplib::Server::is_running()`.
- Produces: `include/shutdown_signal.h` with `void block_shutdown_signals();` and `int wait_for_shutdown_signal();`.

- [ ] **Step 1: Write the failing shutdown integration test**

Create `tests/test_graceful_shutdown.py` with this structure:

```python
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
            raise unittest.SkipTest(
                "set BENCH_WORKER_BIN, BENCH_GATEWAY_BIN and BENCH_MODEL"
            )

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
```

Use `subprocess.Popen`, `process.send_signal(signal.SIGTERM)`, and `process.wait(timeout=5)`. Always terminate any still-running process in `finally`.

- [ ] **Step 2: Run the test to verify it fails**

```sh
BENCH_WORKER_BIN=/path/to/worker_node \
BENCH_GATEWAY_BIN=/path/to/gateway \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_graceful_shutdown -v
```

Expected: FAIL or timeout because the servers block in `server.listen()` and the default SIGTERM handling does not run the new shutdown path.

- [ ] **Step 3: Add `include/shutdown_signal.h`**

Implement header-only functions:

```cpp
inline void block_shutdown_signals() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    if (pthread_sigmask(SIG_BLOCK, &set, nullptr) != 0) {
        throw std::runtime_error("failed to block shutdown signals");
    }
}

inline int wait_for_shutdown_signal() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    sigaddset(&set, SIGTERM);
    int signal_number = 0;
    if (sigwait(&set, &signal_number) != 0) {
        throw std::runtime_error("failed to wait for a shutdown signal");
    }
    return signal_number;
}
```

Include `<csignal>`, `<pthread.h>`, and `<stdexcept>`.

- [ ] **Step 4: Wire shutdown into `src/worker_node.cpp`**

- Call `block_shutdown_signals()` before constructing `WorkerNode` so the ORT threads inherit the blocked mask.
- Start `server.listen("0.0.0.0", port)` in a `std::thread`.
- Call `server.wait_until_ready()`. If `!server.is_running()`, join and return 1.
- Call `wait_for_shutdown_signal()`.
- Call `server.stop()`, join the server thread, then destroy the `WorkerNode` so the batch processor runs its bounded stop.
- Log the received signal number and return 0.

- [ ] **Step 5: Wire shutdown into `src/gateway.cpp`**

- Call `block_shutdown_signals()` before creating `Gateway` and the server thread.
- Start `server.listen("0.0.0.0", listen_port)` in a `std::thread`.
- Wait until ready; on failure, join and return 1.
- Call `wait_for_shutdown_signal()`, `server.stop()`, join, and return 0.

- [ ] **Step 6: Deliver the new header in the AWS snapshot**

Add `include/shutdown_signal.h` to `SOURCE_FILES` in `benchmark-aws/controller.py` and `benchmark-aws/verify_source.py`. Extend `tests/test_source_delivery.py` so the allowlist test includes the new header and the instrumented patch contains `block_shutdown_signals`.

- [ ] **Step 7: Run the shutdown test and full checks**

```sh
BENCH_WORKER_BIN=/tmp/worker_node \
BENCH_GATEWAY_BIN=/tmp/gateway \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_graceful_shutdown -v
python3 -m unittest discover -s tests -v
```

Expected: both shutdown tests pass; default suite remains green.

- [ ] **Step 8: Commit**

```sh
git add include/shutdown_signal.h src/worker_node.cpp src/gateway.cpp \
  benchmark-aws/controller.py benchmark-aws/verify_source.py \
  tests/test_source_delivery.py tests/test_graceful_shutdown.py
git commit -m "Add bounded graceful shutdown to worker and gateway"
```

---

### Task 3: Validate inference requests and return 400 for client errors

**Files:**
- Create: `include/request_validation.h`
- Modify: `src/worker_node.cpp`
- Modify: `src/gateway.cpp`
- Modify: `benchmark-aws/controller.py`
- Modify: `benchmark-aws/verify_source.py`
- Modify: `tests/test_source_delivery.py`
- Create: `tests/test_request_validation.cpp`
- Create: `tests/test_api_validation.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `nlohmann::json`.
- Produces: `namespace inference { class ValidationError : public std::runtime_error; void validateInferRequest(const nlohmann::json&); }`.

- [ ] **Step 1: Write the failing validation unit test**

Create `tests/test_request_validation.cpp`:

```cpp
#include "request_validation.h"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

using nlohmann::json;

void require_invalid(const json& value) {
    try {
        inference::validateInferRequest(value);
    } catch (const inference::ValidationError&) {
        return;
    }
    throw std::runtime_error("expected a validation error");
}

int main() {
    inference::validateInferRequest(json{{"request_id", "r"}, {"input_data", {1, 2, 3}}});
    require_invalid(json::object());
    require_invalid(json{{"input_data", {1, 2, 3}}});
    require_invalid(json{{"request_id", 1}, {"input_data", {1, 2, 3}}});
    require_invalid(json{{"request_id", ""}, {"input_data", {1, 2, 3}}});
    require_invalid(json{{"request_id", "r"}});
    require_invalid(json{{"request_id", "r"}, {"input_data", {}}});
    require_invalid(json{{"request_id", "r"}, {"input_data", {"x"}}});
    return 0;
}
```

- [ ] **Step 2: Run it to verify it fails**

```sh
g++ -std=c++17 -O2 -Iinclude -Iexternal/json/include \
  tests/test_request_validation.cpp -o /tmp/test_request_validation
/tmp/test_request_validation
```

Expected: compile failure because `request_validation.h` does not exist.

- [ ] **Step 3: Add `include/request_validation.h`**

Validate:

- `request` is a JSON object.
- `request_id` is a string of length 1–256.
- `input_data` is an array of length 1–1,000,000.
- Every element is a finite number. Catch `nlohmann::json` conversion exceptions and rethrow `ValidationError`.

Use `ValidationError` for every rejection so both servers can map it to HTTP 400.

- [ ] **Step 4: Map validation errors to HTTP 400**

In both `src/worker_node.cpp` and `src/gateway.cpp`:

- Call `inference::validateInferRequest(request)` before routing or inference.
- Catch `inference::ValidationError` first and return `400` with `{"error": e.what()}`.
- Keep the existing `catch (const std::exception&)` path as `500`.

- [ ] **Step 5: Add the real HTTP validation test**

Create `tests/test_api_validation.py` with these assertions:

```python
def post(port, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/infer",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def get_json(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.load(response)


def test_gateway_rejects_missing_and_invalid_fields(self):
    status, body = post(self.gateway_port, {"request_id": "x"})
    self.assertEqual(status, 400)
    self.assertIn("error", body)

    status, body = post(
        self.gateway_port, {"request_id": "x", "input_data": ["bad"]}
    )
    self.assertEqual(status, 400)

    status, body = post(
        self.worker_port, {"request_id": "x", "input_data": ["bad"]}
    )
    self.assertEqual(status, 400)

    status, body = post(
        self.gateway_port, {"request_id": "ok", "input_data": [1.0, 2.0, 3.0]}
    )
    self.assertEqual(status, 200)
    self.assertIsInstance(body["output_data"], list)

    stats = get_json(self.gateway_port, "/stats")
    self.assertTrue(
        all(item["failures"] == 0 for item in stats["circuit_breakers"])
    )
```

Start the worker on an ephemeral port and the gateway with `GATEWAY_PORT` on an ephemeral port. Use the same helper pattern as `tests/test_graceful_shutdown.py`.

- [ ] **Step 6: Register and run the tests**

Add `test_request_validation` to `CMakeLists.txt` and to the C++ test target list in `.github/workflows/ci.yml` in Task 5. Run:

```sh
g++ -std=c++17 -O2 -Iinclude -Iexternal/json/include \
  tests/test_request_validation.cpp -o /tmp/test_request_validation
/tmp/test_request_validation
python3 -m unittest discover -s tests -v
```

Expected: validation unit test passes; default suite remains green.

- [ ] **Step 7: Commit**

```sh
git add include/request_validation.h src/worker_node.cpp src/gateway.cpp \
  benchmark-aws/controller.py benchmark-aws/verify_source.py \
  tests/test_source_delivery.py tests/test_request_validation.cpp \
  tests/test_api_validation.py CMakeLists.txt
git commit -m "Validate inference requests and return 400 for client errors"
```

---

### Task 4: Add a loopback gateway failover and circuit-breaker test

**Files:**
- Create: `tests/test_gateway_failover.py`

**Interfaces:**
- Consumes: two worker binaries, one gateway binary, and `GATEWAY_PORT`.
- Produces: a test that proves a request routed to a failed worker is retried on a healthy worker.

- [ ] **Step 1: Write the failover test**

Use this test flow:

```python
def test_request_retries_on_healthy_worker(self):
    worker_a_port = free_port()
    worker_b_port = free_port()
    gateway_port = free_port()
    # Start worker_a and worker_b on their ports, then the gateway with
    # ["localhost:<worker_a_port>", "localhost:<worker_b_port>"] and
    # GATEWAY_PORT=<gateway_port>.
    # Wait for both /health endpoints and the gateway /stats endpoint.

    target = None
    for index in range(20):
        payload = {
            "request_id": f"route-{index}",
            "input_data": [1000.0 + index, 0.5, 0.25],
        }
        status, body = post(gateway_port, payload)
        self.assertEqual(status, 200)
        if body["node_id"] == "worker_a":
            target = payload
            break
    self.assertIsNotNone(target, "could not find an input routed to worker_a")

    worker_a.terminate()
    worker_a.wait(timeout=5)
    status, body = post(gateway_port, target)
    self.assertEqual(status, 200)
    self.assertIsInstance(body["output_data"], list)
    self.assertEqual(body["node_id"], "worker_b")

    status, _ = post(gateway_port, target)
    self.assertEqual(status, 200)

    stats = get_json(gateway_port, "/stats")
    breaker = next(
        item for item in stats["circuit_breakers"]
        if item["node"].endswith(f":{worker_a_port}")
    )
    self.assertGreaterEqual(breaker["failures"], 1)
```

Always terminate the gateway and `worker_b` in `finally`. Use only loopback and ephemeral ports. Do not use fixed 8000/8001.

- [ ] **Step 2: Run it to verify current behavior**

```sh
BENCH_WORKER_BIN=/path/to/worker_node \
BENCH_GATEWAY_BIN=/path/to/gateway \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_gateway_failover -v
```

Expected: PASS if the current retry path is correct. If it fails or hangs, record the exact failure and fix `src/gateway.cpp` before the next step. Do not weaken the test.

- [ ] **Step 3: Commit**

```sh
git add tests/test_gateway_failover.py
git commit -m "Cover gateway failover and circuit-breaker failure accounting"
```

---

### Task 5: Build the real C++ targets and run real-model tests in CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: pinned ORT 1.23.2 tarball, the tracked model, and the new tests.
- Produces: a CI job that builds `worker_node`, `gateway`, `test_batch_timings`, `test_batch_shapes`, and `test_request_validation`, then runs `ctest` and the opt-in Python tests.

- [ ] **Step 1: Update the workflow**

Add a `cpp` job (or extend `checks` with a separate job) that:

- Checks out with `submodules: true`.
- Downloads the pinned ORT tarball from `remote.sh` and verifies SHA-256 `1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b`.
- Configures CMake with `-DCMAKE_BUILD_TYPE=Release` and the ORT include/lib directories.
- Builds with `--parallel 2`.
- Runs:

```sh
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx ctest --test-dir build --output-on-failure
BENCH_WORKER_BIN=$PWD/build/worker_node \
BENCH_GATEWAY_BIN=$PWD/build/gateway \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_instrumented_worker tests.test_gateway_latency \
  tests.test_graceful_shutdown tests.test_api_validation \
  tests.test_gateway_failover -v
```

Set `timeout-minutes: 20`. Keep the existing Python and lint job.

- [ ] **Step 2: Run the same commands locally**

Use the local pinned ORT and dependency paths, then run the full suite and `ctest`. Expected: all tests pass locally.

- [ ] **Step 3: Commit, then verify the remote workflow after the owner authorizes the push**

```sh
git add .github/workflows/ci.yml
git commit -m "Build real worker and gateway in CI"
# After owner authorization:
git push origin main
gh run list --workflow ci.yml --limit 3
```

Expected: the new run appears and completes successfully. If the workflow fails, read the job log, fix the workflow, and rerun. Do not claim CI passed until the run is green.

---

### Task 6: Document the hardened behavior and update the resume

**Files:**
- Create: `docs/production-hardening.md`
- Modify: `README.md`
- Modify: `docs/investigation/EXPERIMENTS.md`
- Modify: `docs/investigation/FINDINGS.md`
- Modify: `/home/althaaf/JOB_SEARCH/MAIN_DOCUMENTS/GENERIC/resume_software_engineer.tex` only if the verified work adds a useful bullet.

**Interfaces:**
- Consumes: verified test and CI results.
- Produces: documented behavior, status codes, shutdown semantics, and verified resume evidence.

- [ ] **Step 1: Write `docs/production-hardening.md`**

Include:

- The shutdown contract: SIGINT/SIGTERM stops new connections, waits for in-flight handlers, fails queued batch work, and exits within 5 seconds in the tested loopback case.
- The validation contract: 400 for malformed/missing fields, 500 for internal errors; valid requests unchanged.
- The failover behavior: a failed routed worker is retried on a healthy worker; the failed worker's circuit-breaker failures are recorded.
- The CI coverage and exact commands.
- Limits: loopback only, single host, no production deployment, no multi-host claim.

- [ ] **Step 2: Update README and the experiment ledger**

- Add a short "Production hardening" section to `README.md` with links to the new report.
- Add a "Cycle 6 — production hardening" entry to `EXPERIMENTS.md` with files, tests, results, and cost `$0` (local/CI only).
- Add the reproduced reliability findings to `FINDINGS.md`.

- [ ] **Step 3: Update the resume only with verified work**

If the hardening work passes and the page still fits, extend the existing test/engineering bullet with one concise clause, for example:

> “Hardened the worker/gateway with bounded graceful shutdown, 400-level request validation, loopback failover tests, and a pinned-ORT CI build.”

Compile in `/tmp`, check one page, links, extracted text, and the bullet-tail rule, then regenerate `resume_software_engineer.pdf` and the IPhO-appended copy. Do not add a new bullet if it would force a second page.

- [ ] **Step 4: Run the final checks**

```sh
python3 -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
python3 docs/investigation/audit.py
git diff --check
git status --short --branch
```

Expected: all tests pass, audit reports the investigation and resume artifacts, the worktree is clean after the final commit, and `origin/main` matches.

- [ ] **Step 5: Commit and push after review**

```sh
git add docs/production-hardening.md README.md \
  docs/investigation/EXPERIMENTS.md docs/investigation/FINDINGS.md
git commit -m "Document production hardening and verification"
git push origin main
```

Include the exact commit hashes and CI run URL in the final handoff.

---

## Self-review

- **Spec coverage:** The plan covers the four documented gaps: shutdown, validation/status codes, failover coverage, and CI. The measurement-quality and multi-host items remain explicitly out of scope.
- **Step scan:** Each task has a failing test, an implementation step, a verification command, and a commit. No task says “handle edge cases” without a concrete input.
- **Type consistency:** `block_shutdown_signals`, `wait_for_shutdown_signal`, `ValidationError`, and `validateInferRequest` are defined once and used consistently.
- **Review focus:** Each of the five review items has a test: shutdown queue test, API validation test, failover test, ephemeral-port/CI setup, and existing response-compatibility checks.
- **Proportion:** The plan adds two small headers, two server shutdown paths, one validation path, four new test modules, and CI. It does not redesign the service.
