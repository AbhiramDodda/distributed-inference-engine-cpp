# Image Auto-Tuning Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reusable image-classification benchmark and auto-tuning harness on fixed AWS hardware, then produce a ranked capacity report and production-readiness review.

**Architecture:** Add a model registry, deterministic ImageNet subset, raw float32 transport, async load generator, and a multi-host controller. The harness provisions once, runs model/architecture/settings candidates under one SLO, finds the saturation knee, tests failures and consistency, and emits raw and sanitized evidence.

**Tech Stack:** C++17, cpp-httplib 0.14.3, nlohmann/json 3.11.3, ONNX Runtime 1.23.2, Python 3.12, aiohttp, Pillow, NumPy, AWS EC2.

**Spec:** `docs/superpowers/specs/2026-11-02-image-autotune-benchmark-design.md`

## Prerequisite

The production-hardening plan supplies graceful shutdown, validation, failover tests, and CI:

`docs/superpowers/plans/2026-11-02-production-hardening.md`

If it is incomplete, the benchmark may still run in an isolated private network, but the report must mark shutdown, validation, and failover as blockers.

## Global Constraints

- Budget ceiling: **$10** including setup, reruns, storage, IPv4, and transfer.
- Fixed worker/gateway type: `c6i.large`; load generator: `c6i.xlarge`.
- Maximum 8 workers; test 1, 2, 4, and 8 where budget and time permit.
- Models and data must be pinned by SHA-256. Do not commit model binaries or ImageNet images.
- App ports are private only. SSH only from the local public IPv4 `/32`.
- SLO: 99.9% availability, p95 ≤ 500 ms, p99 ≤ 1 s, errors ≤ 0.1%, CPU ≤ 80%, 5-minute hold.
- Every paid run needs preflight, a dry-run cost estimate, verified cleanup, and a cost-ledger update.
- Commit after each task. Push only after the full suite and owner review, unless immediate push is authorized.

## Review Focus

1. **Fairness:** same data, preprocessing, warm-up, cache reset, host type, and client for every candidate.
2. **Client saturation:** a separate load generator; report client CPU and label client-bound results.
3. **Accuracy:** top-1/top-5 must be credible; a fast model with wrong accuracy is disqualified.
4. **Capacity:** report SLO capacity, knee, collapse point, and absolute throughput separately.
5. **Security and cleanup:** no public app ports; all tagged instances, volumes, and groups removed.

---

### Task 1: Model registry and real ImageNet fixture

**Files:**
- Create: `benchmark-aws/autotune/models.json`
- Create: `benchmark-aws/autotune/prepare_data.py`
- Create: `benchmark-aws/requirements-benchmark.txt`
- Create: `tests/test_autotune_data.py`
- Modify: `.gitignore`

**Interfaces:**
- `models.json` entries: `id`, `url`, `sha256`, `top1`, `top5`, `input_shape`, `preprocess`.
- `prepare_data.py --output benchmark-aws/data/imagenet500 --count 500`.
- Writes `manifest.json` with dataset commit, model metadata, per-image label, image SHA-256, and tensor path.

- [ ] **Step 1: Write the failing data test**

Test deterministic image selection, label mapping, tensor size `3*224*224*4`, and rejection of a short/oversized tensor.

- [ ] **Step 2: Implement `prepare_data.py`**

Use the pinned dataset commit `c55ca047bd854ba01d3af6c283c034924a644db2`. Select one image per class folder 0–499. Download, checksum, preprocess with Pillow/NumPy, write little-endian float32 tensors and a manifest. Refuse to overwrite without `--force`.

- [ ] **Step 3: Download and checksum the five models**

Use the ONNX Model Zoo URLs. Verify size and SHA-256. Store under ignored `benchmark-aws/models/`. Keep `models.json` with the hashes. Do not commit binaries.

- [ ] **Step 4: Create the pinned Python environment**

Create `/tmp/opencode/bench-venv`, install `Pillow`, `numpy`, `aiohttp`, and pin the resolved versions in `requirements-benchmark.txt`.

- [ ] **Step 5: Run the fixture**

```sh
/tmp/opencode/bench-venv/bin/python benchmark-aws/autotune/prepare_data.py \
  --output /tmp/opencode/imagenet5 --count 5
/tmp/opencode/bench-venv/bin/python benchmark-aws/autotune/prepare_data.py \
  --output benchmark-aws/data/imagenet500 --count 500
```

Expected: 500 tensors of 602112 bytes; manifest complete.

- [ ] **Step 6: Commit**

```sh
git add benchmark-aws/autotune/models.json benchmark-aws/autotune/prepare_data.py \
  benchmark-aws/requirements-benchmark.txt tests/test_autotune_data.py .gitignore
git commit -m "Add model registry and ImageNet fixture"
```

---

### Task 2: Binary tensor transport and runtime settings

**Files:**
- Create: `include/tensor_payload.h`
- Modify: `src/worker_node.cpp`
- Modify: `src/gateway.cpp`
- Modify: `benchmark-aws/controller.py`
- Modify: `benchmark-aws/verify_source.py`
- Modify: `tests/test_source_delivery.py`
- Create: `tests/test_tensor_payload.cpp`
- Create: `tests/test_binary_endpoint.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- `decode_float32_le`, `encode_float32_le`.
- `POST /infer_tensor` on worker and gateway.
- Worker env settings: `MAX_BATCH_SIZE`, `CACHE_CAPACITY`, `ORT_INTRA_THREADS`, `BATCH_DEADLINE_MS`.

- [ ] **Step 1: Write the failing codec and endpoint tests**

Codec round-trip; binary vs JSON output equality; cache hit; malformed length; missing request ID.

- [ ] **Step 2: Implement the codec and endpoints**

Little-endian float32. Validate body length (positive multiple of 4, ≤ 16 MB) and request ID (1–256 chars). Reuse the existing cache/batch path. Return raw float32 with timing and cache headers. Gateway routes by FNV-1a over the raw body and forwards the body/headers.

- [ ] **Step 3: Add settings envs**

Replace hardcoded cache capacity 1000 and max batch 32 with validated env overrides. Log the effective values. Keep defaults 1000 and 32.

- [ ] **Step 4: Add the header to the AWS delivery allowlist**

Update `controller.py`, `verify_source.py`, and `tests/test_source_delivery.py`.

- [ ] **Step 5: Run tests**

```sh
g++ -std=c++17 -O2 -Iinclude tests/test_tensor_payload.cpp -o /tmp/test_tensor_payload
/tmp/test_tensor_payload
BENCH_WORKER_BIN=/tmp/worker_node BENCH_GATEWAY_BIN=/tmp/gateway \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_binary_endpoint -v
python3 -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```sh
git add include/tensor_payload.h src/worker_node.cpp src/gateway.cpp \
  benchmark-aws/controller.py benchmark-aws/verify_source.py \
  tests/test_source_delivery.py tests/test_tensor_payload.cpp \
  tests/test_binary_endpoint.py CMakeLists.txt
git commit -m "Add binary tensor transport and runtime settings"
```

---

### Task 3: Async load generator, accuracy, and local baseline

**Files:**
- Create: `benchmark-aws/autotune/loadgen.py`
- Create: `benchmark-aws/autotune/workloads.py`
- Create: `benchmark-aws/autotune/analyze.py`
- Create: `tests/test_autotune_load.py`
- Create: `benchmark-aws/run_autotune.py`
- Create: `benchmark-aws/autotune/matrix/default.json`

**Interfaces:**
- `loadgen.py` records JSONL: latency, status, request ID, node ID, cached, queue/backend timing, batch size, error.
- `--accuracy` stores output vectors and computes top-1/top-5.
- `--transport binary|json`; `--mode unique|repeated|mixed`; `--concurrency` or `--rate`.
- `analyze.py` computes SLO capacity, percentiles, errors, cache rate, per-worker distribution, and cost per 1M.

- [ ] **Step 1: Write the failing load-generator tests**

Test deterministic image selection, mixed ratio, percentile edge cases, error accounting, and accuracy calculation.

- [ ] **Step 2: Implement the async client**

Use `aiohttp` with connection pooling. Precompute payloads. Separate accuracy and load modes. Record client CPU and event-loop lag. Refuse non-loopback targets unless `--allow-remote`.

- [ ] **Step 3: Implement the local runner**

`run_autotune.py` starts worker/gateway on ephemeral ports, resets cache between candidates, runs warm-up, accuracy, unique/repeated/mixed, and saves a run manifest.

- [ ] **Step 4: Run a local real-data smoke test**

Use ResNet50 and 20 images first, then 500 images. Verify accuracy is plausible and binary/JSON outputs match.

- [ ] **Step 5: Commit**

```sh
git add benchmark-aws/autotune/loadgen.py benchmark-aws/autotune/workloads.py \
  benchmark-aws/autotune/analyze.py tests/test_autotune_load.py \
  benchmark-aws/run_autotune.py benchmark-aws/autotune/matrix/default.json
git commit -m "Add async real-data load generator and local runner"
```

---

### Task 4: Multi-host controller and private network

**Files:**
- Create: `benchmark-aws/multihost_controller.py`
- Create: `benchmark-aws/multihost/remote_prepare.sh`
- Create: `benchmark-aws/multihost/remote_worker.sh`
- Create: `benchmark-aws/multihost/remote_gateway.sh`
- Create: `benchmark-aws/multihost/remote_load.sh`
- Create: `tests/test_multihost_setup.py`

**Interfaces:**
- `preflight`, `run --state`, `cleanup --state`.
- Topology: gateway + N workers + load generator; N configurable 1–8.
- Security groups: SSH local `/32`; gateway 8000 from load group; worker ports from gateway group.
- State records all instance, volume, group, and tag IDs.

- [ ] **Step 1: Write unit tests for SG rules and cleanup ownership**

Assert no `0.0.0.0/0` app rule; cleanup refuses a wrong account or wrong tag.

- [ ] **Step 2: Implement provisioning and deployment**

Reuse the single-host controller's AMI, ephemeral key, host-key verification, watchdog, and cleanup patterns. Copy the source patch, models, and scripts. Build in parallel. Start workers, gateway, and load generator.

- [ ] **Step 3: Implement collection and cleanup**

Collect per-host `vmstat`, `pidstat`, `mpstat`, `free`, `lscpu`, service logs, and results. Verify termination, volumes, and security groups.

- [ ] **Step 4: Run a tiny topology smoke test**

Use 1 worker and 10 requests. Verify public probes to 8000/8001 fail and cleanup is complete.

- [ ] **Step 5: Commit**

```sh
git add benchmark-aws/multihost_controller.py benchmark-aws/multihost \
  tests/test_multihost_setup.py
git commit -m "Add isolated multi-host benchmark controller"
```

---

### Task 5: Run the campaign and analyze

**Files:**
- Create: `benchmark-aws/autotune/matrix/campaign.json`
- Create: `docs/autotune-benchmark.md`
- Create: `benchmarks/results/2026-11-02-autotune-*.json`

- [ ] **Step 1: Dry-run the matrix**

Print candidate order, request counts, estimated runtime, and estimated cost. Stop if the estimate exceeds the remaining $10 budget.

- [ ] **Step 2: Run Stage 1 — model screening**

Five models, baseline architecture, default settings. Record accuracy and a saturation preview. Keep the top models.

- [ ] **Step 3: Run Stage 2 — architecture screening**

Baseline, binary transport, process-pool, and direct-worker diagnostic. Keep the best modes.

- [ ] **Step 4: Run Stage 3 — settings**

Coordinate search over threads, batch deadline, batch size, and cache capacity. Repeat finalists.

- [ ] **Step 5: Run Stage 4 — capacity and failure**

Sweep 1/2/4/8 workers and client counts. Find SLO capacity, knee, collapse point. Kill a worker and verify retries/consistency.

- [ ] **Step 6: Analyze and report**

Produce ranked results, Pareto frontier, saturation curves, errors, cost per 1M, and production blockers. Commit.

---

### Task 6: Documentation, resume, and final handoff

**Files:**
- Create: `docs/production-readiness.md`
- Modify: `README.md`
- Modify: `docs/investigation/EXPERIMENTS.md`
- Modify: `docs/investigation/FINDINGS.md`
- Modify: `resume_software_engineer.tex` only with verified results.

- [ ] **Step 1: Write the production-readiness review**

Mark auth, TLS, limits, deadlines, shutdown, validation, failover, observability, container quota, model versioning, and licensing PASS/BLOCKED/OUT OF SCOPE.

- [ ] **Step 2: Update the README and experiment ledger**

Add the harness, model set, SLO, results, cost, cleanup, and limits.

- [ ] **Step 3: Update the resume**

Use only measured results. Keep one page and the IPhO-appended copy. Run the bullet-tail, link, text, and page checks.

- [ ] **Step 4: Final checks**

```sh
python3 -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
python3 docs/investigation/audit.py
git diff --check
git status --short --branch
```

Verify AWS cleanup and update the cost ledger.

- [ ] **Step 5: Commit and push after review**

Commit the report and resume evidence. Push after owner authorization. Include commit hashes, CI URL, cost, cleanup, and remaining blockers.

---

## Self-review

- **Spec coverage:** model diversity, real accuracy, SLO capacity, architecture/settings search, multi-node, failure/consistency, cost, and production review are all covered.
- **Step scan:** each task has a test or smoke check, an implementation step, a verification command, and a commit.
- **Type consistency:** `models.json`, `manifest.json`, `/infer_tensor`, `loadgen.py`, and the controller state are defined once and reused.
- **Review focus:** fairness, client saturation, accuracy, capacity definition, and cleanup each have explicit checks.
- **Proportion:** the plan adds a harness, one transport codec, one controller, and reports. It does not redesign the inference engine.
