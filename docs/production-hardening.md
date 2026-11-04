# Production hardening

## Scope

This phase closes the reliability gaps identified after the image benchmark. It does not add TLS, authentication, rate limiting, deployment, or a public service. It keeps the existing gateway, cache, batching, and ONNX Runtime backend.

## Contracts

### Shutdown

- The worker and gateway block `SIGINT` and `SIGTERM` before starting worker or ORT threads.
- The HTTP listener runs on a joined thread.
- On a shutdown signal, the server stops accepting requests, the listener joins, and the worker destroys its batch processor.
- `BatchProcessor::stop()` is idempotent. It rejects new work and fails every promise still in the queue when stop begins with `std::runtime_error("batch processor stopped")`. A request that was already dequeued may complete. A handler cannot wait forever.
- Idle keep-alive connections use a 2-second timeout. Manual idle-connection checks exited in about 2 seconds, and the failover test kills a worker while the gateway holds a connection to it.
- Valid in-flight inference behavior is unchanged.

### Request validation

`POST /infer` requires:

- A JSON object.
- `request_id` as a string of 1–256 characters.
- `input_data` as an array of 1–1,000,000 finite numbers.

The gateway validates before routing. The worker validates before cache or inference. Invalid client input returns HTTP 400 with an `error` field. Internal failures still return HTTP 500. A rejected request does not increment a worker circuit breaker because it never reaches a worker.

The binary `/infer_tensor` path keeps its existing body and request-ID validation.

### Failover

The loopback test starts two workers and one gateway. It finds an input that hashes to `worker_a`, terminates that worker, and sends the same input again. The gateway retries `worker_b`, returns a valid response, and records a circuit-breaker failure for `worker_a`.

This is retry evidence, not a high-availability claim. There is no spare worker, health-check-driven replacement, or multi-AZ test.

## CI

The `cpp` job:

- Checks out the pinned submodules.
- Downloads ONNX Runtime 1.23.2 and verifies SHA-256 `1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b`.
- Configures and builds `worker_node`, `gateway`, and the C++ tests.
- Runs `ctest` with the tracked ResNet50 model.
- Runs the real worker/gateway loopback tests.

## Verification

Commands run on 2026-11-03 from the repository root:

```sh
BENCH_WORKER_BIN=/tmp/opencode/cmake-build-hardening/worker_node \
BENCH_GATEWAY_BIN=/tmp/opencode/cmake-build-hardening/gateway \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
python3 -m unittest discover -s tests -v
# 70 passed, 7 intended skips

BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
ctest --test-dir /tmp/opencode/cmake-build-hardening --output-on-failure
# 4/4 passed: batch_timings, batch_shapes, tensor_payload, request_validation

BENCH_WORKER_BIN=/tmp/opencode/cmake-build-hardening/worker_node \
BENCH_GATEWAY_BIN=/tmp/opencode/cmake-build-hardening/gateway \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_instrumented_worker tests.test_gateway_latency \
  tests.test_graceful_shutdown tests.test_api_validation tests.test_gateway_failover -v
# 11 passed
```

The C++ build in the local verification used the pinned local ORT and dependency paths because the checkout's gitlinks were empty. CI uses the pinned submodules and the same ORT tarball.

## Limits

- The shutdown and failover tests use loopback and ephemeral ports, not a deployed network.
- The shutdown bound is tested for an idle worker/gateway and one open gateway-to-worker keep-alive connection. A long in-flight inference can extend shutdown up to its inference time; there is no forced cancellation.
- Validation covers the JSON `/infer` contract. The binary endpoint has separate validation.
- The failover test uses one injected worker loss, not repeated failures, network partitions, or recovery of the killed worker.
- The CI job passed on [PR #2](https://github.com/althaafsn/distributed-inference-engine-cpp/pull/2) after the branch was pushed. It validates the pushed branch, not a deployment.
- No production service, credentials, or production data were touched.
- No TLS, authentication, rate limiting, metrics export, model versioning, or multi-AZ work is included.

## Rollback

The hardening changes are source-only. Revert the shutdown, validation, and CI commits if a deployment cannot accept the new status codes or the 2-second idle keep-alive timeout. The binary endpoint and valid JSON response shapes are unchanged.
