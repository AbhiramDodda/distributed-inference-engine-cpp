# Gateway response passthrough

## Problem

The gateway parsed every successful worker response into a `nlohmann::json` object and immediately dumped it back to the client. The gateway does not transform the response. For a ResNet output of about 10 KB, that parse and re-serialization consumes CPU on the same two-vCPU host as the worker. Cache-hit requests are short enough that this cost is visible: a local A/B measured about 30–37% higher cache-hit throughput when the gateway forwarded the worker body instead.

## Change

- `src/gateway.cpp`: `tryNode` and `routeRequest` now return the raw response body string for a successful worker call.
- The gateway handler sets that body directly with `application/json` instead of `response.dump()`.
- Non-200 responses, connection errors, retries, circuit breakers, routing, cache behavior, and the client-visible JSON shape are unchanged for valid worker responses.
- The gateway no longer validates the worker's JSON before forwarding it. The worker is a trusted local component; the benchmark client still checks response shape and request association. A malformed 200 body from the worker would now reach the client instead of being converted to a gateway error.

## Local A/B

Same worker, model, gateway port, CPU budget, and workload; fresh worker per variant. Cache hits are 20 repeated requests per level, unique inputs are 20 cache-miss requests per level.

| Workload | Baseline p50 | Changed p50 | Baseline req/s | Changed req/s |
| --- | ---: | ---: | ---: | ---: |
| Cache hit, c1 | 1.57 ms | 1.11 ms | 141.3 | 147.4 |
| Cache hit, c2 | 2.09 ms | 1.50 ms | 867.7 | 1,211.1 |
| Cache hit, c4 | 4.18 ms | 2.70 ms | 895.9 | 1,228.9 |
| Unique, c1 | 97.5 ms | 97.4 ms | 10.20 | 10.22 |
| Unique, c2 | 184.5 ms | 182.1 ms | 10.78 | 10.89 |
| Unique, c4 | 358.9 ms | 359.0 ms | 11.08 | 11.10 |

All 240 local requests succeeded. Raw summaries are in `benchmark-aws/runs/local-cycle5-final-2026-11-02/`.

## AWS A/B

Baseline: accepted thread-tuned + `TCP_NODELAY` + 10 ms deadline source with gateway JSON parse/dump; runs `infer-f6dfdf8dd788978c`, `infer-d1070c813d774208`.
Changed: same source with response passthrough; runs `infer-a5f5bc6de001b9ae`, `infer-f8f48cbffd6ffd98`.

Each run has 60 repeated and 60 unique requests at concurrency 1/2/4. Values are means of two runs.

| Concurrency | Cache-hit p50 baseline | Cache-hit p50 changed | p50 change | Baseline req/s | Changed req/s | Throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.15 ms | 0.85 ms | −25.9% | 327.7 | 367.9 | +12.3% |
| 2 | 2.17 ms | 1.51 ms | −30.4% | 884.7 | 1,246.8 | +40.9% |
| 4 | 3.69 ms | 2.80 ms | −24.1% | 953.1 | 1,264.4 | +32.7% |

Unique requests were effectively unchanged: p50 98.5/181.6/351.5 ms → 97.8/181.1/350.9 ms, throughput 10.09/10.93/11.36 → 10.14/11.02/11.38 req/s. Backend time and queue wait were unchanged. All 360 requests per run succeeded. The high cache-hit throughput remains client-bound at the upper levels; it is not a server maximum.

## Correctness and regression checks

- The gateway integration test warms an input through the gateway, sends the same input directly to the worker and through the gateway, and asserts equal `output_data` and `node_id` with `cached=true`. This verifies that the forwarded body preserves the worker output.
- `python3 -m unittest discover -s tests -v`: 33 passed, 3 intended skips.
- All six opt-in worker/gateway tests pass with the current binaries.
- `ctest` passes `batch_timings` and `batch_shapes`.
- Source-delivery tests still verify that `src/gateway.cpp` is delivered and that the nodelay settings are present.

## Interpretation and limits

The gateway was spending measurable CPU on a parse/dump round trip that did not change the response. Removing it cuts cache-hit p50 by 24–30% and raises closed-loop cache-hit throughput by 12–41%, while leaving the cache-miss/backend path unchanged. This is consistent with the response size and the shared two-vCPU budget.

The tradeoff is that the gateway now trusts the worker's 200 response body rather than re-validating it. That is reasonable for a trusted local worker, but it is a reliability decision, not only a performance one. The result is single-host, synthetic, closed-loop, with 60 requests per level; baseline and changed AWS runs are from separate sessions. Do not claim production readiness, a server maximum, model accuracy, or multi-host scaling.

## Reproduction

```sh
python3 -m unittest discover -s tests -v
BENCH_WORKER_BIN=build/worker_node BENCH_GATEWAY_BIN=build/gateway \
BENCH_MODEL=models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_instrumented_worker tests.test_gateway_latency -v
python3 benchmark-aws/controller.py preflight
python3 benchmark-aws/controller.py run --source instrumented
```

Sanitized aggregate: [`benchmarks/results/2026-11-02-aws-gateway-passthrough.json`](../benchmarks/results/2026-11-02-aws-gateway-passthrough.json). Raw logs remain local under `benchmark-aws/runs/`. Cost estimates are in the [AWS cost ledger](aws-cost-ledger.md); the thirteen-run project estimate is $0.044703 of the $5 target before tax, transfer, or billing adjustments.

## Attribution

The gateway, cache, batching, and ONNX Runtime backend are upstream work by AbhiramDodda and prior fork changes. The thread tuning, HTTP nodelay, input-alignment fix, batch-deadline tuning, and response passthrough are the owner's investigation contributions. AI assistance was used for the investigation and implementation; the owner remains responsible for review and publication.
