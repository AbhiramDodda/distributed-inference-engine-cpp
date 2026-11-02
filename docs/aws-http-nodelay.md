# TCP_NODELAY for keep-alive inference responses

## Problem and cause

The gateway and worker use cpp-httplib with keep-alive connections. cpp-httplib defaults `CPPHTTPLIB_TCP_NODELAY` to false. When a small response is written as headers followed by a body, Nagle can hold the second write until the peer acknowledges the first, and the peer can delay that acknowledgment. The result is a fixed stall on affected requests.

Local isolation showed the failure mode clearly:

- Direct worker cache-hit p50: 0.4–0.7 ms.
- Through the unmodified gateway: p50 about 83 ms, with values clustered near 1, 42, and 84 ms.
- Enabling `TCP_NODELAY` on the worker server, the gateway server, and the gateway's worker client reduced cache-hit p50 to 0.8 ms.

There are three affected socket paths, so a partial change leaves at least one hop stalled.

## Change

- `src/worker_node.cpp`: enable `TCP_NODELAY` on the worker HTTP server.
- `src/gateway.cpp`: enable it on the gateway HTTP server and on each pooled worker client.
- `src/gateway.cpp`: allow `GATEWAY_PORT` to override the default port 8000 for loopback integration tests; the default is unchanged.
- The AWS source-delivery allowlist now includes `src/gateway.cpp`; the controller rejects an instrumented run missing any of the worker/gateway nodelay settings.
- A new opt-in integration test starts a worker and gateway on loopback, sends sequential repeated inputs, verifies request association and cache-hit output equality, and checks a generous latency ceiling that catches the ~80 ms delayed-ACK regression. It is a regression guard, not a performance claim.

## Benchmark method

- Temporary on-demand `c6i.large`, `us-east-1`; gateway, one CPU worker, and the closed-loop client on the same instance.
- Pinned base `c4d8a08b72df810ffad0dce6a993cecae22494bb`, model SHA-256 `79102261eb6e5fd7af5d27f41316293e388c5cb691e5d25bfb035c4f64fefe31`, ORT 1.23.2.
- Baseline: the accepted thread-tuned source (2 ORT intra-op threads, patch `9b4c0303…`). Runs `infer-1f39a2ac8519f743`, `infer-0601374edef6f33d`.
- Changed: the same source plus `TCP_NODELAY` (patch `1c3bf218…`). Runs `infer-ff9ac124eb8e79ea`, `infer-2063d463870454ba`.
- Each run: 60 repeated and 60 unique requests at concurrency 1, 2, and 4; 360 requests total. Two runs per variant. The baseline and changed runs are from separate sessions on the same instance type and day; the local matched probe used the same gateway and CPU budget.
- The controller verified termination and the absence of the recorded volume and security group after each run.

## Local matched probe

Before the cloud runs, the same gateway and worker were built from the baseline and changed source, pinned to two logical CPUs with two ORT threads, and driven with 20 requests per level. The local probe used the same gateway port and workload for both variants.

| Workload | Baseline p50 | Changed p50 |
| --- | ---: | ---: |
| Cache hit, c1 | 83.2 ms | 1.6 ms |
| Cache hit, c2 | 84.0 ms | 2.1 ms |
| Cache hit, c4 | 83.9 ms | 4.3 ms |
| Unique, c1 | 149.1 ms | 108.0 ms |
| Unique, c2 | 235.2 ms | 193.6 ms |
| Unique, c4 | 408.4 ms | 365.4 ms |

All 240 local requests succeeded. Backend p50 was unchanged. Raw summaries are in `benchmark-aws/runs/local-cycle2-2026-11-02/`.

## AWS results

Repeated inputs are cache hits and do not enter ONNX Runtime. Values are means of two runs, 60 observations per level.

| Concurrency | Baseline p50 | Changed p50 | p50 change | Baseline req/s | Changed req/s | Throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 98.26 ms | 1.16 ms | −98.8% | 13.81 | 311.49 | +2,156% |
| 2 | 98.13 ms | 2.15 ms | −97.8% | 27.72 | 894.77 | +3,128% |
| 4 | 97.94 ms | 3.54 ms | −96.4% | 55.26 | 955.64 | +1,629% |

Unique inputs exercise cache miss, batching, and the backend:

| Concurrency | Baseline p50 | Changed p50 | p50 change | Baseline req/s | Changed req/s | Throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 149.85 ms | 108.02 ms | −27.9% | 7.44 | 9.22 | +24.0% |
| 2 | 231.69 ms | 190.42 ms | −17.8% | 9.11 | 10.48 | +15.1% |
| 4 | 403.91 ms | 360.00 ms | −10.9% | 10.26 | 11.10 | +8.1% |

Whole-batch backend p50 was effectively unchanged: 86.17/168.92/338.46 ms baseline versus 85.76/167.10/334.25 ms changed at c1/c2/c4 (−0.5/−1.1/−1.2%). That is the expected negative control: `TCP_NODELAY` changes transport timing, not inference compute.

All 360 requests in each run succeeded. The changed worker still logged `ORT intra-op threads: 2 (hardware_concurrency=2)`.

The high cache-hit throughput values are system throughput from a closed-loop Python client that also runs on the two-vCPU host. At concurrency 2 and 4 the client is the likely ceiling; these numbers are not a server maximum or an open-loop capacity result.

## Correctness and regression checks

- `python3 -m unittest discover -s tests -v`: 33 passed, 3 intended skips (the opt-in integration classes).
- All five opt-in real-worker/gateway tests passed with the changed binaries:
  - cache miss/hit timing semantics and output equality;
  - benchmark client records real worker timings;
  - default thread count and 2-vs-4-thread output parity within `1e-4`;
  - gateway cache-hit latency regression guard.
- The gateway test asserts matching request IDs, `cached=false` on the first request, `cached=true` afterwards, and exact equality between the miss and hit output vectors.
- The source-delivery tests verify that `src/gateway.cpp` is delivered and that the patch contains the worker/gateway nodelay settings and the test port override.

## Interpretation and limits

The data support the hypothesis that the ~98 ms cache-hit latency was dominated by keep-alive delayed-ACK stalls, not cache lookup or JSON work. Removing those stalls cuts cache-hit p50 by 96–99% and raises closed-loop system throughput. Unique requests also improve because the same response path is used, but the gain is smaller because backend compute dominates.

This is a single-host, synthetic-input, closed-loop experiment. The client shares the CPU; each level has 60 requests per run; p99 is not reported as a stable estimate. The baseline and changed AWS runs are from separate sessions, though the local matched probe and the unchanged backend timings reduce that concern. The result does not establish production readiness, model accuracy, multi-host scaling, or a maximum server capacity. More, smaller TCP segments are a normal tradeoff for latency; no adverse correctness effect was observed.

## Reproduction

```sh
python3 -m unittest discover -s tests -v
BENCH_WORKER_BIN=build/worker_node \
BENCH_GATEWAY_BIN=build/gateway \
BENCH_MODEL=models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_instrumented_worker tests.test_gateway_latency -v
python3 benchmark-aws/controller.py preflight
python3 benchmark-aws/controller.py run --source instrumented
```

Sanitized aggregate: [`benchmarks/results/2026-11-02-aws-http-nodelay.json`](../benchmarks/results/2026-11-02-aws-http-nodelay.json). Raw logs and request records remain local under `benchmark-aws/runs/`. Cost estimates are in the [AWS cost ledger](aws-cost-ledger.md); the nine-run project estimate is $0.031510 of the $5 target before tax, transfer, or billing adjustments.

## Attribution

The gateway, cache, batching, and ONNX Runtime backend are upstream work by AbhiramDodda and prior fork changes. The thread-tuning baseline and this HTTP change are the owner's investigation contributions. AI assistance was used for the investigation and implementation; the owner remains responsible for review and publication.
