# Batch deadline tuning — 20 ms to 10 ms

## Problem

The batch processor waited up to 20 ms before flushing a partial batch. At concurrency 1–4, the requests usually arrive together, so the full 20 ms was often pure queueing delay. The instrumented baseline showed a queue-wait p50 near 20 ms at every level, while the measured 2- and 4-request batches already fit inside a shorter window.

## Change

- `src/worker_node.cpp`: default batch deadline is now 10 ms.
- `BATCH_DEADLINE_MS` overrides the default for experiments and operations.
- The startup banner prints the effective deadline, for example `Batch Timeout:     10ms`.
- The opt-in real-worker test checks both the default and the override.

The batch size cap (32), the adaptive oldest-request deadline semantics, and the cache/HTTP paths are unchanged.

## Local sweep

Direct worker requests, unique inputs, 30 requests per level, two logical CPUs, two ORT threads.

| Deadline | c1 p50 | c2 p50 | c4 p50 | c1 req/s | c2 req/s | c4 req/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ms | 107.0 ms | 193.2 ms | 363.5 ms | 9.30 | 10.33 | 10.91 |
| 10 ms | 97.3 ms | 183.3 ms | 356.0 ms | 10.23 | 10.86 | 11.14 |
| 5 ms | 92.6 ms | 179.0 ms | 349.3 ms | 10.75 | 11.14 | 11.37 |
| 2 ms | 90.4 ms | 176.1 ms | 350.4 ms | 11.00 | 11.31 | 11.27 |

At c1/c2/c4, 10 ms kept the same observed batch sizes (1, 2, and 2/4) and reduced queue wait from about 20 ms to about 10 ms. The 5 ms setting was faster at low concurrency but lost batch size at high concurrency: at c32 its mean batch size fell from 20 to 10.1, while 10 ms still produced batch size 20. The 2 ms setting produced a worse c4 tail (p95 539.8 ms) because batches sometimes did not fill.

The high-concurrency local sweep is in `benchmark-aws/runs/local-cycle4-high-2026-11-02/` and `local-cycle4-high10-2026-11-02/`.

## AWS A/B

Baseline: accepted thread-tuned + `TCP_NODELAY` source with a 20 ms deadline; runs `infer-ff9ac124eb8e79ea`, `infer-2063d463870454ba`.
Changed: same source with the default 10 ms deadline; runs `infer-f6dfdf8dd788978c`, `infer-d1070c813d774208`.

Each run has 60 repeated and 60 unique requests at concurrency 1/2/4. Values are means of two runs.

| Concurrency | Unique p50 20 ms | Unique p50 10 ms | p50 change | 20 ms req/s | 10 ms req/s | Throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 108.0 ms | 98.5 ms | −8.8% | 9.22 | 10.09 | +9.4% |
| 2 | 190.4 ms | 181.6 ms | −4.6% | 10.48 | 10.93 | +4.3% |
| 4 | 360.0 ms | 351.5 ms | −2.4% | 11.10 | 11.36 | +2.4% |

Queue-wait p50 fell from about 20.06 ms to 10.06 ms at c1 and c2; the c4 median was 9.27 ms. Whole-batch backend p50 stayed in the same range. The repeated/cache-hit workload was effectively unchanged: p50 1.16/2.15/3.54 ms at 20 ms versus 1.15/2.17/3.69 ms at 10 ms, with throughput within about 1% except for the client-bound c1 level. All 360 requests per run succeeded.

## Correctness and regression checks

- `python3 -m unittest discover -s tests -v`: 33 passed, 3 intended skips.
- The opt-in real-worker test verifies `Batch Timeout:     10ms` by default and `Batch Timeout:     20ms` with `BATCH_DEADLINE_MS=20`.
- All five opt-in worker/gateway tests pass with the current binaries, including cache miss/hit output equality and 2-vs-4-thread parity within `1e-4`.
- `ctest` in a temporary build runs `batch_timings` and `batch_shapes`; both pass.
- No failed requests in either AWS run.

## Interpretation and limits

The data support the hypothesis that 20 ms was an avoidable low-load queueing tax. A 10 ms default lowers unique-request p50 by 2–9% and raises throughput by 2–9% at the tested concurrency, while keeping the observed batch sizes. The gain is smaller than the thread-tuning and `TCP_NODELAY` changes because backend compute still dominates.

This is a tuning tradeoff, not a free speedup. A shorter deadline can produce smaller batches when arrivals are spread over 10–20 ms. The local high-concurrency sweep shows 10 ms preserves large batches better than 5 ms, but it is still a heuristic. The result is single-host, synthetic, closed-loop, with 60 requests per level; baseline and changed AWS runs are from separate sessions. Do not claim production readiness, a universal optimum, model accuracy, or multi-host scaling.

## Reproduction

```sh
BATCH_DEADLINE_MS=10 build/worker_node 8001 worker_1 models/resnet50-v2-7.onnx
python3 -m unittest discover -s tests -v
BENCH_WORKER_BIN=build/worker_node BENCH_GATEWAY_BIN=build/gateway \
BENCH_MODEL=models/resnet50-v2-7.onnx \
python3 -m unittest tests.test_instrumented_worker tests.test_gateway_latency -v
python3 benchmark-aws/controller.py preflight
python3 benchmark-aws/controller.py run --source instrumented
```

Sanitized aggregate: [`benchmarks/results/2026-11-02-aws-batch-deadline.json`](../benchmarks/results/2026-11-02-aws-batch-deadline.json). Raw logs remain local under `benchmark-aws/runs/`. Cost estimates are in the [AWS cost ledger](aws-cost-ledger.md); the eleven-run project estimate is $0.037929 of the $5 target before tax, transfer, or billing adjustments.

## Attribution

The gateway, cache, batching, and ONNX Runtime backend are upstream work by AbhiramDodda and prior fork changes. The thread tuning, HTTP nodelay, input-alignment fix, and this deadline tuning are the owner's investigation contributions. AI assistance was used for the investigation and implementation; the owner remains responsible for review and publication.
