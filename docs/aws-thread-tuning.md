# ORT intra-op thread tuning — AWS CPU A/B

## Problem and change

The worker hardcoded ONNX Runtime to four intra-op threads. The benchmark host is a `c6i.large` with two vCPUs, so four compute threads competed for two hardware threads while the HTTP handlers waited on the same CPU. The instrumented baseline put most unique-request latency inside the whole-batch backend call: about 196 ms at batch size 1, 372 ms at size 2, and 660 ms at size 4.

This change makes the default capacity-aware:

```cpp
const unsigned int hardware = std::thread::hardware_concurrency();
return static_cast<int>(std::max(1u, std::min(4u, hardware == 0 ? 1u : hardware)));
```

On the two-vCPU benchmark host the default is two threads; hosts with four or more CPUs keep the previous four-thread cap. `ORT_INTRA_THREADS` is an explicit override for experiments or operations, and the worker logs the effective count at startup:

```text
ORT intra-op threads: 2 (hardware_concurrency=2)
```

Source: `src/inference_engine.cpp`. The AWS source-delivery allowlist now includes that file; the controller and verifier reject an instrumented run whose source is missing the thread selection.

## Why this candidate

- The instrumented AWS run showed the backend call dominating unique-request latency, so a backend-side change had the largest plausible effect.
- A local matched probe on two logical CPUs measured whole-batch p50 of 164–197 ms with four threads and 85 ms with two threads.
- The change is one default and one log line. It does not change the API, model, batch policy, cache, or wire format.
- The alternative HTTP `TCP_NODELAY` finding was reproduced (cache-hit p50 83 ms to 0.8 ms) but was not selected: it targets HTTP keep-alive behavior rather than the backend and would confound a single-change comparison. It is recorded as a follow-up in [FINDINGS](investigation/FINDINGS.md).

## Benchmark method

- Temporary on-demand `c6i.large`, `us-east-1`, one gateway, one CPU worker, and the closed-loop Python client on the same instance. No application port was exposed.
- Pinned base revision `c4d8a08b72df810ffad0dce6a993cecae22494bb`, model `resnet50-v2-7.onnx` SHA-256 `79102261eb6e5fd7af5d27f41316293e388c5cb691e5d25bfb035c4f64fefe31`, ORT 1.23.2.
- Each run: 60 repeated-input and 60 unique-input requests at concurrency 1, 2, and 4 (360 requests). Same model, load script, warm-up, timeouts, and cache state.
- Two runs per variant, alternating order: baseline, changed, changed, baseline. Baseline mode uses the pinned upstream source; changed mode uses the reviewed patch. Run records contain the exact patch and manifest.
- The controller verified termination and the absence of the recorded volume and security group after every run. Raw records are in the local, Git-ignored `benchmark-aws/runs/infer-*` directories.

## Results

Unique inputs exercise the cache-miss, batching, and backend path. Values are means of the two runs; each run has 60 observations per level. Change is changed relative to baseline.

| Concurrency | Baseline p50 | Changed p50 | p50 change | Baseline req/s | Changed req/s | Throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 237.9 ms | 149.9 ms | −37.0% | 4.11 | 7.44 | +80.9% |
| 2 | 424.3 ms | 231.7 ms | −45.4% | 4.65 | 9.11 | +95.7% |
| 4 | 701.5 ms | 403.9 ms | −42.4% | 5.56 | 10.26 | +84.5% |

Repeated inputs are cache hits and do not enter ONNX Runtime. They are the negative control: p50 stayed at about 98 ms and throughput stayed at about 13.5, 27.6, and 55.3 req/s. There were no failures in any run.

Whole-batch backend p50 from the instrumented records:

| Concurrency | Baseline 4 threads | Changed 2 threads | Change |
| ---: | ---: | ---: | ---: |
| 1 | 196.0 ms | 86.2 ms | −56.0% |
| 2 | 371.6 ms | 168.9 ms | −54.5% |
| 4 | 659.8 ms | 338.5 ms | −48.7% |

The baseline backend values come from the earlier instrumented four-thread run (`infer-8bb17685b18e63a4`), not from the no-timing baseline runs; both used the same instance type and workload, but the session differs. The local matched probe, which used the same gateway and source except for the thread count, showed the same direction: backend p50 180/331/572 ms with four threads versus 88/173/341 ms with two threads.

## Correctness and repeatability

- `python3 -m unittest discover -s tests -v`: 33 passed, 2 skipped (the opt-in real-worker tests).
- The opt-in real-worker tests pass with the changed binary: the default thread count matches `min(4, cpu_count)`, the override logs the requested count, and outputs from two-thread and four-thread workers agree within `1e-4` on fixed inputs.
- All 360 requests in each AWS run had a valid response shape and matching request ID; no HTTP or client errors were recorded.
- The two changed runs produced p50 values within about 3% of each other at concurrency 1 and within about 3% at concurrency 4. The two baseline runs were similarly stable.
- Alternating the run order did not reverse the result.

## Interpretation and limits

The data support the hypothesis that four intra-op threads oversubscribed the two-vCPU host. Reducing the pool to two roughly halved whole-batch backend time and raised unique-input throughput by about 1.8–2.0×. End-to-end latency fell less than backend time because the fixed 20 ms batching deadline, HTTP handling, and response serialization remain in the path.

This is a single-host, synthetic-input, closed-loop experiment. The client shares the two vCPUs with the gateway and worker; each level has 60 requests per run; p99 is not reported as a stable estimate. The result does not establish model accuracy, production readiness, multi-host scaling, or a maximum server capacity. The backend baseline is from a different session than the changed runs, though the local matched probe and the negative-control workload reduce that concern. The HTTP delayed-ACK finding remains an unselected follow-up.

## Reproduction

```sh
python3 -m unittest discover -s tests -v
python3 benchmark-aws/controller.py preflight
python3 benchmark-aws/controller.py run --source baseline
python3 benchmark-aws/controller.py run --source instrumented
```

Sanitized aggregate: [`benchmarks/results/2026-11-02-aws-thread-tuning.json`](../benchmarks/results/2026-11-02-aws-thread-tuning.json). Raw logs, request records, patches, manifests, and cleanup records remain local under `benchmark-aws/runs/`. Cost estimates are in the [AWS cost ledger](aws-cost-ledger.md); the seven-run project estimate is $0.024787 of the $5 target before tax, transfer, or billing adjustments.

## Attribution

The gateway, cache, batching, ONNX Runtime backend, benchmark client, and original instrumentation are upstream work by AbhiramDodda and prior fork changes, not this contribution. This investigation's contribution is the capacity-aware thread default, its verification, the matched A/B, and this report. AI assistance was used for the investigation and implementation; the owner remains responsible for review and any publication.
