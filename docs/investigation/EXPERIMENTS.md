# Continuous optimization experiment ledger

This is the resumable record for [CONTINUOUS-OPTIMIZATION.md](CONTINUOUS-OPTIMIZATION.md). Raw logs stay in the Git-ignored `benchmark-aws/runs/` directory. Sanitized summaries may be published under `benchmarks/results/`.

## Cycle 0 — trustworthy starting point

- HEAD `9c7ab13`, branch `main`, remote `origin/main`.
- The timing instrumentation, benchmark runner, and backend-selection work were already committed. The only pre-existing untracked path was `docs/investigation/`.
- Preserved the user's uncommitted thread-tuning work. Did not reset, clean, stash, or delete it.
- Local tools: Python 3.14.4, CMake 4.2.3, g++ 15.2.0, AWS CLI, `perf`, Docker. The system has the ONNX Runtime 1.23.2 runtime but no C++ headers; local real-worker probes used the pinned ORT 1.23.2 release in `/tmp/opencode`.
- Initial checks: `python3 -m unittest discover -s tests -v` → 31 passed, 1 skipped. Later final: 33 passed, 2 intended skips; all 4 opt-in real-worker tests passed when the model and worker were supplied.
- Model SHA-256 `79102261eb6e5fd7af5d27f41316293e388c5cb691e5d25bfb035c4f64fefe31`.
- Pinned benchmark base `c4d8a08b72df810ffad0dce6a993cecae22494bb`.

## Cycle 1 — ORT intra-op thread capacity (accepted)

| Field | Record |
| --- | --- |
| Hypothesis | Four hardcoded ORT intra-op threads oversubscribe a two-vCPU host and inflate whole-batch backend time. |
| Baseline | Pinned upstream source with four threads; prior instrumented AWS run `infer-8bb17685b18e63a4`. End-to-end baseline runs `infer-d4522a282d0dcc42`, `infer-ca8db4050033ae08`. |
| Change | `src/inference_engine.cpp`: default `min(4, hardware_concurrency)` with a floor of 1, `ORT_INTRA_THREADS` override, startup log line. |
| Workload | `benchmark-aws/load.py`, 60 repeated and 60 unique requests at concurrency 1/2/4; 360 requests per run; `c6i.large`, us-east-1. |
| Acceptance rule | Lower unique-input p50 and higher throughput without failures, output mismatch, cache-hit regression, or cleanup/cost failure. |
| Correctness | Default suite 33 passed / 2 intended skips; 4/4 opt-in real-worker tests; 2-vs-4-thread output parity within `1e-4`; 360/360 valid responses per AWS run. |
| Local A/B | `benchmark-aws/runs/local-investigation-2026-11-02/`: two-CPU affinity, 30 unique requests per level. p50 229/384/635 ms → 149/234/404 ms; throughput 4.32/5.19/6.16 → 7.38/9.04/10.08 req/s. |
| AWS A/B | Baseline mean p50 237.9/424.3/701.5 ms → changed 149.9/231.7/403.9 ms at c1/c2/c4. Throughput 4.11/4.65/5.56 → 7.44/9.11/10.26 req/s. Backend p50 196.0/371.6/659.8 → 86.2/168.9/338.5 ms. Repeated workload unchanged. No failures. |
| Decision | Accepted. Supported by matched end-to-end runs, a local paired probe, a negative-control workload, and output parity. |
| Artifacts | `benchmarks/results/2026-11-02-aws-thread-tuning.json`, `docs/aws-thread-tuning.md`, raw runs `infer-d4522a282d0dcc42`, `infer-1f39a2ac8519f743`, `infer-0601374edef6f33d`, `infer-ca8db4050033ae08`. |
| Cost | Four new runs: $0.003958 + $0.003755 + $0.003501 + $0.003958 = $0.015172. Seven-run project estimate $0.024787. Cleanup independently verified. |
| Limits | Single host, synthetic padded inputs, closed-loop client sharing CPU, 60 samples/level. Backend baseline is from a prior session. No p99 claim. |

## Candidate queue after cycle 1

| Priority | Candidate | Evidence | Status |
| ---: | --- | --- | --- |
| — | HTTP keep-alive / Nagle delayed-ACK | Cache-hit p50 ~83 ms through the unmodified gateway vs 0.4–0.7 ms direct; enabling `TCP_NODELAY` reduced AWS cache-hit p50 to 1.16–3.54 ms. | **Accepted in cycle 2** |
| — | Input copy/padding performance | Flatten/pad measured 8/18/39/989 µs for batch 1/2/4/32, negligible against 86–345 ms backend time. The same investigation found and fixed an oversized-input batch-alignment defect. | **Rejected for performance; correctness fixed in cycle 3** |
| 1 | Batching policy | Batch 4 amortizes fixed cost (165 ms/request vs 197 ms at batch 1), but benchmark concurrency is only 1–4. Larger batches would trade tail latency for throughput. | Next candidate; needs a higher-concurrency sweep |
| 2 | Gateway logging / JSON | Per-request `std::endl` logging and JSON parse/dump; nodelay removed most cache-hit latency without touching logging. | Deferred; profile after batching |
| 3 | Cache/routing | Repeated workload is ~100% hits; direct lookups are sub-millisecond. No demonstrated bottleneck. | No action |
| 4 | Measurement quality | Closed-loop client, shared CPU, 60 samples/level, p99 unstable, baseline sessions differ. | Improve with paired runs; do not infer p99 |
| 5 | Multi-host scaling | Not authorized and not measurable on one host. | Out of scope; document only |

## Cycle 2 — HTTP keep-alive / Nagle (accepted)

| Field | Record |
| --- | --- |
| Hypothesis | `CPPHTTPLIB_TCP_NODELAY` defaults to false. With keep-alive, the worker response, gateway response, and gateway request each can stall on Nagle/delayed-ACK, adding ~40 ms per affected hop. |
| Baseline | Accepted thread-tuned source with two ORT threads; AWS runs `infer-1f39a2ac8519f743` and `infer-0601374edef6f33d` (patch `9b4c0303…`). |
| Change | Set `TCP_NODELAY` on the worker HTTP server, the gateway HTTP server, and the gateway's worker client. Added `GATEWAY_PORT` for loopback tests without changing the default port. |
| Alternatives | Keep the default and document the latency; change only the gateway client; replace keep-alive. Each is smaller but leaves at least one hop affected. |
| Expected effect | Cache-hit p50 should fall from ~98 ms end-to-end to near the direct-worker range; unique-request end-to-end should improve by roughly the removed HTTP stall while backend time is unchanged. |
| Adverse effects | More, smaller TCP segments; negligible on loopback and small JSON responses. No correctness effect expected. |
| Workload | Same `load.py` 360-request workload; local A/B with cache-hit and unique levels; AWS A/B against the committed thread-tuned baseline. |
| Acceptance rule | Lower cache-hit p50 without throughput regression, no failed requests, unchanged outputs and request association, and verified cleanup. |
| Correctness | Default suite 33 passed / 3 intended skips; all 5 opt-in real-worker/gateway tests passed; miss and hit outputs equal; request IDs match; source-delivery checks include `src/gateway.cpp`. |
| Local A/B | `benchmark-aws/runs/local-cycle2-2026-11-02/`: two-CPU affinity. Cache-hit p50 83.2/84.0/83.9 ms → 1.6/2.1/4.3 ms; unique p50 149.1/235.2/408.4 ms → 108.0/193.6/365.4 ms. Backend unchanged. 240/240 requests succeeded. |
| AWS A/B | Cache-hit p50 98.26/98.13/97.94 ms → 1.16/2.15/3.54 ms at c1/c2/c4; system throughput 13.81/27.72/55.26 → 311.49/894.77/955.64 req/s. Unique p50 149.85/231.69/403.91 → 108.02/190.42/360.00 ms; throughput 7.44/9.11/10.26 → 9.22/10.48/11.10 req/s. Backend p50 unchanged within 1.2%. No failures. |
| Decision | Accepted. The cache-hit failure mode is removed, unique requests improve, backend compute is unchanged, and correctness checks pass. |
| Artifacts | `benchmarks/results/2026-11-02-aws-http-nodelay.json`, `docs/aws-http-nodelay.md`, raw runs `infer-ff9ac124eb8e79ea`, `infer-2063d463870454ba`. |
| Cost | Two new runs: $0.003273 + $0.003450 = $0.006723. Nine-run project estimate $0.031510. Cleanup independently verified. |
| Limits | Single host, synthetic padded inputs, closed-loop client sharing CPU; high cache-hit throughput is client-bound. Baseline and changed AWS sessions differ. No p99 claim. |
| Status | Accepted and documented. |

## Cycle 3 — input shape alignment (accepted correctness fix)

| Field | Record |
| --- | --- |
| Hypothesis | `InferenceEngine::batchPredict` appends every input element and pads only short inputs. An oversized input therefore occupies more than one model-sized slot and shifts every later sample in the batch, producing wrong outputs while still returning the expected shape. Input preparation itself may be too small to optimize. |
| Baseline | Accepted cycle-2 source (thread-tuned, TCP_NODELAY). |
| Preparation measurement | A standalone C++ probe measured flatten/pad at 8/18/39/989 µs for batch 1/2/4/32, against 86–345 ms whole-batch backend time. The performance candidate was rejected. |
| Reproduction | A direct `batchPredict({oversized, small})` call produced a small-sample output that differed from its single-request reference by 10.1369. The new `tests/test_batch_shapes.cpp` failed with “sample after an oversized input was misaligned” before the fix. |
| Change | Cap each sample at the model’s per-sample element count, pad short samples, and truncate oversized samples. Add guards for an empty output tensor and an output batch dimension that differs from the input batch. |
| Verification | `test_batch_shapes` passes with the real model; the default suite remains green; all opt-in worker/gateway tests pass. No AWS run was needed because the defect is host-independent and the benchmark workload uses short inputs. |
| Decision | Accepted as a correctness fix. It is not a performance claim. |
| Cost | $0; local only. |
| Artifacts | `tests/test_batch_shapes.cpp`, `src/inference_engine.cpp`, `CMakeLists.txt`; local output `/tmp/opencode/build/test_batch_shapes.log`. |

## Checkpoint rules

- After each accepted change, update this ledger, the README/results, the cost ledger, and the resume only with verified work.
- Keep the previously accepted source as the next baseline. Do not label the pinned upstream as the current baseline.
- Do not commit or push without checking the exact diff, tests, raw artifacts, secrets, and AWS cleanup.
