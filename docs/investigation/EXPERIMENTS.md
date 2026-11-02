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
| 1 | HTTP keep-alive / Nagle delayed-ACK | Cache-hit p50 ~83 ms through the unmodified gateway vs 0.4–0.7 ms direct; values quantized near 1/42/84 ms; enabling `TCP_NODELAY` on worker server, gateway server, and gateway client reduced p50 to 0.8 ms locally. | **Selected for cycle 2** |
| 2 | Input copy/padding/allocation | Six copies of the request vector on the cache-miss path; synthetic inputs are padded from 3 to 150,528 floats. Expected gain is small relative to backend time, but preparation is currently outside `batch_execution_us`. | Deferred; needs a preparation-only measurement |
| 3 | Batching policy | Batch 4 amortizes fixed cost (165 ms/request vs 197 ms at batch 1), but benchmark concurrency is only 1–4. Larger batches would trade tail latency for throughput. | Deferred; needs a higher-concurrency sweep |
| 4 | Gateway logging / JSON | Per-request `std::endl` logging and JSON parse/dump; nodelay probe removed most cache-hit latency without touching logging. | Lower priority; profile after cycle 2 |
| 5 | Cache/routing | Repeated workload is ~100% hits; direct lookups are sub-millisecond. No demonstrated bottleneck. | No action |
| 6 | Measurement quality | Closed-loop client, shared CPU, 60 samples/level, p99 unstable, backend baseline session differs. | Improve with paired runs; do not infer p99 |
| 7 | Multi-host scaling | Not authorized and not measurable on one host. | Out of scope; document only |

## Cycle 2 — HTTP keep-alive / Nagle (in progress)

| Field | Record |
| --- | --- |
| Hypothesis | `CPPHTTPLIB_TCP_NODELAY` defaults to false. With keep-alive, the worker response, gateway response, and gateway request each can stall on Nagle/delayed-ACK, adding ~40 ms per affected hop. |
| Baseline | Current accepted thread-tuned source with two ORT threads; AWS runs `infer-1f39a2ac8519f743` and `infer-0601374edef6f33d` (patch `9b4c0303…`). |
| Intended one-variable change | Set `TCP_NODELAY` on the worker HTTP server, the gateway HTTP server, and the gateway's worker client. No API or numerical change. |
| Alternatives | Keep the default and document the latency; change only the gateway client; replace keep-alive. Each is smaller but leaves at least one hop affected. |
| Expected effect | Cache-hit p50 should fall from ~98 ms end-to-end to near the direct-worker range; unique-request end-to-end should improve by roughly the removed HTTP stall while backend time is unchanged. |
| Adverse effects | More, smaller TCP segments; negligible on loopback and small JSON responses. No correctness effect expected. |
| Workload | Same `load.py` 360-request workload; local A/B with cache-hit and unique levels; AWS A/B against the committed thread-tuned baseline. |
| Acceptance rule | Lower cache-hit p50 without throughput regression, no failed requests, unchanged outputs and request association, and verified cleanup. |
| Regression check | Opt-in gateway/worker integration test with a generous loopback latency ceiling; no brittle sub-millisecond assertion. |
| Source identity | Baseline patch `9b4c0303f004235a091ebac22acb89c12a9631106650c364663bddab4e3ad87a`; changed patch recorded by the runner. |
| Cost plan | Two changed runs, approximately $0.008 total; stop if cleanup or cost is uncertain. |
| Status | Implementation pending. |

## Checkpoint rules

- After each accepted change, update this ledger, the README/results, the cost ledger, and the resume only with verified work.
- Keep the previously accepted source as the next baseline. Do not label the pinned upstream as the current baseline.
- Do not commit or push without checking the exact diff, tests, raw artifacts, secrets, and AWS cleanup.
