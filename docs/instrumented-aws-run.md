# Instrumented AWS run — 2026-11-02

A temporary `c6i.large` in `us-east-1` ran the pinned upstream source with the local timing patch. The remote host verified the source and script hashes before building. One gateway, one ONNX Runtime CPU worker, and the load generator shared the same instance. All 360 requests passed response-shape checks. All 181 cache misses carried complete queue, backend batch-duration, and batch-size measurements.

## Cache-miss observations

Each row contains 60 unique-input requests. Values are nearest-rank request-weighted medians from the benchmark summary. Batch execution is the wall time for the whole backend batch, repeated for each request in that batch; it includes backend locking, tensor preparation, inference, and output handling. It is not per-request pure model compute time.

| Concurrency | Requests/sec | End-to-end p50 | Queue wait p50 | Whole-batch backend p50 | Observed batch size |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.08 | 249.85 ms | 20.06 ms | 196.05 ms | 1 for all 60 requests |
| 2 | 4.71 | 419.75 ms | 20.06 ms | 371.59 ms | 2 for all 60 requests |
| 4 | 5.52 | 722.20 ms | 18.18 ms | 659.77 ms | 4 for all 60 requests |

The median queue wait stayed near the 20 ms batching deadline, while measured whole-batch backend duration increased with batch size. These observations suggest the backend call is a useful next area to investigate; they do not identify which work inside it dominates or prove that a proposed optimization will help. Medians cannot generally be added to reconstruct an individual request. Each row has only 60 requests, the input is synthetic, the test is closed-loop, and the client shares the CPU with the server. This run is not a multi-host scaling test or a before/after optimization comparison.

## Evidence and cleanup

A sanitized [summary](../benchmarks/results/2026-11-02-aws-instrumented.json) is included in the repository. The source patch and manifest, raw request records, service logs and archive are saved locally in `benchmark-aws/runs/infer-8bb17685b18e63a4/`. The downloaded archive SHA-256 matched the remote record. The controller verified the recorded instance in `terminated`; separate AWS checks reported that its recorded volume and security group did not exist, and no active instance with the benchmark tag remained.

The instance was launched at 09:57:54 UTC and termination was requested at 10:00:29 UTC: 155 seconds for the cost estimate. The estimated third-run subtotal is $0.003932, and the three-run project subtotal is $0.009615 before data transfer, tax, credits, or other billing adjustments. See [the cost ledger](aws-cost-ledger.md). The AWS bill itself was not verified.
