# AWS CPU baseline — 2026-11-02

A temporary `c6i.large` in `us-east-1` ran the upstream gateway and one ONNX Runtime CPU worker on the same instance as the closed-loop load client. The checkout was `c4d8a08b72df810ffad0dce6a993cecae22494bb`. All 360 requests passed the client's response-shape checks; none failed. The model received three synthetic floats padded by the engine to its input dimensions. The run did not assess model prediction accuracy.

| Inputs | Concurrency | Requests | Cache hits | Successful requests/s | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 1 | 60 | 59 | 13.37 | 98.36 | 109.93 | 240.27 |
| Repeated | 2 | 60 | 60 | 28.28 | 98.46 | 109.92 | 118.93 |
| Repeated | 4 | 60 | 60 | 54.67 | 98.54 | 109.93 | 119.42 |
| Unique | 1 | 60 | 0 | 3.92 | 250.53 | 309.90 | 369.96 |
| Unique | 2 | 60 | 0 | 4.79 | 409.49 | 501.09 | 524.22 |
| Unique | 4 | 60 | 0 | 5.46 | 711.09 | 912.38 | 1114.38 |

The cache-hit and cache-miss paths have different latency and throughput in this run. Unique-input throughput increased modestly as concurrency increased, while latency rose. These observations do not establish the cause or an optimization opportunity by themselves.

Each row has only 60 observations, so the reported p99 is sensitive to individual requests. The client and server share CPU, the input is synthetic, the runs are short, and the load is closed-loop. The results do not establish a maximum throughput, an SLO, a distributed-systems scaling result, or a before/after improvement.

A sanitized [summary](../benchmarks/results/2026-11-02-aws-baseline.json) is included in the repository. Raw request records, cloud-init and service logs, model checksum, CPU samples, lifecycle log, and verified archive are in the local, Git-ignored run directory `benchmark-aws/runs/infer-b3ee717ce6dc11fe/`. The controller recorded verified cleanup, and separate AWS queries confirmed that the recorded instance was terminated and its volume and security group were absent. Cost estimates and assumptions are in [the cost ledger](aws-cost-ledger.md).
