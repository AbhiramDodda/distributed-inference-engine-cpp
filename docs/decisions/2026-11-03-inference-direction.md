# Inference direction decision — 2026-11-03

## Context

The image benchmark measured 45 requests/s under the full SLO (p95 104 ms, 0 errors, worst worker mean CPU 55.8%) and 86.1 requests/s under latency/error limits only. The 84 requests/s hold used about 95% CPU. The current limits are CPU per worker, routing imbalance, model batch shape, and operational behavior.

The continuation plan asks for a reasoned choice between:

- **A. A bounded offline autotuner** for routing, batching, threads, and cache settings.
- **B. Failure recovery and long-running stability**, including shutdown, validation, failover coverage, and real C++ CI.

This record ranks both and selects the next implementation scope.

## Primary-source review

| Source | Relevant finding | Effect on this project |
| --- | --- | --- |
| [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) | Intra-op threads parallelize work inside an operator; the default follows physical cores. Oversubscription and thread spinning consume CPU. Thread settings and affinity are model- and host-specific. | Two vCPUs per `c6i.large` are already the measured limit. More threads cannot create capacity. Thread tuning must be measured, not assumed. |
| [ONNX Runtime graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) | Graph optimizations are enabled by default; offline optimized models reduce startup work but depend on the same execution provider and hardware. | The service already uses `ORT_ENABLE_EXTENDED`. Offline optimized models are a possible startup improvement, not a capacity fix. |
| [Triton optimization guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) | Dynamic batching is usually the largest gain; multiple model instances help some models but not all. The guide requires `perf_analyzer` measurements for each model. | Batching is already implemented and measured. Additional instances per 2-vCPU host compete for the same CPU. The guide supports measurement-driven tuning, not a universal autotuner. |
| [Triton dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) | Batch size, queue delay, queue policy, and timeouts are per-model tradeoffs. The recommended process is to measure with a performance analyzer. | The project already has a batch deadline and static-batch cap. Further tuning is bounded and needs representative traffic. |
| [Triton architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html) | Serving includes per-model scheduling, health, metrics, model management, and backend isolation, not only kernels. | A production service needs operational behavior, not only faster inference. |

## Option comparison

| Criterion | A. Offline autotuner | B. Failure recovery and stability |
| --- | --- | --- |
| User benefit | May find a better configuration for one workload. The benchmark already tested threads, batching, routing, and models; remaining gains are uncertain. | Prevents hangs, silent request failures, and undetected C++ regressions. Directly supports safe operation. |
| Evidence | ONNX Runtime and Triton both say tuning is model- and hardware-specific. The project has measurements, but no production traffic sample. | The production-hardening plan names concrete defects: queued handlers can hang on `stop()`, malformed requests can become 500s and trip breakers, failover has no loopback test, CI does not build the real worker/gateway. |
| Feasibility | Medium to high. Requires a search runner, held-out inputs, metadata, and more cloud time. | High. Local and CI only; no paid resources. |
| Cost | Additional AWS campaigns; must be authorized and bounded. | Near zero. Local loopback and CI. |
| Safety | Risk of selecting a faster configuration that changes accuracy or overfits the 500-image subset. Needs strict constraints. | Lower risk. Valid requests must remain byte-compatible; tests cover status codes and shutdown. |
| Measurable outcome | Throughput, latency, CPU, accuracy on held-out images. | Shutdown bound, 400 behavior, failover success, circuit-breaker accounting, CI build result. |
| Scope control | Tempting to expand into a general tuning platform. | Six tasks in an existing plan; bounded files and tests. |

## Decision

**Choose B: failure recovery and stability first.**

Reasons:

1. The measured bottleneck is CPU, and the current configuration is already near the practical CPU ceiling. A search over the same small configuration space is unlikely to produce a large, durable gain.
2. The service has concrete reliability defects that are cheaper to verify and more important for a usable product: unbounded shutdown, missing request validation, no failover regression test, and no real C++ build in CI.
3. The tuning evidence is based on a 500-image subset and one hardware profile. An autotuner trained on that data risks overfitting and would need more representative traffic and cloud budget.
4. Primary serving guidance supports measurement-driven tuning, but the project already has a measurement harness. The missing capability is operational safety, not another search loop.

The offline autotuner remains a reasonable later step, but only after production-like traffic exists and the service has safe shutdown, validation, observability, and failure tests.

## Autotuner guardrails if revisited

If a later phase builds the autotuner, it must:

- Search only a small, documented configuration space.
- Use a held-out image set for final selection.
- Reject candidates that fail accuracy or error constraints.
- Record model hash, data manifest, machine type, run conditions, and candidate parameters.
- Keep manual overrides and safe defaults.
- Report negative and null results.
- Not use the final test set for selection.

## Scope selected for this phase

The implementation follows `docs/superpowers/plans/2026-11-02-production-hardening.md`:

1. Bounded batch-processor shutdown.
2. Graceful SIGTERM/SIGINT shutdown for worker and gateway.
3. JSON request validation with HTTP 400 before routing.
4. Loopback gateway failover and circuit-breaker test.
5. Real worker/gateway build and real-model tests in CI.
6. Documentation and review.

No cloud resource is required for these tasks.

## Sources

Accessed 2026-11-03:

- ONNX Runtime, [Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance/), [Thread Management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), [Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
- NVIDIA Triton Inference Server, [Optimization](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html), [Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html), [Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html).
- Project evidence: `docs/autotune-benchmark.md`, `benchmarks/results/2026-11-02-autotune-summary.json`.
