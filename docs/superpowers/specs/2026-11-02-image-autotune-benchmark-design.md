# Image Auto-Tuning Benchmark Harness — Design

## Intent

Build a reusable harness that answers one question on fixed AWS hardware:

> Given several image-classification ONNX models, which model, configuration, and architecture gives the best quality and the most usable capacity under a defined SLO?

The harness is the product. It must run a default baseline, compare candidates one at a time under the same workload, find the saturation knee, test failures and consistency, and produce a ranked report that a human can defend in a design review.

This is an isolated benchmark deployment. It is not a public service and not proof of production readiness.

## Goals

- Measure real ImageNet accuracy, not only response shape.
- Measure capacity under many concurrent clients and find the SLO knee.
- Compare multiple popular image models that fit the fixed hardware.
- Compare parameter settings and genuinely different architectures.
- Test worker failure, request consistency, and cache behavior.
- Keep every comparison fair: same data, preprocessing, host type, client, warm-up, and cache state.
- Stay under **$10 total** for the complete benchmark campaign.
- Produce resume-ready, verified evidence and a production-readiness review.

## Non-goals

- Public deployment, authentication, TLS termination, multi-tenancy, billing, or autoscaling.
- Text, embedding, detection, segmentation, or generative models in the first version.
- GPU, Kubernetes, multi-region, or serverless.
- Training or fine-tuning.
- A universal “best model” claim. The result is best for this hardware and SLO.

## Anti-goals

- A benchmark that reports only cache-hit throughput.
- Comparing models with different preprocessing or different data.
- Letting the client saturate and calling it a server limit.
- Reporting p99 from a few dozen samples.
- Selecting a winner by one number while hiding accuracy or tail-latency regressions.
- Spending the full budget before the harness is verified.

## Fixed constraints

| Item | Decision |
| --- | --- |
| Worker hardware | `c6i.large`, 2 vCPU, 4 GiB, `us-east-1` |
| Gateway hardware | `c6i.large` |
| Load generator | `c6i.xlarge`, separate host |
| Maximum workers | 8 |
| Node counts tested | 1, 2, 4, 8 workers |
| Total budget | **$10**, including setup, reruns, storage, IPv4, and transfer |
| Runtime cap | 30 minutes per workload run; 60-minute instance watchdog |
| App exposure | Private only; security groups restrict app ports |
| SSH | Local public IPv4 only, ephemeral key |

Estimated cost at 8 workers plus gateway and load generator: about **$0.94/hour**. The campaign should fit in 3–5 hours of paid runtime with contingency.

## Models

All models are ImageNet classifiers with 224×224 RGB input and the same mean/std preprocessing. Keep model files out of Git; download them into ignored storage and verify SHA-256.

| Model | Size | Top-1 | Top-5 | License | Role |
| --- | ---: | ---: | ---: | --- | --- |
| ShuffleNetV2-10 | 9.2 MB | 69.36% | 88.32% | BSD-3-Clause | Smallest/fastest |
| MobileNetV2-7 | 13.5 MB | 70.94% | 89.99% | Apache 2.0 | Mobile baseline |
| ResNet18-v2-7 | 44.6 MB | 69.70% | 89.49% | Apache 2.0 | Mid-size |
| ResNet34-v2-7 | 83.2 MB | 73.36% | 91.43% | Apache 2.0 | Larger |
| ResNet50-v2-7 | 97.7 MB | 75.81% | 92.82% | Apache 2.0 | Highest accuracy, already pinned |

Optional later: quantized MobileNetV2-12-int8 and ShuffleNetV2-12-int8 to study the accuracy/throughput tradeoff. Do not include them in the first matrix unless time and budget remain.

## Dataset and quality

- Dataset: `ndb796/Small-ImageNet-Validation-Dataset-1000-Classes`, pinned commit `c55ca047bd854ba01d3af6c283c034924a644db2`.
- Selection: one image from each class folder 0–499, chosen by sorted filename. 500 labeled images.
- Labels: `imagenet_class_index.json` and `imagenet.json` from the same commit.
- Preprocessing: resize shortest side to 256, center crop 224, RGB, scale to `[0,1]`, ImageNet mean/std, HWC→CHW float32.
- Quality: top-1 and top-5 accuracy per model on the same 500 images.
- Do not redistribute images. Download into ignored `benchmark-aws/data/` and record per-file SHA-256.
- A model whose accuracy is far outside the published range is a correctness failure, not a performance result.

## SLO and capacity definition

Default SLO:

| Target | Value |
| --- | --- |
| Availability | 99.9% |
| Error budget | 0.1% |
| p95 latency | ≤ 500 ms |
| p99 latency | ≤ 1 s |
| Timeout | 2 s |
| CPU headroom | ≤ 80% per host |
| Hold time | 5 minutes at a level |

Capacity is the highest sustained offered load that holds all SLO targets for the hold time. Also report:

- the knee: where p95 leaves the SLO or errors start;
- the collapse point: where throughput stops rising and latency grows without bound;
- safe capacity: capacity with CPU headroom;
- absolute capacity: best throughput regardless of SLO, for diagnosis only.

The SLO is configurable in the harness. Do not change it between candidates in the same comparison.

## Workloads

- **Accuracy:** 500 unique images, one pass.
- **Unique:** 500 images, cache-miss path.
- **Repeated:** a pool of 20 images, cache-hit path.
- **Mixed:** 80% from the 20-image pool, 20% unique from the 500-image set.
- **Closed-loop:** concurrency 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 as needed.
- **Open-loop:** fixed arrival rates from 5 req/s up to the measured knee, 60–300 seconds per level.
- **Warm-up:** at least 20 requests or 10 seconds, excluded from results.
- **Cache state:** restart workers between candidates; same warm-up policy for every candidate.

## Candidate space

### Architecture modes

1. **Baseline** — current gateway → worker process, worker-side dynamic batching, consistent-hash routing, LRU cache, JSON transport.
2. **Binary transport** — baseline plus raw float32 `/infer_tensor`; removes JSON float serialization.
3. **Process pool** — two worker processes per host, each with one ORT thread, versus one process with two ORT threads. Same total CPU and memory budget.
4. **Gateway batching** — gateway accumulates requests and sends one batch request to a worker. Higher implementation cost; tests whether per-request worker HTTP is a bottleneck.
5. **Least-loaded routing** — gateway routes by current in-flight/queue load instead of input hash. Tests cache locality versus load balance.
6. **Direct worker** — client sends directly to one worker, bypassing the gateway. Diagnostic upper bound only; not deployable.

Architecture modes 1–2 are required for the first campaign. Mode 3 is a deployment change. Modes 4–5 are code changes; include them only if the first campaign shows the gateway or routing is the bottleneck. Mode 6 is a diagnostic baseline.

### Settings

- ORT intra-op threads: 1, 2, 4.
- Batch deadline: 5, 10, 20 ms.
- Maximum batch size: 8, 16, 32, 64.
- Cache capacity: 0, 100, 1000.
- Transport: JSON, binary.

Do not run the full Cartesian product. Use staged search.

## Staged search

### Stage 1 — Model screening

Run all five models with the baseline architecture, default settings, unique images at a moderate load, and one saturation preview. Record accuracy, p50/p95, throughput, memory, and model load time. Keep the top two or three models by quality and capacity.

### Stage 2 — Architecture screening

Run the surviving models across architecture modes 1, 2, and 3 at a fixed moderate load and a higher load. Keep the modes that improve capacity or tail latency without a quality change.

### Stage 3 — Parameter tuning

For the best model/architecture pairs, run a coordinate search:

1. ORT threads.
2. Batch deadline.
3. Maximum batch size.
4. Cache capacity.

Use short screening runs, then repeat the best two settings with full SLO sweeps. Do not tune on cache hits alone; include unique and mixed workloads.

### Stage 4 — Capacity sweep

For the finalists, sweep concurrency or arrival rate to find the knee and collapse point. Repeat each level if variance is high. Report the SLO capacity and the absolute limit separately.

### Stage 5 — Failure and consistency

- Kill one worker during a mixed load; verify successful retries and circuit-breaker state.
- Send malformed requests; verify 400 and no breaker failure (requires the hardening plan).
- Send the same image 100 times; verify identical output hashes across cache hit and miss.
- Restart a worker and verify recovery.
- Repeat the finalist run twice; report run-to-run variation.

## Fairness rules

- Same model weights, dataset, preprocessing, image order, and label mapping.
- Same host type, AMI, ORT version, compiler flags, and CPU budget.
- Same client on a separate host; client CPU and event-loop lag recorded.
- Same warm-up, cache reset, and measurement window.
- Randomized candidate order where practical; alternate repeated runs.
- Physical host metrics collected per process and per host.
- Any candidate that fails accuracy, request association, or cleanup is disqualified before performance ranking.
- Report neutral and negative results. A faster model with worse accuracy is a tradeoff, not an automatic winner.

## Harness architecture

```
benchmark-aws/autotune/
  models.json          # model URLs, SHA-256, input shape, accuracy, preprocessing
  data.py              # ImageNet subset download, labels, tensor cache
  loadgen.py           # async HTTP client, binary/JSON, open/closed loop
  workloads.py         # unique/repeated/mixed selection
  candidates.py        # architecture/settings definitions and staged search
  analyze.py           # SLO capacity, percentiles, Pareto ranking
  report.py            # Markdown + JSON report
  matrix/              # candidate matrix files
benchmark-aws/multihost_controller.py
benchmark-aws/run_autotune.py
docs/benchmark-harness.md
```

Key interfaces:

- `run_autotune.py --matrix matrix/default.json --output runs/<id>`
- The runner provisions once, deploys models and data, runs candidates sequentially, and cleans up once.
- Each candidate records its model hash, data hash, git revision, config, host metadata, raw request records, and per-host metrics.
- The analyzer produces a ranked table, SLO capacity, saturation curves, failure results, consistency results, and a Pareto frontier for accuracy versus capacity.
- A `--dry-run` prints the candidate order, estimated requests, estimated runtime, and estimated cost before any paid action.

## Topology and security

- Gateway `c6i.large` with app port 8000 open only to the load-generator security group.
- Workers `c6i.large` with app ports open only to the gateway security group.
- Load generator `c6i.xlarge` with no inbound application port.
- SSH only from the local public IPv4 `/32`, ephemeral keys, host-key verification through EC2 console output.
- No public inference endpoint. Verify with a public probe that 8000/8001/8002 are unreachable.
- Tag every resource with the run ID; cleanup verifies instances, volumes, and security groups are gone.
- No customer data. No credentials in logs or results.

## Outputs

1. **Raw results:** request JSONL, per-host metrics, service logs, model/data hashes, candidate manifest.
2. **Sanitized summary:** ranked candidates, accuracy, SLO capacity, saturation curves, failure and consistency results.
3. **Human report:** model and architecture tradeoffs, winner per objective, cost per 1,000 images and per 1,000,000 inferences, and limitations.
4. **Production-readiness review:** PASS/BLOCKED/OUT OF SCOPE for auth, TLS, limits, deadlines, shutdown, validation, failover, observability, container quota, model versioning, and licensing.
5. **Resume evidence:** only claims supported by the measured results.

## Production analysis

The benchmark does not make the service production-ready. The final report must state the blockers explicitly. At minimum:

- No authentication or authorization.
- No TLS; private-network-only is a benchmark control, not a production control.
- No request-rate or queue-depth limit; risk of memory exhaustion.
- No per-request deadline/cancellation.
- No observability stack or log rotation.
- `hardware_concurrency()` does not respect container CPU quotas.
- Model and data licenses need review before redistribution.
- Gateway passthrough trusts worker response bodies.
- Shutdown and validation remain blockers until the hardening plan is complete.

## Decisions left to the builder

- Exact async HTTP client library and connection-pool settings.
- Whether gateway batching or least-loaded routing is implemented in the first campaign or deferred.
- Whether quantized model variants are included after the first campaign.
- Exact candidate counts and repetitions within the $10 budget.
- Whether the 8-worker run uses one availability zone or multiple zones.
- How to display the Pareto frontier and saturation curves in the report.

## Risks

- The client may saturate before the server; mitigate with a separate `c6i.xlarge`, async I/O, client CPU monitoring, and a client-bound label.
- Model download or dataset download may fail; verify checksums and retain partial logs.
- Worker memory may be tight for ResNet50 with two processes; measure RSS and skip the process-pool mode if it swaps.
- Multi-host setup and cleanup are the largest operational risk; test the controller with a tiny smoke run first.
- The $10 budget can be consumed by a long 8-worker campaign; use a dry-run cost estimate, per-run caps, and stop on cleanup uncertainty.
- Accuracy on a 500-image sample has sampling error; report the sample result and the published baseline, not a false precision claim.
