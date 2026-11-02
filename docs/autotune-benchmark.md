# Image auto-tuning benchmark — 2026-11-02

## Summary

This benchmark measures a distributed C++ inference engine on fixed AWS hardware with real labeled images. It uses five ONNX image-classification models, a 500-image ImageNet subset, cold-cache capacity sweeps, a 5-minute SLO hold, and a live worker-failure test.

Main verified results:

- ResNet50-v2-7 latency/error capacity scales from **13.5 to 86.1 requests/s** when the worker count scales from 1 to 8 with least-loaded routing.
- A 5-minute hold at **45 requests/s** passed all SLO limits: **13,500 requests, 0 errors, p95 104.0 ms, p99 105.8 ms, worst worker mean CPU 55.8% (maximum sample 67%)**.
- A 5-minute hold at **84 requests/s** passed latency and errors (**25,200 requests, p95 176.6 ms, p99 183.4 ms**) but used about **95% worker CPU**, above the 80% limit. It is a latency-only result, not a full-SLO result.
- Least-loaded routing raised 8-worker latency/error capacity from **31.4 to 86.1 requests/s** (2.7×) compared with the consistent-hash baseline.
- ShuffleNetV2-10 reached **201 requests/s** on 2 workers with **72.2% top-1 / 90.6% top-5**. ResNet50-v2-7 reached **21.9 requests/s** with **78.6% top-1 / 94.2% top-5** on the same data. These model rates are latency-only because CPU was not reduced for the model screen.
- A worker kill during 40 requests/s traffic caused **0 failed requests**. The same 20 images were served by up to 8 workers with **0 top-1 conflicts and 0.0 maximum output difference**.
- Estimated AWS cost: **$3.31 of the $10 budget**. Cleanup verification found **0 active instances, 0 tagged security groups, and 0 tagged volumes**.

These results apply to the tested workload and topology. They do not prove production readiness. The last section lists the open items.

## Scope and SLO

| Item | Value |
| --- | --- |
| Region | `us-east-1` |
| Worker and gateway | `c6i.large` (2 vCPU, 4 GiB) |
| Load generator | `c6i.xlarge` (4 vCPU, 16 GiB) |
| Worker counts | 1, 2, 4, 8 |
| Model | ResNet50-v2-7 for scaling; five models for screening |
| Data | 500 labeled images, one per class, class indices 0–499 |
| Transport | Raw float32 little-endian `POST /infer_tensor` |
| Cache | `CACHE_CAPACITY=1` for capacity and hold runs |
| Routing | Least-loaded unless labeled `hash` |

SLO:

| Metric | Limit |
| --- | ---: |
| Availability | 99.9% |
| p95 latency | ≤ 500 ms |
| p99 latency | ≤ 1 s |
| Error rate | ≤ 0.1% |
| Worker CPU | ≤ 80% |
| Hold | 5 minutes |

An SLO capacity result is the highest measured throughput at a concurrency level that satisfies the latency and error limits. The concurrency sweep is coarse, so the true knee is between the last passing level and the first failing level.

The capacity sweep does not enforce the 80% CPU limit. The CPU limit is checked separately with 5-minute holds and the saved `vmstat` samples. The report labels each result as **latency/error capacity** or **full-SLO sustained capacity**. The full-SLO result is the stronger claim.

## Method

### Data and models

The dataset is `ndb796/Small-ImageNet-Validation-Dataset-1000-Classes` at commit `c55ca047bd854ba01d3af6c283c034924a644db2`. The harness selects one image per class for classes 0–499, resizes the short side to 256, center-crops 224×224, applies ImageNet mean and standard deviation, and writes little-endian float32 NCHW tensors. The manifest records the image SHA-256 and tensor SHA-256.

Models are pinned by SHA-256 in `benchmark-aws/autotune/models.json`:

| Model | Top-1 | Top-5 | License |
| --- | ---: | ---: | --- |
| ShuffleNetV2-10 | 69.36 | 88.32 | BSD-3-Clause |
| MobileNetV2-7 | 70.94 | 89.99 | Apache-2.0 |
| ResNet18-v2-7 | 69.70 | 89.49 | Apache-2.0 |
| ResNet34-v2-7 | 73.36 | 91.43 | Apache-2.0 |
| ResNet50-v2-7 | 75.81 | 92.82 | Apache-2.0 |

The published values above are references from the ONNX Model Zoo. The measured values in this report come from the 500-image subset and are not directly comparable to full ImageNet validation.

### Fairness controls

- Every capacity level starts fresh worker and gateway processes. The controller stops the old processes, starts new ones, and rejects a worker whose `/health` counters are not zero.
- Capacity levels use `CACHE_CAPACITY=1` and a unique 500-image workload, so each request runs inference. The summaries confirm `cache_hit_rate=0`.
- The load generator is a separate `c6i.xlarge` host. It records client CPU and per-request latency, status, node ID, cache state, batch size, queue wait, and batch execution time.
- Accuracy runs use 500 labeled images and real model outputs. A response-shape check is not used as accuracy evidence.
- The controller creates private application security groups. Port 8000 allows only the load group, and port 8001 allows only the gateway group. SSH allows only the owner IPv4 `/32`.

### Routing policies

The gateway supports three policies:

- `hash`: consistent hashing by input tensor. This preserves cache locality.
- `least_loaded`: choose the worker with the fewest in-flight requests. Equal loads rotate across candidates.
- `bounded_load`: use the hash target until its in-flight load exceeds a factor of the cluster average, then fall back to least-loaded.

The capacity comparison uses `hash` and `least_loaded`. The default remains `hash`; the benchmark selects the policy with `ROUTING_POLICY`.

### Static batch limit

Some ONNX models declare a fixed input batch dimension of 1. The batch processor originally formed batches larger than 1 and sent them to these models, which caused HTTP 500 responses. The engine now reads the model's static first dimension and caps the effective batch size to that value. Dynamic-batch models keep the configured limit (32 in these runs).

## Results

### Accuracy

All runs use the same 500-image subset. The measured values are from the AWS runs listed in the evidence table.

| Model | Measured top-1 | Measured top-5 | Published top-1 | Published top-5 | Batch dimension | Effective max batch |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| ShuffleNetV2-10 | 72.2% | 90.6% | 69.36% | 88.32% | static 1 | 1 |
| MobileNetV2-7 | 71.8% | 90.8% | 70.94% | 89.99% | static 1 | 1 |
| ResNet18-v2-7 | 71.6% | 90.6% | 69.70% | 89.49% | dynamic | 32 |
| ResNet34-v2-7 | 76.2% | 92.2% | 73.36% | 91.43% | dynamic | 32 |
| ResNet50-v2-7 | 78.6% | 94.2% | 75.81% | 92.82% | dynamic | 32 |

The measured values are 1–3 points above the published values on this subset. The ranking is consistent. Preprocessing and label mapping are therefore credible for this benchmark.

### ResNet50 capacity scaling

All rows use least-loaded routing, a cold cache, 500 unique images per level, and 500 requests per level. These are latency/error capacities; the CPU limit is checked separately in the hold section. The SLO capacity row is the highest measured passing level.

| Workers | Concurrency | Requests/s | p50 ms | p95 ms | p99 ms | SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4 | 13.54 | 294.97 | 298.16 | 307.47 | pass |
| 1 | 16 | 13.61 | 1175.05 | 1245.06 | 1309.49 | fail |
| 2 | 4 | 21.92 | 181.90 | 185.17 | 190.69 | pass |
| 2 | 16 | 22.59 | 701.79 | 719.01 | 751.79 | fail |
| 4 | 16 | 45.66 | 362.96 | 380.95 | 406.62 | pass |
| 4 | 64 | 44.31 | 1391.11 | 1570.24 | 1619.88 | fail |
| 8 | 16 | 86.07 | 181.66 | 194.13 | 218.98 | pass |
| 8 | 64 | 84.39 | 720.46 | 836.96 | 873.02 | fail |
| 8 | 128 | 79.41 | 1440.56 | 1739.41 | 1784.59 | fail |

Scaling efficiency relative to one worker:

| Workers | Latency/error capacity | Relative to 1 worker | Ideal linear |
| ---: | ---: | ---: | ---: |
| 1 | 13.54 | 1.00× | 1.00× |
| 2 | 21.92 | 1.62× | 2.00× |
| 4 | 45.66 | 3.37× | 4.00× |
| 8 | 86.07 | 6.36× | 8.00× |

The 8-worker system reaches **6.36× the 1-worker latency/error capacity**, or 80% of linear scaling. The full-SLO sustained scaling is **7 → 12 → 24 → 45 requests/s**, or **6.43×**. The remaining gap comes from the fixed gateway, the load client, and the concurrency sweep granularity. The p95 knee for 8 workers is between concurrency 16 and 64. The collapse point is at concurrency 128, where p95 exceeds 1.7 s.

### Hash routing versus least-loaded routing

The consistent-hash ring distributes a fixed image set unevenly. The hottest worker share is shown at the highest passing latency/error level.

| Workers | Routing | Latency/error requests/s | Hottest worker share | Average share | p95 at next level |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | hash | 17.52 | 63.4% | 50.0% | 1373.7 ms at c=16 |
| 2 | least-loaded | 21.92 | 50.0% | 50.0% | 719.0 ms at c=16 |
| 4 | hash | 23.49 | 38.4% | 25.0% | 1438.5 ms at c=16 |
| 4 | least-loaded | 45.66 | 29.6% | 25.0% | 1570.2 ms at c=64 |
| 8 | hash | 31.44 | 22.0% | 12.5% | 1074.3 ms at c=16 |
| 8 | least-loaded | 86.07 | 13.0% | 12.5% | 837.0 ms at c=64 |

At 8 workers, least-loaded routing raises latency/error capacity by **2.7×** and reduces p95 at concurrency 16 from **1074 ms to 194 ms**. The result shows that the scaling limit was routing imbalance, not aggregate worker CPU. The CPU-limited full-SLO rate is lower; see the hold section.

### Model accuracy and throughput tradeoff

All rows use 2 workers, least-loaded routing, a cold cache, and the same 500-image workload. The requests/s values are the highest passing latency/error level for that model. CPU was not reduced for the model screen, so these rates are not full-SLO claims.

| Model | Top-1 | Top-5 | Latency-only requests/s | Concurrency | p95 ms | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShuffleNetV2-10 | 72.2% | 90.6% | 201.36 | 4 | 22.24 | 19.64 |
| MobileNetV2-7 | 71.8% | 90.8% | 68.01 | 16 | 248.21 | 222.66 |
| ResNet18-v2-7 | 71.6% | 90.6% | 47.02 | 16 | 338.83 | 330.13 |
| ResNet34-v2-7 | 76.2% | 92.2% | 27.58 | 4 | 147.89 | 144.50 |
| ResNet50-v2-7 | 78.6% | 94.2% | 21.92 | 4 | 185.17 | 181.90 |

ShuffleNetV2-10 is **9.2× faster** than ResNet50-v2-7 with **6.4 points lower top-1** on this subset. MobileNetV2-7 is **3.1× faster** with **6.8 points lower top-1**. This is the expected accuracy/latency tradeoff and gives a usable Pareto frontier for model selection.

### Transport comparison

The same 2-worker topology and repeated 20-image workload are used for both transports. The JSON client is Python, so its serialization cost is included. This comparison is not a server-only measurement.

| Transport | Concurrency | Requests/s | p50 ms | p95 ms | p99 ms | Client CPU | SLO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| binary | 4 | 21.41 | 184.84 | 196.81 | 203.65 | 2.93% | pass |
| JSON | 4 | 12.98 | 303.92 | 345.38 | 457.26 | 101.23% | pass |
| binary | 16 | 23.05 | 682.66 | 737.36 | 776.06 | 2.90% | fail |
| JSON | 16 | 11.11 | 1589.68 | 1880.46 | 1915.70 | 87.09% | fail |

Binary transport is **1.65× faster at concurrency 4** and **2.08× faster at concurrency 16**. It also cuts client CPU by about 34× at concurrency 4. The JSON client reaches 101% CPU, so the JSON result is partly client-bound. The server-side JSON parse remains a real cost, but the exact split is not measured.

### Five-minute holds and CPU

Each hold uses a cold cache, unique 500-image workload, least-loaded routing, and 8 workers unless labeled otherwise. CPU is the per-worker `us + sy` value from 1-second `vmstat` samples. The table reports the worst worker.

Full-SLO sustained results (latency, errors, and CPU):

| Workers | Offered rate | Requests | Errors | p95 | p99 | Cache hits | Worst worker mean CPU | Worst worker p95 CPU | Max sample | CPU SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 7/s | 2,100 | 0 | 102.1 ms | 102.8 ms | 0 | 67.5% | 69% | 77% | pass |
| 2 | 12/s | 3,600 | 0 | 103.6 ms | 105.2 ms | 0 | 58.6% | 60% | 67% | pass |
| 4 | 24/s | 7,200 | 0 | 102.6 ms | 104.7 ms | 0 | 58.2% | 60% | 65% | pass |
| 8 | 45/s | 13,500 | 0 | 104.0 ms | 105.8 ms | 0 | 55.8% | 60% | 67% | pass |

The 8-worker 45 requests/s run is the primary full-SLO result. It has no cache hits and perfect balance (1,685–1,690 requests per worker). Zero errors in 13,500 requests gives a 95% upper bound of about 0.022% error rate, below the 0.1% limit. The **6.43× scaling** from 1 to 8 workers is close to the 6.36× latency-only scaling.

CPU boundary at 8 workers:

| Offered rate | Achieved | p95 | Worst worker mean CPU | Worst worker p95 CPU | Max sample | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 45/s | 44.99/s | 104.0 ms | 55.8% | 60% | 67% | full SLO pass |
| 50/s | 49.98/s | 108.8 ms | 67.0% | 80% | 82% | marginal CPU pass at p95 |
| 60/s | 59.98/s | 107.8 ms | 86.3% | 95% | 95% | CPU fail |
| 84/s | 83.97/s | 176.6 ms | 94.8% | 100% | 101% | CPU fail |
| 86/s (mixed) | 85.96/s | 178.7 ms | 95.5% | 100% | 101% | CPU fail |

The CPU knee is between 50 and 60 requests/s at 8 workers. The 45 requests/s point has clear CPU headroom. The 84 requests/s unique run is a latency-only result; do not call it a full-SLO result.

### Worker failure and consistency

The failure test used 8 workers, least-loaded routing, 40 requests/s, 60 seconds, and a repeated 20-image workload. The controller killed `worker-1` 20 seconds after the load started.

| Metric | Result |
| --- | ---: |
| Requests | 2,400 |
| Failures | 0 |
| Error rate | 0% |
| p50 | 101.30 ms |
| p95 | 113.97 ms |
| p99 | 175.28 ms |
| Cache hits | 21 (0.88%) |
| SLO | pass |

Consistency check over the same 2,400 responses:

| Metric | Result |
| --- | ---: |
| Labels | 20 |
| Requests per label | 120 |
| Labels served by more than one worker | 19 |
| Top-1 conflicts | 0 |
| Maximum absolute output difference | 0.0 |

The gateway retried requests that hit the dead worker. The gateway log shows repeated `no response` messages for the killed node, but the client saw no failed request. The circuit breaker for the dead node opened after the configured failure threshold. This is useful evidence for retry behavior. It is not a full failover test: the cluster had no spare worker, no health-check-driven replacement, and no network partition.

### CPU and batch behavior

The worker health endpoint reports batch metrics per run. For ResNet50, batch size rises with concurrency:

| Workers | Concurrency | Mean batch size | Requests/s | p95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 4.00 | 13.54 | 298 ms |
| 2 | 4 | 2.00 | 21.92 | 185 ms |
| 4 | 16 | 3.43 | 45.66 | 381 ms |
| 8 | 16 | 1.50 | 86.07 | 194 ms |

At the full-SLO hold of 45 requests/s, the mean batch size is 1.0. The offered rate gives about 5.6 requests/s per worker, below the per-worker batching knee. The hold therefore verifies low-latency, low-CPU capacity without relying on large batches.

The worker CPU limit is reduced from the saved `vmstat` samples. The worst worker mean is 55.8% at 45 requests/s and 86.3% at 60 requests/s. The 45 requests/s point passes the 80% limit with margin; the 60 requests/s point fails it. The 84 requests/s latency-only hold uses 94.8% CPU and is not a full-SLO result.

## Failure analysis

1. **Cache contamination in the first 2-worker run.** The runner restarted services between accuracy and load, but not between load levels. Level 1 had 60 misses and 140 hits. Levels 4, 16, and 64 reused those cached images and reported 100% hits. Those level results are invalid for capacity. The fix starts fresh services for every level and rejects any worker whose health counters are not zero. All final capacity runs report 0 cache hits.

2. **Static batch-1 models returned HTTP 500.** ShuffleNetV2-10 and MobileNetV2-7 declare a fixed batch dimension of 1. The first ShuffleNet screen failed every level because the batch processor formed batches larger than 1. The fix caps the effective batch size to the model's static dimension. The rerun passed with batch size 1. This is a real robustness fix, not a benchmark workaround.

3. **Consistent-hash imbalance limited scaling.** With 8 workers, the hottest worker received 22.0% of a 500-image set while the average was 12.5%. Its saturation caused the p95 breach at concurrency 16. Least-loaded routing removed that bottleneck and raised latency/error capacity by 2.7×. The CPU limit still caps the full-SLO rate at about 45–50 requests/s.

4. **The gateway logged every request.** The original gateway wrote two lines per request to stdout. Those logs are expensive at high request rates. The benchmark removed the per-request logs and kept errors and startup messages. The isolated effect is not measured, because routing changed at the same time. Treat the logging change as a safe cleanup, not a proven speedup.

5. **The JSON client is CPU-bound.** The Python JSON client reached 101% CPU at concurrency 4. The binary transport result is the better capacity signal. A compiled client or a different serializer could narrow the transport gap.

6. **The failover retry path wastes work.** Least-loaded routing can still select the dead worker before its circuit breaker opens. Requests then fail at the gateway and retry on another worker. The client sees success, but the retry adds latency and duplicate work. A health-aware load balancer or a faster failure detector would reduce this cost.

## Cost and cleanup

Budget ceiling: **$10**. Estimated spend for the new campaign: **$3.267**. The earlier 13 single-host runs cost **$0.0447**. The total is **$3.312**, leaving about **$6.69**.

The estimate uses `c6i.large` at $0.085/hour, `c6i.xlarge` at $0.17/hour, public IPv4 at $0.005/hour per instance, and 12 GB gp3 at $0.08/GB-month. For runs whose instances were still visible in AWS, runtime is launch to termination. For older runs that AWS had already aged out of `describe-instances`, runtime is the console-log end time minus the recorded start time. That method is conservative. Data transfer, tax, credits, and billing adjustments are excluded. See `docs/aws-cost-ledger.md` for the per-run table.

Cleanup check after the campaign:

| Resource | Count |
| --- | ---: |
| Active tagged instances | 0 |
| Tagged security groups | 0 |
| Tagged volumes | 0 |
| Terminated instance records still visible | 43 |

The terminated records are AWS retention records. They are not running resources. The controller deleted the temporary security groups after each run. The local SSH key stays in the git-ignored run directory.

## Production readiness

| Area | Status | Evidence or gap |
| --- | --- | --- |
| Real accuracy | PASS | 500 labeled images, five models, credible ranking |
| Binary transport | PASS | 1.65–2.08× faster than JSON in this test |
| Least-loaded routing | PASS | 2.7× latency/error capacity at 8 workers |
| Static batch-1 models | PASS | Effective batch cap added and verified |
| Worker-failure retry | PASS | 0 failed requests in the 60-second kill test |
| 5-minute full-SLO hold | PASS at 45 requests/s | 13,500 requests, 0 errors, p95 104.0 ms, CPU ≤67% |
| Latency-only hold | PASS (latency only) | 84 requests/s, p95 176.6 ms, CPU ~95% (fails CPU) |
| Private network and cleanup | PASS | SSH `/32`, private app ports, 0 active resources |
| CPU SLO | PASS at 45 requests/s | Worst worker mean 55.8%, p95 60%, max 67%; 50 requests/s is marginal and 60 requests/s fails |
| TLS and authentication | BLOCKED | There is no TLS, client authentication, or authorization |
| Rate limiting and backpressure | BLOCKED | No admission control or queue limit |
| Graceful shutdown | BLOCKED | Stop path uses process termination; no drain test |
| Observability | BLOCKED | `/health` and `/stats` exist; no metrics export, tracing, or alerting |
| Model versioning and rollback | BLOCKED | No model registry integration or rollback test |
| Container quota and isolation | OUT OF SCOPE | No container runtime in this test |
| Multi-AZ and instance replacement | OUT OF SCOPE | Single region, single subnet, no spare capacity |
| Process-pool and gateway batching | OUT OF SCOPE | Not implemented or tested in this campaign |
| Cache-enabled capacity | PARTIAL | Cache behavior was measured, but the capacity sweeps use a cold cache |
| Security review | BLOCKED | No penetration test, dependency audit, or secret review |

## Limitations

- The dataset is a 500-image subset with one image per class. It is not a full ImageNet validation run.
- The capacity sweep is closed-loop and coarse. The reported latency/error capacity is the highest measured passing level, not an interpolated maximum.
- The 8-worker latency/error capacity is measured at concurrency 16. Concurrency 64 fails at p95 837 ms. The true knee is between these levels.
- The full-SLO sustained scaling uses verified points at 7, 12, 24, and 45 requests/s. These points have CPU headroom. The exact CPU-limited maximum is between 50 and 60 requests/s at 8 workers and was not found with a finer search.
- The failover consistency test uses 20 images repeated 120 times. It proves deterministic outputs for those images, not all 500.
- The failover test has no spare worker and no automated replacement. It tests retry, not full high availability.
- The 5-minute hold is the longest sustained run. Multi-hour stability, memory growth, and file-descriptor leaks are not tested.
- The JSON transport comparison uses a Python client and includes client serialization. It does not isolate server parse cost.
- All runs use on-demand instances in one region. Spot interruption, multi-AZ latency, and cross-region behavior are not tested.
- Model and dataset licenses must be reviewed before any public or commercial use. The ImageNet subset is for research use.

## Reproduction

The harness lives in `benchmark-aws/`.

```sh
# Build the benchmark environment locally.
python3 -m venv /tmp/opencode/bench-venv
/tmp/opencode/bench-venv/bin/pip install -r benchmark-aws/requirements-benchmark.txt

# Create the 500-image fixture.
/tmp/opencode/bench-venv/bin/python benchmark-aws/autotune/prepare_data.py \
  --output benchmark-aws/data/imagenet500 --count 500

# Run one cold-cache capacity candidate.
python3 benchmark-aws/multihost_controller.py \
  --workers 8 --routing least_loaded --levels 1 4 16 64 128 \
  --requests 500 --mode unique --cache 1 --skip-accuracy \
  --model resnet50-v2-7

# Run a 5-minute full-SLO hold with CPU headroom.
python3 benchmark-aws/multihost_controller.py \
  --workers 8 --routing least_loaded --skip-accuracy --skip-levels \
  --mode unique --cache 1 --hold-rate 45 --hold-seconds 300 \
  --model resnet50-v2-7

# Latency-only boundary check at 84 requests/s.
python3 benchmark-aws/multihost_controller.py \
  --workers 8 --routing least_loaded --skip-accuracy --skip-levels \
  --mode unique --cache 1 --hold-rate 84 --hold-seconds 300 \
  --model resnet50-v2-7

# Run a worker-failure test.
python3 benchmark-aws/multihost_controller.py \
  --workers 8 --routing least_loaded --skip-accuracy --skip-levels \
  --cache 1 --failover-rate 40 --failover-seconds 60 \
  --failover-kill-after 20 --failover-kill-worker 1 \
  --model resnet50-v2-7
```

Every paid run requires `python3 benchmark-aws/controller.py preflight`, a dry-run cost estimate, verified cleanup, and a ledger update. The controller tags all resources with `temporary-inference-benchmark`.

## Evidence

| Purpose | Run ID |
| --- | --- |
| 1-worker cold capacity | `infer-e81b49f6d3f17765` |
| 2-worker hash capacity | `infer-d0b6b9aeb075e99d` |
| 4-worker hash capacity | `infer-c3758d5650859167` |
| 8-worker hash capacity | `infer-83b8dbd4d084cdc0` |
| 2-worker least-loaded capacity | `infer-7bcee360549dabf9` |
| 4-worker least-loaded capacity | `infer-3b483e9f0e17133e` |
| 8-worker least-loaded capacity | `infer-a709308362635b90` |
| 1-worker 7 requests/s full-SLO hold | `infer-c0e784095ecbb4ec` |
| 2-worker 12 requests/s full-SLO hold | `infer-4fb41cf43bfe9fc7` |
| 4-worker 24 requests/s full-SLO hold | `infer-77f5980e5018e459` |
| 8-worker 45 requests/s full-SLO hold | `infer-9b33aaad4138a3ab` |
| 8-worker 50 requests/s marginal-CPU hold | `infer-faec7003a94741a3` |
| 8-worker 60 requests/s CPU-fail hold | `infer-72587b02bbc2e3a1` |
| 8-worker 69 requests/s hold | `infer-01625ee5bd706ff0` |
| 8-worker 84 requests/s latency-only hold | `infer-fd3725f22b42726b` |
| Worker-failure test | `infer-27787d94d40229e6` |
| ShuffleNetV2-10 screen | `infer-38fce5baffb45b73` |
| MobileNetV2-7 screen | `infer-abece6d2475cf3b2` |
| ResNet18-v2-7 screen | `infer-af28a46c98b37890` |
| ResNet34-v2-7 screen | `infer-d41d6985fa110f4f` |
| Binary transport comparison | `infer-22d7935dd62ad193` |
| JSON transport comparison | `infer-89263c6af926b86d` |
| ResNet50 accuracy | `infer-5fd1ee38b8821e83` |

Sanitized summary: `benchmarks/results/2026-11-02-autotune-summary.json`.

Raw evidence stays under the git-ignored `benchmark-aws/runs/` directory. It includes per-run state, prepare logs, load summaries, request JSONL, health JSON, `vmstat`, `mpstat`, `pidstat`, and worker/gateway log tails.
