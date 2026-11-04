# Overload protection and admission control

## What changed

The gateway now supports an optional global in-flight limit:

```sh
ADMISSION_MAX_INFLIGHT=7
```

- `0` is the default and keeps the previous behavior.
- When the limit is reached, the gateway returns HTTP `503` with `Retry-After: 1` and a JSON body.
- The limit covers `/infer` and `/infer_tensor`.
- Health and `/stats` are never rejected.
- `/stats` reports:

```json
"admission": {
  "max_in_flight": 7,
  "in_flight": 0,
  "accepted": 264,
  "rejected": 236
}
```

The gateway checks admission before parsing or routing the inference request. A rejected request does not reach a worker.

## Why

The benchmark showed a throughput plateau and p95 collapse at high concurrency. Without admission, clients queue for seconds. The service remains “successful” only because every request eventually completes; the latency is not usable.

Admission control makes the tradeoff explicit:

- protect accepted requests,
- reject excess with a retryable status,
- keep the queue bounded.

This is a static limit, not an autoscaler.

## Local experiment

Hardware:

- AMD Ryzen 7 5800H, 16 logical CPUs.
- 2 local worker processes.
- 2 ORT intra-op threads per worker.
- ResNet50-v2-7.
- 500 unique images, cold cache (`CACHE_CAPACITY=1`).
- Binary float32 transport.
- Open-loop offered rate, 10-second duration, 128 connection limit.

Commands:

```sh
BENCH_WORKER_BIN=/tmp/opencode/cmake-build-hardening/worker_node \
BENCH_GATEWAY_BIN=/tmp/opencode/gateway_admission \
BENCH_MODEL=$PWD/models/resnet50-v2-7.onnx \
/tmp/opencode/bench-venv/bin/python benchmark-aws/run_autotune.py \
  --model models/resnet50-v2-7.onnx \
  --data benchmark-aws/data/imagenet500 \
  --output /tmp/opencode/overload-ad7-50-rep1 \
  --worker-bin /tmp/opencode/cmake-build-hardening/worker_node \
  --gateway-bin /tmp/opencode/gateway_admission \
  --workers 2 --admission-max 7 \
  --levels 1 --rate 50 --duration 10 --rate-concurrency 128 \
  --mode unique --cache 1 --threads 2 --force
```

## Results

At 25 requests/s, admission is inactive in practice:

| Condition | Accepted | 503 | Accepted p95 | Accepted p99 |
| --- | ---: | ---: | ---: | ---: |
| Baseline, no admission | 250 / 250 | 0 | 279.6 ms | 467.1 ms |
| Admission 7 | 250 / 250 | 0 | 156.0 ms | 197.6 ms |

At 50 requests/s, twice the 25 requests/s operating point:

| Condition | Accepted | 503 | Accepted throughput | Accepted p95 | Accepted p99 | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline rep 1 | 500 / 500 | 0 | 31.1 /s | 6,159 ms | 6,631 ms | 16.1 s |
| Baseline rep 2 | 500 / 500 | 0 | 31.4 /s | 6,196 ms | 6,471 ms | 15.9 s |
| Baseline rep 3 | 500 / 500 | 0 | 26.3 /s | 9,005 ms | 9,382 ms | 19.0 s |
| Admission 7 rep 1 | 264 / 500 | 236 | 25.7 /s | 428.9 ms | 449.7 ms | 10.3 s |
| Admission 7 rep 2 | 214 / 500 | 286 | 20.8 /s | 427.5 ms | 450.1 ms | 10.3 s |
| Admission 7 rep 3 | 258 / 500 | 242 | 25.1 /s | 373.4 ms | 383.3 ms | 10.3 s |

All admission-run failures were controlled `503` responses. There were no other 5xx responses.

The baseline accepted all 500 requests but took 16–19 seconds to drain the offered 10-second workload. Its p95 was 6.2–9.0 seconds. That is queue collapse, even though the client saw HTTP 200 for every request.

## Interpretation

- Admission control does not increase capacity.
- It prevents the accepted requests from waiting behind an unbounded queue.
- At 2× the 25 requests/s operating point, the limit of 7 sheds about 48–57% of requests and keeps accepted p95 at 373–429 ms.
- The shed requests are explicit `503` responses with `Retry-After`, not silent timeouts.
- The accepted throughput is about 21–26 requests/s, close to the 25 requests/s operating point.
- A real client must honor `Retry-After` and retry with jitter. The load generator does not retry, so the 503 fraction is a property of the offered concurrency, not a client policy.

## Limits

- Local host, not AWS `c6i.large`.
- Two workers, not 1/2/4/8.
- One 10-second run per condition, with three repeats for the key 50 requests/s admission case.
- The admission limit is static and manually selected. A production control plane would derive it from model, worker count, and SLO.
- The load generator is open-loop but does not implement retry/backoff.
- No TLS, authentication, per-user quota, or public deployment.
- The benchmark is not a production availability claim.

## Artifacts

- Sanitized summary: `benchmarks/results/2026-11-04-overload-protection.json`.
- Raw runs remain under the ignored `/tmp/opencode/overload-*` directories.
- Test: `tests/test_admission_control.py`.

## Rollback

Remove `ADMISSION_MAX_INFLIGHT` or set it to `0`. The previous gateway behavior returns.
