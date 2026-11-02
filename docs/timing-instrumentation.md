# Queue and batch timing

## Goal

Record where cache-miss requests spend time before choosing an optimization. This is instrumentation, not an optimization or a claim that a bottleneck has been proven.

## Response fields

The worker includes the following fields in each `/infer` response. The gateway forwards that response.

- `queue_wait_us`: microseconds from insertion into the batch queue until the batch begins processing. It excludes time spent acquiring the queue mutex before insertion and excludes later response handling. A worker busy processing an earlier batch can exceed the configured 20 ms batching deadline.
- `batch_execution_us`: wall-clock microseconds spent in the backend's `batchPredict` call for the **entire batch**. This includes backend locking, tensor preparation, runtime inference and output handling. It is not pure model compute time or an individually measured request duration.
- `batch_size`: number of requests in that batch. Every member of the same batch gets the same batch execution duration and batch size.

For cache hits these three fields are JSON `null`: the request did not queue for a batch or run inference. Durations use `steady_clock` and are truncated to integer microseconds. Existing `inference_time_us` remains available with its existing cache-hit and cache-miss meanings; it should not be compared directly across those paths.

## Benchmark output

`benchmark-aws/load.py` saves the fields in each JSONL request record. Each level's `worker_timings_us` contains the number of available samples and request-weighted p50 and p95. Cache hits and responses from an uninstrumented server have no timing samples and are not counted as zero. Batch time is repeated for each request in a batch; do not sum it as if each request ran a separate batch. Small samples yield uncertain percentiles. The client and server may still contend for the same CPU.

## Local verification

Run the Python checks with `python3 -m unittest discover -s tests -v`. The real-worker integration test is skipped unless `BENCH_WORKER_BIN` and `BENCH_MODEL` point to a locally built worker and model. `ctest --test-dir <build-dir> --output-on-failure` exercises queue timing and response pairing. A real worker test checks miss and hit fields, legacy field presence, and equal output.

## Deployment boundary

The AWS runner snapshots a patch for the instrumented batch/timing files, the capacity-aware ORT thread selection, and the worker/gateway TCP_NODELAY settings against the pinned upstream revision before creating resources. Its default `instrumented` mode checks patch and source hashes and requires timing samples; `--source baseline` uses the original upstream files without a patch. Each run saves the exact patch, source manifest, and benchmark scripts locally, and returns the patch and manifest in its result archive. The pinned baseline must be an ancestor of local HEAD. Changes to other files under `src/` or `include/` are rejected rather than silently omitted. No AWS resources were created to validate this delivery change.
