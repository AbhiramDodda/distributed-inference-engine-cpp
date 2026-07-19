#!/usr/bin/env python3
"""Ad-hoc probes used during the tail-latency / adaptive-batching investigation.

This is the raw investigation tool behind ANALYSIS.md (both are gitignored). It is
deliberately more flexible than tail_latency_sweep.py — it can:

  * target a single worker OR the gateway            (--target)
  * use a NEW TCP connection per request (default)   -> mirrors benchmark.py, and
    reproduces the connection-churn regime
  * use a keep-alive Session with a pooled connection (--keepalive) -> isolates
    server-side behavior (this is the transport that cleanly exposed the 5s
    keep-alive thread-pool starvation tail)
  * sample TIME_WAIT socket counts during a run       (--time-wait)
  * report per-worker batch occupancy (avg batch size, full%) by diffing
    /health counters across each level

Every request carries a UNIQUE input (global counter -> distinct float vector) so
it misses the cache and exercises the real inference + batching path.

Examples
--------
# calibrate steady-state at one concurrency
python3 analysis_probes.py --target http://localhost:8001 --concurrency 32 \
    --requests-per-level 300

# single worker, new-connection transport, batch sweep (repro of the pool-15 wall)
python3 analysis_probes.py --target http://localhost:8001 \
    --workers http://localhost:8001 --concurrency 8 16 32 48 64 96 128

# single worker, KEEP-ALIVE transport (isolates thread-pool starvation tail)
python3 analysis_probes.py --target http://localhost:8001 \
    --workers http://localhost:8001 --keepalive \
    --concurrency 8 16 32 64 128 256

# full cluster via gateway, keep-alive
python3 analysis_probes.py --keepalive --concurrency 16 32 64 96 128 192 \
    --requests-per-level 2000
"""

import argparse
import itertools
import subprocess
import threading
import time

import requests
from requests.adapters import HTTPAdapter

# Global monotonic id -> unique input per request across the whole run.
_uid = itertools.count()


def make_session(pool):
    s = requests.Session()
    a = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
    s.mount("http://", a)
    return s


def batch_counters(worker_urls):
    """(total_batches, full_batches) summed across the given workers."""
    tot = full = 0
    for w in worker_urls:
        try:
            bp = requests.get(f"{w}/health", timeout=5).json()["batch_processor"]
            tot += bp["total_batches"]
            full += bp["full_batches"]
        except requests.RequestException:
            pass
    return tot, full


def time_wait_count():
    try:
        out = subprocess.run(["bash", "-c", "ss -tan state time-wait | wc -l"],
                             capture_output=True, text=True).stdout.strip()
        return out
    except Exception:
        return "?"


def run_level(target, workers, conc, n, timeout_s, keepalive, sample_tw):
    infer_url = f"{target}/infer"
    sess = make_session(conc) if keepalive else None  # shared keep-alive pool
    b0 = batch_counters(workers)

    lat = []
    lock = threading.Lock()
    remaining = [n]
    fails = [0]

    def claim():
        with lock:
            if remaining[0] <= 0:
                return False
            remaining[0] -= 1
            return True

    def one():
        k = next(_uid)
        x = float(k % 100000)
        payload = {"request_id": f"p_{k}", "input_data": [x, x * 0.5 + 0.5, x * 0.25 + 0.25]}
        start = time.perf_counter()
        try:
            client = sess if keepalive else requests  # requests.post = new conn each call
            r = client.post(infer_url, json=payload, timeout=timeout_s)
            return (time.perf_counter() - start) * 1000.0, r.status_code == 200
        except requests.RequestException:
            return (time.perf_counter() - start) * 1000.0, False

    def worker():
        while claim():
            l, ok = one()
            with lock:
                lat.append(l)
                if not ok:
                    fails[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    lat.sort()
    b1 = batch_counters(workers)
    dtot = b1[0] - b0[0]
    dfull = b1[1] - b0[1]
    avg_batch = (n / dtot) if dtot else 0.0
    full_pct = (100.0 * dfull / dtot) if dtot else 0.0

    def p(q):
        return lat[min(len(lat) - 1, int(len(lat) * q))] if lat else 0.0

    tw = f" TW={time_wait_count()}" if sample_tw else ""
    print(f"conc={conc:>4} tput={len(lat) / wall:>7.1f}/s p50={p(.5):>7.0f} "
          f"p90={p(.9):>7.0f} p99={p(.99):>7.0f} p999={p(.999):>8.0f} "
          f"avgBatch={avg_batch:>5.1f} full%={full_pct:>3.0f} fails={fails[0]:>4}{tw}")


def main():
    ap = argparse.ArgumentParser(description="Ad-hoc tail-latency / batching probes")
    ap.add_argument("--target", default="http://localhost:8000",
                    help="Endpoint to load (a single worker isolates the batcher)")
    ap.add_argument("--workers", nargs="+",
                    default=["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"],
                    help="Worker URLs to pull batch stats from")
    ap.add_argument("--concurrency", type=int, nargs="+",
                    default=[8, 16, 32, 48, 64, 96, 128])
    ap.add_argument("--requests-per-level", type=int, default=1500)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--keepalive", action="store_true",
                    help="Reuse a pooled keep-alive Session (isolates server behavior "
                         "from client-side TCP connection churn)")
    ap.add_argument("--time-wait", action="store_true",
                    help="Print the TIME_WAIT socket count after each level")
    ap.add_argument("--warmup", type=int, default=50,
                    help="Warmup requests at conc 8 before measuring (CUDA cold start)")
    args = ap.parse_args()

    transport = "keep-alive Session" if args.keepalive else "new connection per request"
    print(f"target={args.target}  transport={transport}  "
          f"req/level={args.requests_per_level}")
    print("-" * 100)

    if args.warmup > 0:
        run_level(args.target, args.workers, 8, args.warmup, args.timeout,
                  args.keepalive, False)
        print("(warmup done)")

    for c in args.concurrency:
        run_level(args.target, args.workers, c, args.requests_per_level,
                  args.timeout, args.keepalive, args.time_wait)


if __name__ == "__main__":
    main()
