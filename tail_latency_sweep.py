#!/usr/bin/env python3
"""Tail-latency load sweep for the distributed inference engine.

Unlike benchmark.py (which reuses 10 inputs and therefore measures the cache
path after warmup), this harness sends a UNIQUE input on every request so each
one misses the cache and exercises the real inference + adaptive-batching path.

For a list of concurrency levels it runs a closed-loop load test, records
per-request end-to-end latency, and reports throughput and the latency tail
(p50/p90/p99/p99.9) at each level. It also diffs each worker's batch-processor
counters across the level to attribute the tail: as load rises you should see
average batch size climb and batches shift from deadline-flushed (partial) to
full, which is the adaptive batching window widening under pressure.

Finally it locates the saturation knee -- the level past which throughput stops
climbing but the tail keeps growing -- and writes the raw series to JSON/CSV for
charting.
"""

import argparse
import csv
import itertools
import json
import statistics
import threading
import time
from collections import defaultdict

import requests


# Global, monotonically increasing id so every request across the whole sweep
# gets a distinct input vector -> guaranteed cache miss -> real inference path.
_uid_counter = itertools.count()
_uid_lock = threading.Lock()


def next_uid():
    with _uid_lock:
        return next(_uid_counter)


def make_payload():
    n = next_uid()
    # Three floats, unique per request. Spread across a wide range so no two
    # requests collide and the routing hash distributes them across workers.
    x = float(n % 100000)
    return {
        "request_id": f"sweep_{n}",
        "input_data": [x * 1.0, x * 0.5 + 0.5, x * 0.25 + 0.25],
    }


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def get_batch_counters(worker_urls):
    """Snapshot per-worker batch-processor counters for later diffing."""
    snap = {}
    for url in worker_urls:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                bp = r.json().get("batch_processor", {})
                snap[url] = {
                    "total_batches": bp.get("total_batches", 0),
                    "full_batches": bp.get("full_batches", 0),
                    "timeout_batches": bp.get("timeout_batches", 0),
                    "avg_batch_size": bp.get("avg_batch_size", 0.0),
                }
        except requests.RequestException:
            pass
    return snap


def diff_batch_counters(before, after):
    """Aggregate the batch counts formed *during* a level across all workers."""
    d_total = d_full = d_deadline = 0
    for url, aft in after.items():
        bef = before.get(url, {"total_batches": 0, "full_batches": 0, "timeout_batches": 0})
        d_total += aft["total_batches"] - bef["total_batches"]
        d_full += aft["full_batches"] - bef["full_batches"]
        d_deadline += aft["timeout_batches"] - bef["timeout_batches"]
    return d_total, d_full, d_deadline


def run_level(gateway_url, concurrency, requests_per_level, timeout_s):
    """Closed-loop: `concurrency` threads pull from a shared budget of requests."""
    latencies = []
    errors = defaultdict(int)
    successes = 0
    lock = threading.Lock()
    remaining = [requests_per_level]

    def claim():
        with lock:
            if remaining[0] <= 0:
                return False
            remaining[0] -= 1
            return True

    def worker():
        nonlocal successes
        while claim():
            payload = make_payload()
            start = time.perf_counter()
            try:
                resp = requests.post(f"{gateway_url}/infer", json=payload, timeout=timeout_s)
                elapsed = (time.perf_counter() - start) * 1000.0
                ok = resp.status_code == 200
                with lock:
                    latencies.append(elapsed)
                    if ok:
                        successes += 1
                    else:
                        errors[f"http_{resp.status_code}"] += 1
            except requests.RequestException as e:
                elapsed = (time.perf_counter() - start) * 1000.0
                with lock:
                    latencies.append(elapsed)
                    errors[type(e).__name__] += 1

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start

    latencies.sort()
    total = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": total,
        "successes": successes,
        "failures": total - successes,
        "wall_s": wall,
        "throughput": successes / wall if wall > 0 else 0.0,
        "mean": statistics.mean(latencies) if latencies else 0.0,
        "p50": pct(latencies, 0.50),
        "p90": pct(latencies, 0.90),
        "p99": pct(latencies, 0.99),
        "p999": pct(latencies, 0.999),
        "max": latencies[-1] if latencies else 0.0,
        "errors": dict(errors),
    }


def find_knee(levels):
    """Knee = first level where throughput gains <5% over the prior level while
    p99 grows >20%. Returns the concurrency at the knee, or None."""
    for prev, cur in zip(levels, levels[1:]):
        if prev["throughput"] <= 0:
            continue
        tput_gain = (cur["throughput"] - prev["throughput"]) / prev["throughput"]
        p99_growth = (cur["p99"] - prev["p99"]) / prev["p99"] if prev["p99"] > 0 else 0
        if tput_gain < 0.05 and p99_growth > 0.20:
            return cur["concurrency"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Tail-latency load sweep")
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--target", default=None,
                    help="Endpoint to load. Defaults to --gateway. Point at a single "
                         "worker (e.g. http://localhost:8001) to isolate the batcher: the "
                         "multi-worker gateway path saturates the load generator before any "
                         "one worker's batch queue fills.")
    ap.add_argument("--workers", nargs="+",
                    default=["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"],
                    help="Worker URLs to pull batch stats from. When isolating a single "
                         "worker, pass just that worker here too.")
    ap.add_argument("--concurrency", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    help="Concurrency levels to sweep")
    ap.add_argument("--requests-per-level", type=int, default=5000,
                    help="Requests sent at each level (>=5000 for a meaningful p99.9)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout (s)")
    ap.add_argument("--json-out", default="sweep_results.json")
    ap.add_argument("--csv-out", default="sweep_results.csv")
    args = ap.parse_args()
    target = args.target or args.gateway

    print("Tail-latency load sweep (unique inputs -> cache-miss / inference path)")
    print(f"  Target:             {target}")
    print(f"  Concurrency levels: {args.concurrency}")
    print(f"  Requests / level:   {args.requests_per_level}")
    print()
    header = (f"{'conc':>5} {'tput/s':>9} {'mean':>8} {'p50':>8} {'p90':>8} "
              f"{'p99':>9} {'p99.9':>9} {'max':>9} {'avgBatch':>9} {'full%':>6} {'fail':>5}")
    print(header)
    print("-" * len(header))

    levels = []
    for c in args.concurrency:
        before = get_batch_counters(args.workers)
        res = run_level(target, c, args.requests_per_level, args.timeout)
        after = get_batch_counters(args.workers)

        d_total, d_full, d_deadline = diff_batch_counters(before, after)
        res["batches_formed"] = d_total
        res["full_batches"] = d_full
        res["deadline_batches"] = d_deadline
        res["full_batch_pct"] = (100.0 * d_full / d_total) if d_total else 0.0
        # Average size of batches formed during this level = requests / batches.
        res["avg_batch_size"] = (res["requests"] / d_total) if d_total else 0.0
        levels.append(res)

        print(f"{res['concurrency']:>5} {res['throughput']:>9.1f} {res['mean']:>8.1f} "
              f"{res['p50']:>8.1f} {res['p90']:>8.1f} {res['p99']:>9.1f} {res['p999']:>9.1f} "
              f"{res['max']:>9.1f} {res['avg_batch_size']:>9.1f} {res['full_batch_pct']:>5.0f}% "
              f"{res['failures']:>5}")

    print()
    knee = find_knee(levels)
    if knee is not None:
        peak = max(levels, key=lambda l: l["throughput"])
        print(f"Saturation knee at concurrency = {knee} "
              f"(peak throughput ~{peak['throughput']:.0f} req/s at concurrency {peak['concurrency']}).")
        print("Beyond the knee, added concurrency inflates the tail without raising throughput.")
    else:
        print("No clear knee within the swept range (throughput still scaling).")

    with open(args.json_out, "w") as f:
        json.dump({"levels": levels, "knee_concurrency": knee}, f, indent=2)
    with open(args.csv_out, "w", newline="") as f:
        cols = ["concurrency", "throughput", "mean", "p50", "p90", "p99", "p999",
                "max", "avg_batch_size", "full_batch_pct", "requests", "failures"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for lvl in levels:
            w.writerow(lvl)
    print(f"\nWrote {args.json_out} and {args.csv_out}")


if __name__ == "__main__":
    main()
