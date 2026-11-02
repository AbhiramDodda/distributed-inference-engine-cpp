#!/usr/bin/env python3
"""Small closed-loop synthetic load generator using only the standard library."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def send_one(base: str, number: int, repeated: bool) -> dict:
    value = 1.0 if repeated else float(number + 100)
    request_id = f"bench-{number}"
    payload = json.dumps({"request_id": request_id, "input_data": [value, value / 2, value / 4]})
    request = urllib.request.Request(
        base + "/infer", payload.encode(), {"Content-Type": "application/json"}, method="POST"
    )
    started = time.monotonic()
    result = {
        "request_id": request_id,
        "success": False,
        "status": None,
        "cached": None,
        "queue_wait_us": None,
        "batch_execution_us": None,
        "batch_size": None,
    }
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result["status"] = response.status
            body = json.load(response)
        if (
            not isinstance(body, dict)
            or body.get("request_id") != request_id
            or not isinstance(body.get("output_data"), list)
            or not body["output_data"]
        ):
            result["error"] = "invalid_response"
        else:
            result["success"] = True
            result["cached"] = body.get("cached")
            for field in ("queue_wait_us", "batch_execution_us", "batch_size"):
                value = body.get(field)
                minimum = 1 if field == "batch_size" else 0
                if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
                    result[field] = value
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        result["error"] = "http_error"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError) as exc:
        result["error"] = type(exc).__name__
    result["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
    return result


def missing_timings(row: dict) -> bool:
    """A successful cache miss needs all three instrumentation fields."""
    if not row["success"]:
        return False
    if row["cached"] is True:
        return False
    return row["cached"] is not False or any(
        row.get(field) is None for field in ("queue_wait_us", "batch_execution_us", "batch_size")
    )


def run_level(
    base: str, count: int, concurrency: int, repeated: bool, offset: int
) -> tuple[dict, list]:
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda n: send_one(base, n, repeated), range(offset, offset + count)))
    elapsed = time.monotonic() - start
    successful = [row["latency_ms"] for row in rows if row["success"]]
    timing_summaries = {}
    for field in ("queue_wait_us", "batch_execution_us"):
        values = [row[field] for row in rows if row["success"] and row[field] is not None]
        timing_summaries[field] = {
            "samples": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }
    summary = {
        "workload": "repeated" if repeated else "unique",
        "concurrency": concurrency,
        "requests": len(rows),
        "successes": len(successful),
        "failures": len(rows) - len(successful),
        "wall_seconds": round(elapsed, 3),
        "successful_requests_per_second": len(successful) / elapsed,
        "successful_latency_ms": {f"p{p}": percentile(successful, p / 100) for p in (50, 95, 99)},
        "cache_hits": sum(row["cached"] is True for row in rows),
        "missing_worker_timings": sum(missing_timings(row) for row in rows),
        "worker_timings_us": timing_summaries,
    }
    return summary, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-timings", action="store_true")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    offset = 0
    with (output / "requests.jsonl").open("w", encoding="utf-8") as raw:
        for repeated in (True, False):
            for concurrency in (1, 2, 4):
                summary, rows = run_level(
                    "http://127.0.0.1:8000", 60, concurrency, repeated, offset
                )
                offset += 60
                summaries.append(summary)
                for row in rows:
                    raw.write(
                        json.dumps(
                            {"workload": summary["workload"], "concurrency": concurrency, **row}
                        )
                        + "\n"
                    )
                print(json.dumps(summary), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    successful = all(item["successes"] > 0 and item["failures"] == 0 for item in summaries)
    if args.require_timings:
        complete = all(item["missing_worker_timings"] == 0 for item in summaries)
        has_samples = any(
            item["worker_timings_us"]["queue_wait_us"]["samples"] > 0 for item in summaries
        )
        if not complete or not has_samples:
            print("ERROR: required worker timing measurements are missing", flush=True)
            return 1
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
