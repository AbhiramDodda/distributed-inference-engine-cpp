#!/usr/bin/env python3
"""Run one local auto-tuning candidate against a real worker and gateway."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from autotune import loadgen, workloads


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_url(url: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    raise RuntimeError(f"timeout waiting for {url}")


def model_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_services(args: argparse.Namespace, output: Path, phase: str):
    worker_port = free_port()
    gateway_port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "ORT_INTRA_THREADS": str(args.threads),
            "BATCH_DEADLINE_MS": str(args.deadline),
            "MAX_BATCH_SIZE": str(args.max_batch),
            "CACHE_CAPACITY": str(args.cache),
            "GATEWAY_PORT": str(gateway_port),
        }
    )
    logs = []
    worker_log = open(output / f"{phase}-worker.log", "w")
    logs.append(worker_log)
    worker = subprocess.Popen(
        [args.worker_bin, str(worker_port), "autotune", str(args.model)],
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    gateway_log = open(output / f"{phase}-gateway.log", "w")
    logs.append(gateway_log)
    gateway = subprocess.Popen(
        [args.gateway_bin, f"localhost:{worker_port}"],
        stdout=gateway_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    target = f"http://127.0.0.1:{gateway_port}"
    try:
        wait_url(f"http://127.0.0.1:{worker_port}/health")
        wait_url(f"{target}/stats")
        time.sleep(0.5)
    except Exception:
        stop_services(worker, gateway, logs)
        raise
    return worker, gateway, target, logs


def stop_services(worker, gateway, logs) -> None:
    for process in (gateway, worker):
        if process is None:
            continue
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for log in logs:
        log.close()


def run_candidate(args: argparse.Namespace) -> int:
    output = args.output
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"{output} is not empty; pass --force to replace it")
    output.mkdir(parents=True, exist_ok=True)
    manifest = workloads.load_manifest(args.data)
    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")

    summaries = []
    if args.accuracy:
        worker = gateway = None
        logs = []
        try:
            worker, gateway, target, logs = start_services(args, output, "accuracy")
            summary, rows, accuracy_result = asyncio.run(
                loadgen.run(
                    target,
                    args.data,
                    manifest,
                    "unique",
                    "binary",
                    args.accuracy_limit,
                    1,
                    None,
                    args.timeout,
                    True,
                    "accuracy",
                )
            )
            summary["name"] = "accuracy"
            summaries.append(summary)
            (output / "accuracy-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (output / "accuracy-requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            if accuracy_result is not None:
                (output / "accuracy.json").write_text(json.dumps(accuracy_result, indent=2) + "\n")
                print(json.dumps(accuracy_result), flush=True)
        finally:
            stop_services(worker, gateway, logs)

    for level in [] if args.accuracy_only else args.levels:
        worker = gateway = None
        logs = []
        try:
            worker, gateway, target, logs = start_services(args, output, f"level-{level}")
            count = (
                max(1, int(args.rate * args.duration)) if args.rate is not None else args.requests
            )
            summary, rows, _ = asyncio.run(
                loadgen.run(
                    target,
                    args.data,
                    manifest,
                    args.mode,
                    args.transport,
                    count,
                    level,
                    args.rate,
                    args.timeout,
                    False,
                    f"c{level}",
                    args.seed,
                )
            )
            summary["name"] = f"{args.transport}-{args.mode}-c{level}"
            summaries.append(summary)
            level_dir = output / f"level-{level}"
            level_dir.mkdir(exist_ok=True)
            (level_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (level_dir / "requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            print(json.dumps(summary), flush=True)
        finally:
            stop_services(worker, gateway, logs)

    run_record = {
        "model": str(model),
        "model_sha256": model_sha(model),
        "data_commit": manifest.get("dataset_commit"),
        "data_count": manifest.get("count"),
        "transport": args.transport,
        "mode": args.mode,
        "levels": args.levels,
        "requests_per_level": args.requests,
        "rate": args.rate,
        "duration": args.duration,
        "threads": args.threads,
        "deadline_ms": args.deadline,
        "max_batch": args.max_batch,
        "cache_capacity": args.cache,
        "summaries": summaries,
    }
    (output / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-bin", default="/tmp/opencode/worker_tensor")
    parser.add_argument("--gateway-bin", default="/tmp/opencode/gateway_tensor")
    parser.add_argument("--transport", choices=("binary", "json"), default="binary")
    parser.add_argument("--mode", choices=("unique", "repeated", "mixed"), default="unique")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--accuracy-limit", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--deadline", type=int, default=10)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--cache", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run_candidate(args)


if __name__ == "__main__":
    raise SystemExit(main())
