"""Summaries, SLO capacity, and simple cost estimates for benchmark runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summaries_from_directory(directory: Path) -> list[dict]:
    return [
        load_summary(path) for path in sorted(directory.glob("*/summary.json")) if path.is_file()
    ]


def best_within_slo(summaries: list[dict]) -> dict | None:
    candidates = [item for item in summaries if item.get("slo", {}).get("within_slo")]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.get("successful_requests_per_second", 0))


def cost_per_million(requests_per_second: float, hourly_cost: float) -> float | None:
    if requests_per_second <= 0:
        return None
    return round(hourly_cost / (requests_per_second * 3600) * 1_000_000, 4)


def rank(summaries: list[dict]) -> list[dict]:
    return sorted(
        summaries,
        key=lambda item: (
            not item.get("slo", {}).get("within_slo", False),
            -item.get("successful_requests_per_second", 0),
            item.get("latency_ms", {}).get("p95") or float("inf"),
        ),
    )


def pareto(summaries: list[dict]) -> list[dict]:
    ordered = sorted(
        summaries,
        key=lambda item: item.get("successful_requests_per_second", 0),
        reverse=True,
    )
    frontier = []
    best_p95 = float("inf")
    for item in ordered:
        p95 = item.get("latency_ms", {}).get("p95")
        if p95 is None:
            continue
        if p95 < best_p95:
            frontier.append(item)
            best_p95 = p95
    return frontier


def markdown_table(summaries: list[dict]) -> str:
    lines = [
        "| Candidate | Requests/s | p50 ms | p95 ms | p99 ms | Errors | Cache hit | Within SLO |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in rank(summaries):
        latency = item.get("latency_ms", {})
        lines.append(
            "| {name} | {rps} | {p50} | {p95} | {p99} | {errors} | {cache} | {slo} |".format(
                name=item.get("name", item.get("mode", "?")),
                rps=item.get("successful_requests_per_second", 0),
                p50=latency.get("p50"),
                p95=latency.get("p95"),
                p99=latency.get("p99"),
                errors=item.get("error_rate", 0),
                cache=item.get("cache_hit_rate", 0),
                slo="yes" if item.get("slo", {}).get("within_slo") else "no",
            )
        )
    return "\n".join(lines)


def consistency(rows: list[dict]) -> dict:
    """Compare outputs for repeated requests to the same labeled image."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("success") and row.get("output"):
            groups[row["label_index"]].append(row)

    conflicts = 0
    multi_node = 0
    max_abs_diff = 0.0
    samples = 0
    for items in groups.values():
        samples += len(items)
        outputs = [item["output"] for item in items]
        top1 = {max(range(len(output)), key=output.__getitem__) for output in outputs}
        if len(top1) > 1:
            conflicts += 1
        reference = outputs[0]
        for output in outputs[1:]:
            for left, right in zip(reference, output):
                max_abs_diff = max(max_abs_diff, abs(left - right))
        nodes = {item.get("node_id") for item in items if item.get("node_id")}
        if len(nodes) > 1:
            multi_node += 1

    return {
        "samples": samples,
        "labels": len(groups),
        "top1_conflicts": conflicts,
        "multi_node_labels": multi_node,
        "max_abs_diff": round(max_abs_diff, 8),
    }
