"""Workload selection for the image auto-tuning benchmark."""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("images"):
        raise ValueError("manifest has no images")
    return manifest


def select_images(manifest: dict, mode: str, count: int, seed: int = 0) -> list[dict]:
    images = manifest["images"]
    if count < 1:
        raise ValueError("count must be positive")
    if mode == "unique":
        return [images[index % len(images)] for index in range(count)]
    if mode == "repeated":
        pool = images[: min(20, len(images))]
        return [pool[index % len(pool)] for index in range(count)]
    if mode == "mixed":
        pool = images[: min(20, len(images))]
        rng = random.Random(seed)
        selected = []
        for index in range(count):
            source = pool if rng.random() < 0.8 else images
            selected.append(source[index % len(source)])
        return selected
    raise ValueError(f"unknown workload mode: {mode}")


def tensor_path(data_dir: Path, entry: dict) -> Path:
    return data_dir / entry["tensor_path"]


def tensor_bytes(data_dir: Path, entry: dict) -> bytes:
    payload = tensor_path(data_dir, entry).read_bytes()
    if not payload or len(payload) % 4 != 0:
        raise ValueError(f"invalid tensor payload: {entry['tensor_path']}")
    return payload
