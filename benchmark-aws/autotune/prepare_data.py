#!/usr/bin/env python3
"""Build a deterministic ImageNet subset as float32 NCHW tensors."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_COMMIT = "c55ca047bd854ba01d3af6c283c034924a644db2"
DATASET_REPO = "ndb796/Small-ImageNet-Validation-Dataset-1000-Classes"
TREE_URL = f"https://api.github.com/repos/{DATASET_REPO}/git/trees/{DATASET_COMMIT}?recursive=1"
RAW_ROOT = f"https://raw.githubusercontent.com/{DATASET_REPO}/{DATASET_COMMIT}"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TENSOR_BYTES = 3 * 224 * 224 * 4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opencode-benchmark"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str) -> object:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def select_images(tree: list[dict], count: int) -> list[tuple[str, int]]:
    grouped: dict[int, list[str]] = {}
    for item in tree:
        path = item.get("path", "")
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "ILSVRC2012_img_val_subset":
            continue
        if not parts[2].lower().endswith((".jpeg", ".jpg")):
            continue
        try:
            class_index = int(parts[1])
        except ValueError:
            continue
        grouped.setdefault(class_index, []).append(path)
    selected = []
    for class_index in range(count):
        candidates = sorted(grouped.get(class_index, []))
        if not candidates:
            raise ValueError(f"no image found for class {class_index}")
        selected.append((candidates[0], class_index))
    return selected


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    scale = 256.0 / min(width, height)
    resized = image.resize(
        (max(224, round(width * scale)), max(224, round(height * scale))),
        getattr(Image, "Resampling", Image).BILINEAR,
    )
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    normalized = (array - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def write_tensor(image: Image.Image, path: Path) -> str:
    tensor = preprocess_image(image)
    payload = tensor.astype("<f4", copy=False).tobytes()
    if len(payload) != TENSOR_BYTES:
        raise ValueError(f"unexpected tensor size: {len(payload)}")
    path.write_bytes(payload)
    return sha256_bytes(payload)


def prepare_one(item: tuple[str, int], output: Path) -> dict:
    source_path, label_index = item
    image_bytes = fetch_bytes(f"{RAW_ROOT}/{source_path}")
    image_sha = sha256_bytes(image_bytes)
    image = Image.open(io.BytesIO(image_bytes))
    tensor_name = f"{label_index:04d}.bin"
    tensor_path = output / "tensors" / tensor_name
    tensor_sha = write_tensor(image, tensor_path)
    return {
        "source_path": source_path,
        "label_index": label_index,
        "image_sha256": image_sha,
        "tensor_path": f"tensors/{tensor_name}",
        "tensor_sha256": tensor_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.count < 1 or args.count > 1000:
        parser.error("--count must be between 1 and 1000")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        parser.error(f"{args.output} is not empty; pass --force to replace it")
    if args.force and args.output.exists():
        for child in sorted(args.output.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    (args.output / "tensors").mkdir(parents=True, exist_ok=True)

    tree = fetch_json(TREE_URL).get("tree", [])
    images = select_images(tree, args.count)
    labels = fetch_json(f"{RAW_ROOT}/imagenet_class_index.json")
    simple_labels = fetch_json(f"{RAW_ROOT}/imagenet.json")

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(prepare_one, item, args.output): item for item in images}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            index = record["label_index"]
            record["label"] = labels[str(index)][1]
            record["wnid"] = labels[str(index)][0]
            record["simple_label"] = simple_labels[index]
            records.append(record)
    records.sort(key=lambda item: item["label_index"])

    manifest = {
        "schema_version": 1,
        "dataset": DATASET_REPO,
        "dataset_commit": DATASET_COMMIT,
        "selection": "one image per class, sorted filename",
        "count": len(records),
        "tensor_bytes": TENSOR_BYTES,
        "preprocess": "resize-shortest-256 center-crop-224 RGB imagenet-mean-std NCHW float32",
        "images": records,
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "manifest.json")
    print(f"prepared {len(records)} images in {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
