#!/usr/bin/env python3
"""Select distinct, within-category F.1 v2 pairs using frozen CLIP features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
import open_clip
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGES = ROOT / "data/robustness/adv_attack/images/clean"
DEFAULT_ANNOTATIONS = ROOT / "data/robustness/adv_attack/adv_untarget.json"
DEFAULT_OUTPUT = Path(__file__).with_name("pairing_manifest.json")
DEFAULT_HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
DEFAULT_WEIGHTS = DEFAULT_HF_CACHE / (
    "hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/"
    "snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/"
    "open_clip_model.safetensors"
)
EXPECTED_GROUPS = {"brain", "isic", "lung", "mura", "retina"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_answer_text(row: dict) -> str:
    options = {}
    for line in row["options"].splitlines():
        letter, value = line.split(". ", 1)
        options[letter] = value.strip().lower()
    return options[row["answer"]]


def quality_features(path: Path) -> list[float]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        aspect = rgb.width / rgb.height
        gray = np.asarray(rgb.resize((256, 256)).convert("L"), dtype=np.float32) / 255.0
    hist, _ = np.histogram(gray, bins=64, range=(0, 1), density=False)
    probabilities = hist.astype(np.float64) / hist.sum()
    entropy = -sum(value * math.log2(value) for value in probabilities if value > 0)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    sharpness = float((dx * dx).mean() + (dy * dy).mean())
    return [
        float(gray.mean()),
        float(gray.std()),
        entropy,
        math.log1p(sharpness),
        float((gray <= 0.01).mean()),
        float((gray >= 0.99).mean()),
        math.log(aspect),
    ]


def encode(paths: list[Path], weights: Path, batch_size: int) -> np.ndarray:
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(weights), device="cpu"
    )
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            batch = torch.stack(tensors)
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            output.append(features.cpu().numpy())
    return np.concatenate(output, axis=0)


def select(images_dir: Path, annotation_file: Path, weights: Path) -> dict:
    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
    rows = []
    for row in annotations:
        group = parse_answer_text(row)
        path = images_dir / row["image"]
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "source_id": int(row["id"]),
                "image": row["image"],
                "group": group,
                "sha256": sha256(path),
                "quality": quality_features(path),
                "path": path,
            }
        )
    groups = {name: [row for row in rows if row["group"] == name] for name in EXPECTED_GROUPS}
    if set(row["group"] for row in rows) != EXPECTED_GROUPS:
        raise ValueError("Unexpected source groups")
    if any(len(group) != 40 for group in groups.values()):
        raise ValueError({name: len(group) for name, group in groups.items()})
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source image bytes")

    ordered = sorted(rows, key=lambda row: row["source_id"])
    embeddings = encode([row["path"] for row in ordered], weights, batch_size=16)
    embedding_by_id = {
        row["source_id"]: embeddings[index] for index, row in enumerate(ordered)
    }

    pairs = []
    for group_name in sorted(groups):
        group = sorted(groups[group_name], key=lambda row: row["source_id"])
        quality = np.asarray([row["quality"] for row in group], dtype=np.float64)
        scale = quality.std(axis=0)
        scale[scale < 1e-8] = 1
        quality_z = (quality - quality.mean(axis=0)) / scale
        graph = nx.Graph()
        graph.add_nodes_from(range(len(group)))
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                left_embedding = embedding_by_id[group[left]["source_id"]]
                right_embedding = embedding_by_id[group[right]["source_id"]]
                cosine = float(np.dot(left_embedding, right_embedding))
                quality_distance = float(
                    np.linalg.norm(quality_z[left] - quality_z[right]) / math.sqrt(quality_z.shape[1])
                )
                cost = (1.0 - cosine) + 0.08 * quality_distance
                graph.add_edge(left, right, weight=cost)
        matching = nx.min_weight_matching(graph, weight="weight")
        if len(matching) != 20:
            raise ValueError(f"Non-perfect matching for {group_name}")
        for left_index, right_index in matching:
            left, right = sorted((group[left_index], group[right_index]), key=lambda row: row["source_id"])
            cosine = float(
                np.dot(embedding_by_id[left["source_id"]], embedding_by_id[right["source_id"]])
            )
            left_q = np.asarray(left["quality"])
            right_q = np.asarray(right["quality"])
            pairs.append(
                {
                    "group": group_name,
                    "source_a_id": left["source_id"],
                    "source_a_image": left["image"],
                    "source_a_sha256": left["sha256"],
                    "source_b_id": right["source_id"],
                    "source_b_image": right["image"],
                    "source_b_sha256": right["sha256"],
                    "clip_cosine_similarity": cosine,
                    "raw_quality_distance": float(np.linalg.norm(left_q - right_q)),
                    "source_a_quality": left["quality"],
                    "source_b_quality": right["quality"],
                }
            )
    pairs.sort(key=lambda row: (row["group"], row["source_a_id"], row["source_b_id"]))
    for index, pair in enumerate(pairs):
        pair["pair_id"] = f"pair_{index:03d}"
    used_ids = [value for pair in pairs for value in (pair["source_a_id"], pair["source_b_id"])]
    if sorted(used_ids) != list(range(200)):
        raise ValueError("Pair selection does not use each source exactly once")
    return {
        "schema_version": 1,
        "status": "candidate_pending_double_review",
        "source_annotation": str(annotation_file.relative_to(ROOT)),
        "source_annotation_sha256": sha256(annotation_file),
        "source_image_dir": str(images_dir.relative_to(ROOT)),
        "feature_model": "ViT-B-32/laion2B-s34B-b79K",
        "feature_weights_sha256": sha256(weights),
        "matching": "minimum-weight perfect matching within source group; cost=(1-cosine)+0.08*quality_distance",
        "source_count": len(rows),
        "base_pair_count": len(pairs),
        "group_pair_counts": {
            name: sum(pair["group"] == name for pair in pairs) for name in sorted(EXPECTED_GROUPS)
        },
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    # Keep the .safetensors symlink name: resolving it to the extensionless
    # blob makes open_clip select the wrong checkpoint loader.
    manifest = select(args.images_dir.resolve(), args.annotation_file.resolve(), args.weights.expanduser())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ["status", "source_count", "base_pair_count", "group_pair_counts"]}, indent=2))


if __name__ == "__main__":
    main()
