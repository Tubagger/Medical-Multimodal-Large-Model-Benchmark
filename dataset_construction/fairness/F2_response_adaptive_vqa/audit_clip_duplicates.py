#!/usr/bin/env python3
"""Find semantically similar F.2 images with OpenCLIP image embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


def main() -> None:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=repository / "data/fairness/F2_response_adaptive_vqa/images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "run_state/data_checks/F2/clip_near_duplicate_audit.json",
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths = sorted(args.image_dir.glob("img_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device
    )
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            batch = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in batch_paths])
            features = model.encode_image(batch.to(args.device))
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu())

    matrix = torch.cat(embeddings).numpy().astype(np.float32)
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -1.0)
    nearest = similarities.max(axis=1)
    pairs = []
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            pairs.append(
                {
                    "similarity": float(similarities[left, right]),
                    "left": int(paths[left].stem.split("_")[-1]),
                    "right": int(paths[right].stem.split("_")[-1]),
                }
            )
    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    payload = {
        "model": args.model,
        "pretrained": args.pretrained,
        "count": len(paths),
        "nearest_min": float(nearest.min()),
        "nearest_mean": float(nearest.mean()),
        "nearest_max": float(nearest.max()),
        "pairs_ge_0_98": sum(item["similarity"] >= 0.98 for item in pairs),
        "pairs_ge_0_95": sum(item["similarity"] >= 0.95 for item in pairs),
        "top_unique_pairs": pairs[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

