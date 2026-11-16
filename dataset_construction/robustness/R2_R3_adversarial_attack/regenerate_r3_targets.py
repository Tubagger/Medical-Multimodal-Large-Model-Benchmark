#!/usr/bin/env python3
"""Regenerate the three R.3 attacks invalidated by the ISIC→skin correction."""

from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ATTACK_CODE = ROOT / "ours/utils"
sys.path.insert(0, str(ATTACK_CODE))

REQUIRED_IDS = (53, 67, 70)


class TargetLoss:
    def __call__(self, loss, *args):
        return loss


def build_models(model_root: Path, target_text: str, device: str):
    from surrogates import DYPOpenClipAttackVisionModel, DYPTransformerClipAttackVisionModel

    models = [
        DYPTransformerClipAttackVisionModel(
            str(model_root / "clip-vit-large-patch14"), target_text=target_text
        ).to(device),
        DYPOpenClipAttackVisionModel(
            model_name="ViT-B-32",
            pretrained=str(model_root / "CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"),
            target_text=target_text,
        ).to(device),
        DYPOpenClipAttackVisionModel(
            model_name="ViT-bigG-14",
            pretrained=str(model_root / "CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin"),
            target_text=target_text,
        ).to(device),
        DYPOpenClipAttackVisionModel(
            model_name="ViT-B-16",
            pretrained=str(model_root / "CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin"),
            target_text=target_text,
        ).to(device),
        DYPOpenClipAttackVisionModel(
            model_name="convnext_large_d_320",
            pretrained=str(model_root / "CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin"),
            target_text=target_text,
        ).to(device),
    ]
    return models


def generate(clean: Path, output: Path, target: str, models, device: str) -> None:
    import torch
    from PIL import Image
    from torchvision import transforms
    from SSA import SSA_CommonWeakness

    clean_image = Image.open(clean).convert("RGB")
    if clean_image.size != (224, 224):
        raise ValueError(f"Expected formal 224x224 letterbox clean input, got {clean_image.size}: {clean}")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = transform(clean_image).unsqueeze(0).to(device).float()
    attacker = SSA_CommonWeakness(
        models,
        epsilon=16 / 255,
        step_size=1 / 255,
        total_step=50,
        criterion=TargetLoss(),
    )
    for model in models:
        model.change_target_text(target)
    adversarial = attacker(image, torch.tensor([0], device=device))
    output.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(adversarial.detach().cpu().squeeze(0)).save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True,
                        help="Formal 224x224 letterbox clean-image directory")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--open-clip-site", type=Path,
                        help="Optional existing site-packages directory containing open_clip")
    args = parser.parse_args()
    if args.open_clip_site:
        site.addsitedir(str(args.open_clip_site.resolve()))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the five-surrogate SSA attack")
    labels = {int(row["id"]): row for row in json.loads(args.labels.read_text(encoding="utf-8"))}
    models = build_models(args.model_root.resolve(), "dummy", args.device)
    for sample_id in REQUIRED_IDS:
        row = labels[sample_id]
        if not row.get("attack_image_regenerated"):
            raise ValueError(f"ID {sample_id} is not marked for regeneration")
        destination = args.output_dir.resolve() / row["image"]
        if destination.exists():
            raise FileExistsError(destination)
        generate(
            args.clean_dir.resolve() / row["image"], destination,
            row["attack_target_label"], models, args.device,
        )
        print(f"generated ID {sample_id}: {row['attack_target_label']} -> {destination}")


if __name__ == "__main__":
    main()
