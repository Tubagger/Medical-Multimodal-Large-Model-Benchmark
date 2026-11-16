#!/usr/bin/env python3
"""Install regenerated R.3 attacks and finalize integrity metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


REQUIRED_IDS = (53, 67, 70)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("replacements", type=Path)
    parser.add_argument("--replace-existing", action="store_true",
                        help="Replace a previously finalized three-image set and refresh metadata")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    replacements = args.replacements.resolve()
    labels = {int(row["id"]): row for row in json.loads((dataset / "labels.json").read_text(encoding="utf-8"))}
    provenance = [json.loads(line) for line in (dataset / "provenance.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_id = {int(row["id"]): row for row in provenance}
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    pending = sorted(manifest.get("missing_regenerated_attack_ids", []))
    if pending != list(REQUIRED_IDS) and not (args.replace_existing and not pending):
        raise ValueError("Dataset is not in the expected pending/finalized state")
    for sample_id in REQUIRED_IDS:
        name = labels[sample_id]["image"]
        source = replacements / name
        destination = dataset / "attacked/images" / name
        if not source.is_file(): raise FileNotFoundError(source)
        if destination.exists() and not args.replace_existing: raise FileExistsError(destination)
        with Image.open(source) as attack:
            attack_format, attack_size = str(attack.format), attack.size
            attack.verify()
        with Image.open(dataset / "clean/images" / name) as clean:
            clean_size = clean.size
        if attack_size != clean_size: raise ValueError(f"Dimension mismatch for ID {sample_id}")
        if sha256(source) == sha256(dataset / "clean/images" / name):
            raise ValueError(f"Attack equals clean image for ID {sample_id}")
        shutil.copyfile(source, destination)
        by_id[sample_id]["attack_sha256"] = sha256(destination)
        by_id[sample_id]["attack_format"] = attack_format
        by_id[sample_id]["attack_regenerated"] = True
        by_id[sample_id]["attack_target_label"] = labels[sample_id]["attack_target_label"]
        by_id[sample_id]["attack_generation"] = {
            "algorithm": "SSA_CommonWeakness",
            "epsilon": "16/255",
            "step_size": "1/255",
            "total_steps": 50,
            "input_preprocessing": "formal 224x224 aspect-preserving letterbox clean image",
            "surrogate_count": 5,
        }
    with (dataset / "provenance.jsonl").open("w", encoding="utf-8") as handle:
        for row in provenance:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    checksum_lines = []
    for condition in ("clean", "attacked"):
        for path in sorted((dataset / condition / "images").iterdir(), key=lambda value: value.name):
            checksum_lines.append(f"{sha256(path)}  {condition}/images/{path.name}")
    (dataset / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    manifest["missing_regenerated_attack_ids"] = []
    manifest["regenerated_attack_ids"] = list(REQUIRED_IDS)
    manifest["regenerated_attack_targets"] = {
        str(sample_id): labels[sample_id]["attack_target_label"] for sample_id in REQUIRED_IDS
    }
    manifest["attack_generation"] = {
        "algorithm": "SSA_CommonWeakness",
        "epsilon": "16/255",
        "step_size": "1/255",
        "total_steps": 50,
        "input_preprocessing": "formal 224x224 aspect-preserving letterbox clean image",
        "surrogates": [
            "CLIP ViT-L/14",
            "OpenCLIP ViT-B/32 LAION2B",
            "OpenCLIP ViT-bigG/14 LAION2B",
            "OpenCLIP ViT-B/16 LAION2B",
            "OpenCLIP ConvNeXt-Large LAION2B"
        ]
    }
    manifest["status"] = "passed"
    (dataset / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Finalized {dataset}")


if __name__ == "__main__":
    main()
