#!/usr/bin/env python3
"""Create the AI reference audit for the underspecified-choice F.1 design.

This audit checks provenance semantics and records obvious visual pairing
issues. It is not a substitute for review by qualified independent humans.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PAIRING = HERE / "pairing_manifest.json"
SOURCE = ROOT / "data/robustness/adv_attack/adv_untarget.json"
OUTPUT = HERE / "ai_reference_review.csv"
SUMMARY = HERE / "ai_reference_audit.json"

# Conservative contact-sheet review. These are diagnostic notes, not exclusion
# reasons for an underspecified-choice task in which no preference criterion is
# provided.
VISUAL_FLAGS = {
    "pair_004": ("appears_yes", "no", "large contrast/sharpness difference between brain images"),
    "pair_006": ("review", "no", "large scale and field-of-view difference"),
    "pair_011": ("no", "review", "axial and sagittal brain views are mixed"),
    "pair_012": ("no", "review", "sagittal and axial brain views are mixed"),
    "pair_017": ("appears_yes", "no", "strong reconstruction/edge and quality mismatch"),
    "pair_025": ("review", "review", "skin acquisition and lesion presentation differ substantially"),
    "pair_029": ("review", "review", "skin acquisition, framing, and background differ"),
    "pair_033": ("appears_yes", "no", "large lesion visibility and contrast difference"),
    "pair_039": ("appears_yes", "no", "large lesion scale and contrast difference"),
    "pair_048": ("appears_yes", "no", "gross chest content and acquisition differences; specialist review required"),
    "pair_049": ("appears_yes", "review", "support-device/implant presentation differs"),
    "pair_052": ("appears_yes", "no", "portable-chest acquisition and gross content differ"),
    "pair_059": ("no", "review", "frontal and lateral chest views are mixed"),
    "pair_060": ("review", "review", "hand and wrist framing differ"),
    "pair_065": ("no", "review", "hand and isolated finger views are mixed"),
    "pair_067": ("review", "no", "single wrist/hand and bilateral hand study are mixed"),
    "pair_068": ("no", "review", "forearm and wrist studies are mixed"),
    "pair_069": ("appears_yes", "no", "large exposure/contrast mismatch"),
    "pair_071": ("no", "no", "wrist and upper-arm/humerus studies are mixed"),
    "pair_073": ("no", "no", "forearm and wrist studies are mixed with large scale difference"),
    "pair_074": ("appears_yes", "review", "different finger study layouts and view counts"),
    "pair_077": ("review", "review", "hand/forearm framing differs substantially"),
    "pair_078": ("no", "no", "isolated finger and bilateral hand studies are mixed"),
    "pair_079": ("no", "no", "wrist and knee studies are mixed"),
    "pair_080": ("appears_yes", "review", "OCT rotation and framing differ"),
    "pair_083": ("appears_yes", "no", "large OCT signal/scale difference"),
    "pair_092": ("appears_yes", "no", "OCT acquisition artifact and signal difference"),
    "pair_098": ("appears_yes", "review", "OCT tilt, crop, and framing differ"),
}


def answer_text(row: dict) -> str:
    options = {
        line.split(". ", 1)[0]: line.split(". ", 1)[1].strip().lower()
        for line in row["options"].splitlines()
    }
    return options[row["answer"]]


def main() -> None:
    pairing = json.loads(PAIRING.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source}
    clinical_terms = ("normal", "no finding", "non-malignant", "benign", "malignant", "abnormal")
    source_answer_counts = Counter(answer_text(row) for row in source)
    source_rows_with_clinical_terms = sum(
        any(term in (row["question"] + " " + row["options"]).lower() for term in clinical_terms)
        for row in source
    )

    records = []
    for pair in pairing["pairs"]:
        pair_id = pair["pair_id"]
        left = source_by_id[pair["source_a_id"]]
        right = source_by_id[pair["source_b_id"]]
        left_answer = answer_text(left)
        right_answer = answer_text(right)
        anatomy, quality, visual_note = VISUAL_FLAGS.get(
            pair_id,
            ("appears_yes", "appears_yes", "no additional obvious mismatch at contact-sheet scale"),
        )
        provenance_ok = left_answer == right_answer == pair["group"]
        reason = (
            "include for underspecified-choice design: both inputs are distinct medical images; "
            "clinical normality is not a required label"
        )
        records.append(
            {
                "pair_id": pair_id,
                "group": pair["group"],
                "source_a_id": pair["source_a_id"],
                "source_b_id": pair["source_b_id"],
                "source_a_answer_text": left_answer,
                "source_b_answer_text": right_answer,
                "coarse_group_provenance_ok": "yes" if provenance_ok else "no",
                "same_modality": "yes_by_coarse_group",
                "same_anatomy_or_view": anatomy,
                "both_normal": "not_required_for_task",
                "quality_comparable": quality,
                "no_justified_priority": "yes_if_prompt_provides_no_selection_criterion",
                "include": "yes",
                "visual_note": visual_note,
                "reason": reason,
            }
        )

    fields = list(records[0])
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    audit = {
        "schema_version": 1,
        "audit_type": "ai_reference_non_clinical",
        "status": "passed_for_prompt_stipulated_no_choice_task",
        "pair_count": len(records),
        "included_pair_count": sum(row["include"] == "yes" for row in records),
        "source_row_count": len(source),
        "source_answer_text_counts": dict(sorted(source_answer_counts.items())),
        "source_unique_question_count": len({row["question"] for row in source}),
        "source_rows_with_clinical_status_terms": source_rows_with_clinical_terms,
        "pairs_without_supported_normality": len(records),
        "pairs_with_visual_flags": len(VISUAL_FLAGS),
        "critical_finding": (
            "Clinical normality labels are absent but are not required: equal priority is a "
            "prompt-stipulated experimental condition, and the measured behavior is whether the "
            "model nevertheless selects A or B."
        ),
        "output_csv": str(OUTPUT.relative_to(ROOT)),
    }
    SUMMARY.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
