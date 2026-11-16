# F.2 Response-Adaptive VQA image audit

This directory contains the reproducible image-integrity audit for F.2.  It
does not regenerate or modify the 415 source images.

Run from the repository root:

```bash
python dataset_construction/fairness/F2_response_adaptive_vqa/audit_images.py
```

The optional semantic near-duplicate audit requires PyTorch and OpenCLIP:

```bash
python dataset_construction/fairness/F2_response_adaptive_vqa/audit_clip_duplicates.py
```

The default outputs are written to `revision_audit/F2_image_audit/`:

- `image_audit.json`: machine-readable integrity and log-mapping summary;
- `image_inventory.csv`: dimensions, format, hashes, byte size and category;
- `checksums.sha256`: frozen SHA-256 inventory;
- `contact_sheets/`: nine pages covering all 415 images;
- `clip_near_duplicate_audit.json`: semantic near-duplicate check;
- `REPORT.md`: combined machine and manual review conclusions.

The historical F.2 model outputs are under `logs/fairness/F2/` even
though the directory name says F.1.  The audit checks that all four F.2
branches in those logs reference the expected image for every case.

