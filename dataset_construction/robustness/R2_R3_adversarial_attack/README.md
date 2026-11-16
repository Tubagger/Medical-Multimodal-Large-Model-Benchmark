# Corrected R.2/R.3 adversarial-attack construction

This construction preserves the published, model-independent seed-42 selection of 200 images (40 each from Brain MRI, ISIC, CheXpert, MURA, and OCT) while correcting the two source-dataset names that were incorrectly used as anatomical answers.

- IDs 40–79: `isic` is replaced by `skin`.
- IDs 120–159: `mura` is replaced by the MURA anatomical region recovered from the original source path.
- The 80 affected prompts use a neutral anatomical-structure/body-region question.
- R.2 reuses the frozen untargeted attacks because their construction does not use a target text label.
- R.3 reuses 77 targeted attacks whose targets remain wrong after correction.
- R.3 IDs 53, 67, and 70 originally targeted `skin`. They require newly generated attacks toward `brain`, `bladder`, and `prostate`, respectively.

No tested-model responses or correctness values are used in selection, correction, or retargeting. Existing model logs are not valid for the 80 changed prompts.

The builder validates the exact seed-42 source replay against the archived original-resolution copied images before materializing the corrected frozen evaluation images.

`regenerate_r3_targets.py` reproduces the original five-surrogate SSA settings for the three invalidated attacks. Its input must be the formal 224×224 letterbox clean images, not the archived original-resolution copies; the script rejects any other dimensions. After generation, `finalize_r3.py` installs the new images and updates checksums/provenance; run `validate.py` once more before admitting R.3.
