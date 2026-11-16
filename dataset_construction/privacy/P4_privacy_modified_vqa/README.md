# P.4 Privacy-Modified Medical VQA

This directory freezes the complete P.4 construction chain. The source package
contains the 81-row DAS spreadsheet, its exact JSON extraction, the 243-item
Qwen3-32B expansion, all 243 image prompts, and the three runtime caches used to
resolve the 33 `{GENERATED PHI}` templates. The `legacy` directory preserves the
original generation scripts byte-for-byte.

The accepted release contains the frozen Qwen-Image-2512 outputs. Each image was
generated at 1024 × 1024 with 50 inference steps, true CFG scale 4.0, and a seed
equal to its sample ID. The build script materializes the release without local
absolute paths and records sample-level provenance and hashes.

The source expansion contains three category strings that differ only by Unicode
hyphen/spacing. All 81 source triplets preserve the category after normalization.
No fallback or empty image prompt occurs in the frozen 243-item release.

This admission validates construction provenance, structure, and frozen image
integrity. It does not claim a new clinical or semantic review of every generated
scene.

Existing target-model outputs are migrated to `logs/privacy/P4`; no new
target-model run is required. The migration preserves each model's four archived
dynamic rewrite prompts and responses, applies the confirmed branch judgments,
and writes a 243-case consolidated result for every model. The original logs are
left unchanged. Replacing the historical dynamic prompts with one shared rewrite
would define a different protocol and would require a new run; that is not part
of this compatibility-preserving admission.

