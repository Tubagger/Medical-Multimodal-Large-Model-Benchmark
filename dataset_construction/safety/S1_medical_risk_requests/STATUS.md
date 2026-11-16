# S.1 status

Status: **passed with observations**

- 214,219 source rows and 94 source categories verified;
- selected seven-category pool contains 213,553 rows;
- seeded selection exactly reproduces the frozen 210 rows;
- seven categories contain exactly 30 samples each;
- all frozen translation and image-prompt stages align by order and ID;
- zero fallback or empty image prompts;
- 210 unique, valid RGB PNG images at 1664 × 928;
- all released images match the upstream frozen outputs byte-for-byte;
- 3,780 archived target responses across 18 models migrated;
- nine individually confirmed refusal-judge errors applied;
- all rebuilt AbR summaries independently validated;
- no target-model rerun was performed or required.

Eight image prompts conflict with the nominal no-visible-text instruction. The
images remain admitted for historical compatibility, with the issue recorded in
the validation report. Legacy absolute paths and a partial-resume defect are
isolated by the verified frozen-data builder and did not affect this release.
