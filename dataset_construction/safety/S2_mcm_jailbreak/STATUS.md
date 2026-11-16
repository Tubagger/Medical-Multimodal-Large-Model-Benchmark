# S.2 MCM status

Data status: `passed`. Model generation: `pending`.

All 270 optimized images and all 270 complete attack prompts have been frozen
and validated. New runtime inputs exactly match these artifacts. Historical
clean-image / missing-suffix logs are incompatible and require a fresh run.

The launcher prepares the current 16 local models; model inference has not been
started as part of code preparation. Source CSVs and selection cohort are frozen;
the corrected build reuses the original cohort rather than selecting by results.
