# R.4 Dynamic Response Robustness

Status: `passed_with_observations`

The original four model-specific dynamic transformations are preserved. The
200-item base cohort and all 200 images passed checksum, decoding, metadata,
ID, option, and answer validation.

Across 18 historical models, 14,400 rewrite-cache rows and 18,000 saved result
rows match their images, targets, transformation rules, and copied prompts.
Question negation is explicitly treated as a multi-label task whose target is
the set of all non-original option letters; scoring uses set equality.

The current parser changes 16 historical parsed predictions and 12 row scores,
all in InternVL3.5-14B answer flip. The four-branch intersection and final R.4
score are unchanged. Existing JSON wrappers, 229 no-keyword negation instances,
8 duplicate expansion options, and 400 absent response fields are recorded as
observations and are not silently rewritten.
