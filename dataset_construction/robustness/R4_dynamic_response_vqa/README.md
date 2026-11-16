# R.4 dynamic response robustness

The original four dynamic transformations are preserved: answer flip, question negation, option expansion, and narrative distraction. The base cohort contains 200 seed-42 PMC-VQA samples.

Validation is limited to image integrity, source/prompt/cache alignment, transformation-rule structure, and rule-parser consistency. Question negation is interpreted as a multi-label set task: all non-original option letters form the target set, and prediction matching is set equality.

The existing audit of 18 models × 200 samples × 5 conditions is recorded in `revision_audit/R5评分核查表.md`. It found zero image-path, cache, label-rule, or saved-summary mismatches. Replaying the saved raw responses with the current parser changed 16 parsed predictions and 12 row decisions, all in the InternVL3.5-14B answer-flip branch; the four-branch intersection and final R.4 score did not change. The verified log copy contains the current-parser decisions while the historical logs remain untouched.

JSON wrappers in generated negation text, missing explicit negation keywords, and duplicate added options are retained as protocol observations rather than silently changing the dynamic task. No clinical or semantic relabeling was performed.
