# Score Aggregation

The scripts in this directory consume JSON outputs under `logs/` and write
derived tables or figures under `score_calculation/results/`.

- `build_scores.py` rebuilds the 16 task scores and five dimension means. R.4 is
  the mean accuracy across its four frozen rewrite branches.
- `collect.py` flattens saved `*.scored.json` metrics without running inference
  or a judge.
- `build_dimension_tables.py` and `build_supplementary_tables.py` render tables
  from a frozen aggregate snapshot.
- `plot_*.py` regenerate manuscript figures from frozen result files.

Result artifacts are intentionally ignored by Git. To compare runs, retain the
same data snapshot, model revision, judge configuration, prompts, and generation
settings.
