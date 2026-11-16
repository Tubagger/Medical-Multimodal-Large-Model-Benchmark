# Data Layout

The code-only release does not redistribute the benchmark images, annotations,
or frozen rewrite payloads. It provides the construction and validation code,
the task manifests, and the expected directory contract.

The expected task layout follows `data/manifest.json`:

```text
data/
  capability/
  robustness/
  fairness/
  privacy/
  safety/
  shared_inputs/
```

Task-level `validation_report.json` and, where applicable, rule-validation
reports document the checks applied to the released snapshot. `manifest.json`
records sample counts and construction paths; it does not grant access to any
upstream dataset.

To reconstruct the benchmark:

1. Obtain each source dataset and third-party attack artifact under its original
   license.
2. Run the corresponding builder in `dataset_construction/<dimension>/<task>/`.
3. Run the task validator and retain the generated validation report.
4. Keep frozen rewrite inputs and prompts unchanged for comparisons.
5. Do not commit patient data, rendered PHI, model responses, or judge outputs.

The directory structure alone is not evidence of clinical validity. See the
paper and the task-level documentation for construct boundaries and review
status.
