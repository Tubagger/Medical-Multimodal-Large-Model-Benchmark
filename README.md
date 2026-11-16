# MedMMLM-Trust-Bench

Code for **MedMMLM-Trust-Bench**, a benchmark for evaluating medical multimodal
large language models (MLLMs) across capability, robustness, fairness, privacy,
and safety.

This repository is a **code-only release**. It does not redistribute source
medical images, rendered PHI overlays, model weights, model responses, judge
outputs, or API credentials. Those artifacts remain subject to the licenses and
access conditions of their original providers.

## What Is Included

| Path | Contents |
|---|---|
| `run_model.py` | Entry point for selecting a model, checking weights, and running shared branches |
| `run_task.py` | Config-level entry point for one task, answer generation, or re-scoring |
| `ours/` | Model adapters, datasets, tasks, evaluators, metrics, prompts, and YAML configs |
| `scripts/` | 16-task batch launcher, saved-answer judging, inspection tools, tests, and environment snapshots |
| `score_calculation/` | Task/dimension aggregation and plotting code |
| `dataset_construction/` | Construction and validation code for the benchmark tasks |
| `data/manifest.json` | Metadata for the expected released data layout and task sizes |
| `run_state/shared/model_catalog.json` | Model directory names and lightweight weight manifests |
| `env/apikey.example.yaml` | Placeholder API credential file |

## What Is Not Included

- Raw or derived medical images, including synthetic PHI overlays.
- Downloaded upstream datasets and third-party attack artifacts.
- Model checkpoints or LoRA weights.
- Responses, judge outputs, `logs/`, SQLite indexes, and run caches.
- API keys, provider credentials, or private endpoint configuration.
- Manuscript sources, PDFs, and generated figure files.

## Installation

The framework is tested primarily with Python 3.10--3.12 and NVIDIA GPUs.
Install the common analysis dependencies with:

```bash
python -m pip install -r requirements.txt
```

Local MLLM inference also needs model-specific packages such as PyTorch,
Transformers, and provider libraries. The exported environment snapshots are in
`scripts/environments/`. They document the original environment versions but
may need adjustment for another CUDA or driver stack.

Create the local API credential file without committing it:

```bash
cp env/apikey.example.yaml env/apikey.yaml
# Fill only the keys needed for your run.
```

## Data Preparation

The code expects the final data under `data/` and the output tree under `logs/`.
`data/manifest.json` records the task IDs, expected sample counts, provenance
fields, and validation reports. See `data/README.md` for the directory contract.

The exact released data snapshot is not bundled with this repository. To run the
benchmark, obtain the upstream datasets under their respective licenses and use
the scripts in `dataset_construction/` to reconstruct and validate the task
files. Prompts and judge rubrics are already versioned under `ours/configs/`.

## Model Preparation

Place local checkpoints under `models/<directory_name>`, where
`<directory_name>` matches `run_state/shared/model_catalog.json`. For example:

```text
models/
  Lingshu-7B/
  Hulu-Med-7B/
  medgemma-4b-it/
```

Use `--model-path` to override the catalog location for one run. Weight-file
size checks in the catalog are examples from the evaluated snapshots; update
them if you intentionally use a different revision.

## Quick Checks

```bash
python -m compileall -q run_model.py run_task.py ours scripts score_calculation dataset_construction
python scripts/check_project.py
python run_model.py --list-models
```

`scripts/check_project.py` validates Python syntax, shell launch scripts, and
model configs. Data-dependent checks are skipped when the code-only release has
no local data snapshot.

## Running

List the available local models:

```bash
python run_model.py --list-models
```

Check a local model without loading it:

```bash
python run_model.py --model lingshu-7b --gpus 0,1 --check-only
```

Run the 16 task groups for one model:

```bash
python scripts/run_tasks.py --model lingshu-7b --gpus 0,1 --score none
python scripts/run_tasks.py --model gpt-5.4 --tasks C1 P1 --score none
```

Re-score saved answers without loading the evaluated model:

```bash
python scripts/run_tasks.py --model lingshu-7b --tasks P3 P4 S1 S2 S3 \
  --eval-only --score all
```

Run one task directly from its YAML config:

```bash
python run_task.py --config ours/configs/task/capability/t1-lesion-localization.yaml \
  --score rules
```

The task runner reuses completed records and only fills missing answers. Use the
same data snapshot, judge configuration, generation settings, and model revision
when comparing runs.

## Scoring

After task outputs exist under `logs/`, rebuild the 16-task and five-dimension
summary:

```bash
python score_calculation/build_scores.py
```

`score_calculation/build_scores.py` follows the manuscript definitions. In
particular, R.4 uses the mean branch accuracy over the four frozen rewrites.
The script writes final tables only to `score_calculation/results/`, which is
ignored by Git.

## Reproducibility Notes

- Capability, robustness, fairness, privacy, and safety use separate task
  definitions and scorer families.
- Frozen rewrites are shared across evaluated models; response-adaptive
  branches depend on each model's initial answer.
- The fixed LLM judge and its prompts are versioned in `ours/configs/judges/`.
- The repository does not include the frozen data, answers, or judge responses
  used for the manuscript tables. Exact numerical replication therefore requires
  the corresponding data and output snapshot in addition to this code.

## Responsible Use

The benchmark contains medical and potentially sensitive text and images when
fully populated. Do not commit data, model outputs, logs, or credentials. Do not
use benchmark scores as evidence of clinical safety or effectiveness. The tasks
measure behavior under controlled experimental conditions, not fitness for
clinical deployment.
