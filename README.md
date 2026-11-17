# MedMLLM-Trust-Bench

**Benchmarking Medical Multimodal Large Language Models from Capability to Trustworthiness**

MedMLLM-Trust-Bench evaluates medical multimodal large language models across
**Capability, Robustness, Fairness, Privacy, and Safety**. The paper protocol
contains **4,017 base evaluation instances across 16 tasks** and compares
**17 models: 15 open-weight and two proprietary systems**.

The framework combines static inputs, frozen LLM-assisted modifications, and
response-adaptive challenges. It reports task and dimension profiles alongside
component metrics for privacy protection, harmful compliance, response
appropriateness, and paired robustness.

**Release scope:** this repository contains code, configuration files, prompts,
and lightweight manifests. It does not bundle benchmark images, annotations,
frozen rewrite payloads, model weights, evaluation responses, or judge outputs.
Exact reproduction of the paper's numerical results requires the corresponding
frozen inputs and output snapshots in addition to this code.

[Tasks](#benchmark-tasks) · [Setup](#setup) · [Usage](#usage) ·
[Scoring](#scoring) · [Data access](#data-access-and-reproducibility) ·
[License](#license)

## Benchmark Tasks

| Dimension | ID | Task | Base instances |
|---|---|---|---:|
| Capability | C1 | Abnormality recognition | 380 |
| Capability | C2 | Lesion localization | 240 |
| Capability | C3 | Medical reasoning | 270 |
| Robustness | R1 | OOD medical VQA | 219 |
| Robustness | R2 | Untargeted attack | 200 |
| Robustness | R3 | Targeted attack | 200 |
| Robustness | R4 | Robustness-modified VQA | 200 |
| Fairness | F1 | Visual preference | 200 |
| Fairness | F2 | Fairness response-adaptive VQA | 415 |
| Privacy | P1 | Privacy recognition | 260 |
| Privacy | P2 | PHI field identification and transcription | 260 |
| Privacy | P3 | Privacy inference | 240 |
| Privacy | P4 | Privacy-modified VQA | 243 |
| Safety | S1 | Safety risk recognition | 210 |
| Safety | S2 | MCM jailbreak | 270 |
| Safety | S3 | Safety-modified VQA | 210 |
| **Total** | | **16 tasks** | **4,017** |

Counts refer to task-specific base units, not unique patients or model calls.
Clean/attacked pairs, rewrite branches, and repeated baseline responses add
derived inputs. CLI task IDs omit the dot used in the paper, for example `C1`
corresponds to C.1.

## Repository Structure

| Path | Purpose |
|---|---|
| [`run_model.py`](run_model.py) | Model selection, environment/weight checks, shared-branch execution, and saved-answer judging |
| [`run_task.py`](run_task.py) | Single-config inference and re-scoring |
| [`ours/`](ours/) | Model adapters, datasets, task runners, evaluators, and prompt/config definitions |
| [`scripts/`](scripts/) | Batch launcher, saved-answer judging, inspection tools, checks, and environment snapshots |
| [`dataset_construction/`](dataset_construction/) | Task construction and validation code |
| [`data/`](data/) | Expected data layout and task manifest; see the data README |
| [`score_calculation/`](score_calculation/) | Saved-metric collection, aggregation, and plotting |
| [`run_state/shared/model_catalog.json`](run_state/shared/model_catalog.json) | Checkpoint directory names and expected weight-file metadata |
| [`env/apikey.example.yaml`](env/apikey.example.yaml) | API credential template without real keys |

Local results are written to `logs/` and `score_calculation/results/`.
These generated artifacts are excluded from Git.

## Setup

Run the following commands from the repository root.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

This installs shared analysis and API-client dependencies. Local inference also
requires model-specific versions of PyTorch, Transformers, and related packages.
See [`scripts/environments/`](scripts/environments/) for environment snapshots.
CUDA, driver, and Python compatibility depends on the selected model.

The launchers select model-specific Conda environments. Use `--env-root` or
`MEDBENCH_ENV_ROOT` to specify the directory containing those environments;
installing `requirements.txt` alone does not create them.

### 2. Prepare API access when needed

```bash
cp env/apikey.example.yaml env/apikey.yaml
```

Fill in only the credentials required for your run. Configure the relevant
model/provider endpoint in `ours/configs/`. See [`env/README.md`](env/README.md)
for supported key names. The local credential file is ignored by Git.

API-backed inference, judging, and any uncached LLM-assisted construction may
incur provider charges. Setting `--score none` disables scoring, not API calls
required by target inference or task construction.

### 3. Prepare data and checkpoints

Obtain source datasets through their official channels and reconstruct the
required task files under `data/`. Follow [`data/README.md`](data/README.md) and
the corresponding task documentation in `dataset_construction/`.

Place local checkpoints under the directories specified by
`run_state/shared/model_catalog.json`, for example:

```text
models/
  Lingshu-7B/
  Hulu-Med-7B/
  medgemma-4b-it/
```

For `run_model.py`, `--model-path` overrides the checkpoint location for one
model. The catalog includes expected filenames and sizes from evaluated
snapshots; update that metadata if you intentionally use a different revision.
Weights and upstream datasets are not supplied by this repository.

## Usage

### Inspect the setup

```bash
# List local model adapters without loading a model.
python run_model.py --list-models

# Validate framework files; absent code-only data are reported separately.
python scripts/check_project.py

# Check one local model without inference.
python run_model.py --model lingshu-7b --gpus 0,1 --check-only
```

The paper's 17-model set is defined in `score_calculation/build_scores.py`.
Additional adapters may be present; the runtime adapter list is not the list
of models evaluated in the paper.

### Generate answers

```bash
# Preview the full task plan without inference or judging.
python scripts/run_tasks.py --model lingshu-7b --dry-run

# Run all 16 task groups for one model, without scoring.
python scripts/run_tasks.py --model lingshu-7b --gpus 0,1 --score none

# Select task groups for an API model.
python scripts/run_tasks.py --model gpt-5.4 --tasks C1 P1 --score none
```

The full-suite launcher accepts `C1`–`C3`, `R1`–`R4`, `F1`–`F2`,
`P1`–`P4`, and `S1`–`S3`. A dry run still requires the selected environment
to exist so that the launcher can resolve its Python executable.

### Score saved answers

```bash
# Re-score the five rubric-based tasks without loading the target model.
python scripts/run_tasks.py --model lingshu-7b --tasks P3 P4 S1 S2 S3 \
  --eval-only --score all

# Re-score a reference-based task using deterministic rules.
python scripts/run_tasks.py --model lingshu-7b --tasks C1 \
  --eval-only --score rules
```

Use `--score all` during generation to combine generation and scoring.
Missing answer files must be generated before an evaluation-only run.

### Run a single task config

```bash
python run_task.py --config ours/configs/task/capability/t1-lesion-localization.yaml \
  --score rules
```

Task configs define the model, dataset, prompt, generation settings, and output
location. Adjust these before a single-config run. Completed answers are reused
by the runner; retain the same input snapshot and settings when resuming.

## Scoring

After saved outputs are available under `logs/`:

```bash
# Collect stored metrics without new inference or judging.
python score_calculation/collect.py --model lingshu-7b

# Aggregate the paper's 16 tasks and five dimensions.
python score_calculation/build_scores.py
```

Task scores use a 0–100 scale. Numbered tasks receive equal weight within each
dimension, and the overall score is the unweighted mean of the five dimensions.
R4 uses mean accuracy across its four rewrite branches. Privacy and safety
composites retain separate risk and response-appropriateness components.

Aggregation writes to `score_calculation/results/`. Table and plotting scripts
may additionally require frozen aggregate snapshots; see
[`score_calculation/README.md`](score_calculation/README.md). Code availability
alone does not supply the paper's saved experiment results.

## Data Access and Reproducibility

Original datasets, images, annotations, and model weights remain subject to
their providers' licenses and access agreements. Obtain them through their
official channels and cite the corresponding sources. Task documentation and
construction manifests identify the expected sources and processing steps.

Adding adversarial perturbations, fictional PHI, or new annotations does not
establish permission to redistribute a source image. Check the applicable
source terms before sharing any raw or derived material. The project license
does not grant rights to third-party datasets or model weights.

For comparable runs, retain:

- Source sample IDs, task annotations, image checksums, and frozen rewrite inputs.
- Model/checkpoint revisions, image preprocessing, and generation settings.
- Judge model/configuration, rubric version, and criterion-level outputs.
- Sample pairing, valid denominators, and separate generation/parser/judge failures.

The fixed judge configuration is in `ours/configs/judges/`. Changing the judge
or regenerating frozen transformations defines a different evaluation condition.
Preserve the actual inputs and output snapshots for numerical reproduction.

## Responsible Use

The benchmark supports controlled evaluation and failure analysis. Its scores
do not certify clinical safety, effectiveness, or suitability for deployment.
Use fictional PHI for privacy stimuli and respect source-data access conditions.
Do not commit patient data, responses, judge outputs, or credentials.

Please cite the accompanying paper and the original datasets used in your work.

## License

Unless otherwise indicated, original code and documentation contributed to
MedMLLM-Trust-Bench are licensed under
[CC BY-NC-SA 4.0](LICENSE) (Creative Commons Attribution–NonCommercial–ShareAlike
4.0 International).

The license permits noncommercial sharing and adaptation subject to its terms:
retain attribution, link to the license, indicate changes, and distribute
adaptations under the same or a compatible license. Commercial use is not
authorized by this license. Any separately permitted commercial use requires
permission from the relevant rights holders.

The project is intended for academic research. This describes its purpose and
does not add an academic-only restriction to the CC BY-NC-SA 4.0 license.
Third-party code, datasets, and model weights retain their original licenses;
see [Third-Party Notices](THIRD_PARTY_NOTICES.md).

The [official license summary](https://creativecommons.org/licenses/by-nc-sa/4.0/)
explains the BY, NC, and SA conditions. Creative Commons
[recommends software-specific licenses for software](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software);
CC licenses do not provide software-specific source-distribution or patent terms.

If you identify potentially infringing material, please
[open an issue](https://github.com/Tubagger/Medical-Multimodal-Large-Model-Benchmark/issues)
with the affected paths and supporting information. We will review the report
and take appropriate action.
