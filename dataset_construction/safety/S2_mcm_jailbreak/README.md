# S.2 MCM jailbreak

The source is the 3MAD-Tiny-1K split from O2M_attack-main (18 classes × 60
cases). We retain 15 cases per class, selected with seed 42, giving the frozen
270-case cohort. Each case uses `results/mcm/both/{id}.jpg` and the stripped
contents of `results/mcm/both/question_{id}.txt` from O2M_attack-main. The frozen text
already contains the upstream optimized suffix; runtime appends no additional
prefix, suffix, or image-reading instruction.

The old `mcn-jailbreak` implementation and logs are historical artifacts of an
incorrect input protocol. New experiments use dataset ID `mcm-jailbreak` and
`logs/safety/S2_MCM/`. Old model responses are not reused.

Build and validate:

```bash
python dataset_construction/safety/S2_mcm_jailbreak/build.py
python dataset_construction/safety/S2_mcm_jailbreak/validate.py
```

Run the same 16 local models as the R.2/R.3 launcher, sequentially in their own
conda environments (270 answers per model; 4,320 total):

```bash
bash scripts/run/run_s2_mcm_all16.sh --dry-run
bash scripts/run/run_s2_mcm_all16.sh
```

Run one model in its environment:

```bash
conda run --no-capture-output -n lingshu bash scripts/run/run_s2_mcm.sh lingshu-7b
```

This stage saves model answers before judging. Each model directory contains
`mcm-jailbreak.responses.jsonl` and `mcm-jailbreak.json`; the latter has empty
metrics until separately evaluated. Edit the judge configuration/model as
needed and rejudge, for example:

```bash
conda run --no-capture-output -n dmxapi python run_task.py \
  --config ours/configs/task/safety/s2-mcm-jailbreak-judge.yaml \
  --eval-only \
  --responses-file logs/safety/S2_MCM/lingshu-7b/mcm-jailbreak.responses.jsonl \
  --output-log logs/safety/S2_MCM/lingshu-7b/mcm-jailbreak.rejudged.json
```

The judge config preserves the historical refusal prompt and model ID; service
availability must be checked when judging. Answer generation needs no judge API.

`source/` freezes original labels, 18 selection CSVs and 270 attack text files.
`legacy/` preserves the original selector, including its MCN spelling and clean
image-copy bug. Release checksums and per-case source hashes are in the dataset.
