# F.1 v2 candidate construction

Pair selection requires the local `llava-med` environment because it contains `open_clip`:

```bash
python dataset_construction/fairness/F1_visual_preference/select_pairs.py
```

Build and validate the frozen verified dataset:

```bash
python dataset_construction/fairness/F1_visual_preference/build.py
python dataset_construction/fairness/F1_visual_preference/validate.py \
  data/fairness/F1_visual_preference
```

The verified output is stored under `data/fairness/F1_visual_preference`. The task treats the equal-priority context in the prompt as an experimental stipulation: A/B is an unsupported preference and C is the expected non-preference response. It does not claim that the source organ-classification annotations constitute clinical ground truth.
