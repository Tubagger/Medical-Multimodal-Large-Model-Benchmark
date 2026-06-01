#!/bin/bash

echo "===================================="
echo "Step 1: Run safety evaluation local"
echo "===================================="
# cd /home/aiotlab/lhx/code/Ours2.0

models=(
    "lingshu-7b"
    # "lingshu-32b"
    # "qwen3-vl-32b-instruct"
)

for model_id in "${models[@]}"
do
    echo "=================================="
    echo "Running model: $model_id"
    echo "=================================="

    # bash scripts/run/run_truthfulness.sh $model_id

    # bash scripts/run/run_robustness.sh $model_id

    # bash scripts/run/run_fairness.sh $model_id

    # bash scripts/run/run_privacy.sh $model_id

    bash scripts/run/run_safety.sh $model_id
done