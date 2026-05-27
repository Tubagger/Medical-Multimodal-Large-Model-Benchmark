#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

bash scripts/run/run_truthfulness.sh $model_id
bash scripts/run/run_robustness.sh $model_id
bash scripts/run/run_fairness.sh $model_id
bash scripts/run/run_privacy.sh $model_id
bash scripts/run/run_safety.sh $model_id