#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    # "ood"
    # "perturbed-data"
    "adv-target"
    "adv-untarget"
)

for dataset_id in "${dataset_ids[@]}"; do
    echo "Running $dataset_id ..."

    if [[ "$dataset_id" == "ood" ]]; then
        config="ours/configs/task/robustness/r1-ood.yaml"
        log_dir="logs/robustness/r1-ood"
    elif [[ "$dataset_id" == "adv-target" ]]; then
        config="ours/configs/task/robustness/r2-adv-target.yaml"
        log_dir="logs/robustness/r2-adv-attack"
    elif [[ "$dataset_id" == "adv-untarget" ]]; then
        config="ours/configs/task/robustness/r2-adv-untarget.yaml"
        log_dir="logs/robustness/r2-adv-attack"
    else
        config="ours/configs/task/robustness/r3-perturbed-data.yaml"
        log_dir="logs/robustness/r3-perturbed"
    fi

    python run_task.py \
        --config ${config} \
        --cfg-options \
            dataset_id=${dataset_id} \
            model_id=${model_id} \
            log_file="${log_dir}/${model_id}/${dataset_id}.json"
done