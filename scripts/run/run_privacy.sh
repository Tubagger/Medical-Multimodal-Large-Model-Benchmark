#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "privacy-recognition"

    # "privacy-inflow"
    # "privacy-detection"

    # "privacy-inference"
    # "privacy-vqa"

)

for dataset_id in "${dataset_ids[@]}"; do

    echo "Running $dataset_id ..."

    # =========================
    # task stage
    # =========================
    if [[ "$dataset_id" == "privacy-recognition" ]]; then
        stage="p1-awareness"
    else
        stage="p2-leakage"
    fi

    # =========================
    # config path
    # =========================
    config="ours/configs/task/privacy/${stage%%-*}-${dataset_id}.yaml"

    # =========================
    # log path
    # =========================
    log_file="logs/privacy/${stage}/${model_id}/${dataset_id}.json"

    echo "Config: $config"
    echo "Log: $log_file"

    python run_task.py \
        --config "${config}" \
        --cfg-options \
            dataset_id="${dataset_id}" \
            model_id="${model_id}" \
            log_file="${log_file}"

done