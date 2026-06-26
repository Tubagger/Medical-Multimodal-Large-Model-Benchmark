#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "anomaly-detection-brain"
    "anomaly-detection-chexpert"
    "anomaly-detection-isic"
    "anomaly-detection-mura"
    "anomaly-detection-oct"
    "lesion-localization"
    "logical-reasoning"
)

for dataset_id in "${dataset_ids[@]}"; do

    echo "======================================"
    echo "Running $dataset_id ..."
    echo "======================================"

    # =========================
    # determine stage
    # =========================
    if [[ "$dataset_id" == *"logical-reasoning"* ]]; then

        stage="t2-logic"

    else

        stage="t1-basic"

    fi

    # =========================
    # config path
    # =========================
    config="ours/configs/task/truthfulness/${stage%%-*}-${dataset_id}.yaml"

    # =========================
    # log path
    # =========================
    log_file="logs/truthfulness/${stage}/${model_id}/${dataset_id}.json"

    # =========================
    # mkdir
    # =========================
    mkdir -p "$(dirname "$log_file")"

    echo "Config: $config"
    echo "Log: $log_file"

    python run_task.py \
        --config "${config}" \
        --cfg-options \
            dataset_id="${dataset_id}" \
            model_id="${model_id}" \
            log_file="${log_file}"

done