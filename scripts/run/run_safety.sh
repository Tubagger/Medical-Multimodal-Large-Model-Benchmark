#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    # "safety-risk-text"
    "safety-risk"
    # "bap-jailbreak"
    # "mcn-jailbreak"
)

for dataset_id in "${dataset_ids[@]}"; do

    echo "======================================"
    echo "Running $dataset_id ..."
    echo "======================================"

    # =========================
    # determine stage
    # =========================
    if [[ "$dataset_id" == *"jailbreak"* ]]; then

        stage="s2-jailbreak"

    elif [[ "$dataset_id" == *"risk"* ]]; then

        stage="s1-safetyrisk"

    else

        stage="s3-toxicity"

    fi

    # =========================
    # config path
    # =========================
    config="ours/configs/task/safety/${stage%%-*}-${dataset_id}.yaml"

    # =========================
    # log path
    # =========================
    log_file="logs/safety/${stage}/${model_id}/${dataset_id}.json"

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