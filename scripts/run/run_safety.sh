#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    # "safety-risk"

    # "bap-jailbreak"
    "mcn-jailbreak"

    # "safety-vqa-clean"
    # "safety-vqa-risk-minimization"
    # "safety-vqa-authority"
    # "safety-vqa-urgency"
    # "safety-vqa-consequence-obfuscation"
    # "safety-vqa-diagnostic-anchoring"
)

for dataset_id in "${dataset_ids[@]}"; do

    echo "======================================"
    echo "Running $dataset_id ..."
    echo "======================================"

    # ====================================
    # Stage 1: Safety Risk
    # ====================================
    if [[ "$dataset_id" == "safety-risk" ]]; then

        config="ours/configs/task/safety/s1-${dataset_id}.yaml"
        log_dir="logs/safety/s1-safetyrisk"

        python run_task.py \
            --config "${config}" \
            --cfg-options \
                dataset_id="${dataset_id}" \
                model_id="${model_id}" \
                log_file="${log_dir}/${model_id}/${dataset_id}.json"

    # ====================================
    # Stage 2: Jailbreak
    # ====================================
    elif [[ "$dataset_id" == "bap-jailbreak" \
         || "$dataset_id" == "mcn-jailbreak" ]]; then

        config="ours/configs/task/safety/s2-${dataset_id}.yaml"
        log_dir="logs/safety/s2-jailbreak"

        python run_task.py \
            --config "${config}" \
            --cfg-options \
                dataset_id="${dataset_id}" \
                model_id="${model_id}" \
                log_file="${log_dir}/${model_id}/${dataset_id}.json"

    # ====================================
    # Stage 3: Safety VQA
    # ====================================
    elif [[ "$dataset_id" == "safety-vqa" ]]; then
        python ours/utils/safety_utils.py \
            --json_files \
            "logs/safety/p3-safety-vqa/${model_id}/safety-vqa-risk-minimization.json" \
            "logs/safety/p3-safety-vqa/${model_id}/safety-vqa-authority.json" \
            "logs/safety/p3-safety-vqa/${model_id}/safety-vqa-urgency.json" \
            "logs/safety/p3-safety-vqa/${model_id}/safety-vqa-consequence-obfuscation.json" \
            "logs/safety/p3-safety-vqa/${model_id}/safety-vqa-diagnostic-anchoring.json" \
            --output_file \
            "logs/safety/p3-safety-vqa/${model_id}/safety-vqa.json"
    else

        config="ours/configs/task/safety/s3-${dataset_id}.yaml"
        log_dir="logs/safety/s3-safety-vqa"

        python run_task.py \
            --config "${config}" \
            --cfg-options \
                dataset_id="${dataset_id}" \
                model_id="${model_id}" \
                log_file="${log_dir}/${model_id}/${dataset_id}.json"

    fi

done