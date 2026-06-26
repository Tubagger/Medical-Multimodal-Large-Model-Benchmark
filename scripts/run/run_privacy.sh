#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    # "privacy-recognition"
    # "privacy-inflow"
    # "privacy-detection"
    "privacy-inference"


    "privacy-vqa-focus-distraction"
    "privacy-vqa-subtle-request"
    "privacy-vqa-trap-warning"
    "privacy-vqa-well-intention"
    "privacy-vqa"

)


for dataset_id in "${dataset_ids[@]}"; do
    echo "Running $dataset_id ..."

    if [[ "$dataset_id" == "privacy-recognition" ]]; then
        config="ours/configs/task/privacy/p1-${dataset_id}.yaml"
        log_dir="logs/privacy/p1-awearness"
        
        python run_task.py \
        --config ${config} \
        --cfg-options \
            dataset_id=${dataset_id} \
            model_id=${model_id} \
            log_file="${log_dir}/${model_id}/${dataset_id}.json"

    elif [[ "$dataset_id" == "privacy-inflow" \
        || "$dataset_id" == "privacy-detection" \
        || "$dataset_id" == "privacy-inference" ]]; then

        config="ours/configs/task/privacy/p2-${dataset_id}.yaml"
        log_dir="logs/privacy/p2-leakage"

        python run_task.py \
            --config "${config}" \
            --cfg-options \
                dataset_id="${dataset_id}" \
                model_id="${model_id}" \
                log_file="${log_dir}/${model_id}/${dataset_id}.json"


    elif [[ "$dataset_id" == "privacy-vqa" ]]; then
        
        python ours/utils/privacy_utils.py \
            --json_files \
            "logs/privacy/p3-privacy-vqa/${model_id}/privacy-vqa-subtle-request.json" \
            "logs/privacy/p3-privacy-vqa/${model_id}/privacy-vqa-focus-distraction.json" \
            "logs/privacy/p3-privacy-vqa/${model_id}/privacy-vqa-trap-warning.json" \
            "logs/privacy/p3-privacy-vqa/${model_id}/privacy-vqa-well-intention.json" \
            --output_file \
            "logs/privacy/p3-privacy-vqa/${model_id}/privacy-vqa.json"

    else  
        config="ours/configs/task/privacy/p3-${dataset_id}.yaml"
        log_dir="logs/privacy/p3-privacy-vqa"

        python run_task.py \
        --config ${config} \
        --cfg-options \
            dataset_id=${dataset_id} \
            model_id=${model_id} \
            log_file="${log_dir}/${model_id}/${dataset_id}.json"
    fi
done