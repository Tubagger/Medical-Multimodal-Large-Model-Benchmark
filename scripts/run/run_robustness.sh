#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    # "ood"
    
    # "perturbed-data-clean"
    # "perturbed-data"

    # "adv-target-clean"
    # "adv-untarget-clean"
    # "adv-target"
    # "adv-untarget"

    "robustness-vqa-clean"
    "robustness-vqa-answer-flip"
    "robustness-vqa-question-negation"
    "robustness-vqa-option-expansion"
    "robustness-vqa-narrative-distraction"
    "robustness-vqa"
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
    elif [[ "$dataset_id" == "adv-target-clean" ]]; then
        config="ours/configs/task/robustness/r2-adv-target.yaml"
        log_dir="logs/robustness/r2-adv-attack"
    elif [[ "$dataset_id" == "adv-untarget-clean" ]]; then
        config="ours/configs/task/robustness/r2-adv-untarget.yaml"
        log_dir="logs/robustness/r2-adv-attack"
    elif [[ "$dataset_id" == "perturbed-data" ]]; then
        config="ours/configs/task/robustness/r3-perturbed-data.yaml"
        log_dir="logs/robustness/r3-perturbed"

    elif [[ "$dataset_id" == "robustness-vqa" ]]; then
        
        python ours/utils/robustness_utils.py \
            --json_files \
            "logs/robustness/r3-robustness-vqa/${model_id}/robustness-vqa-answer-flip.json" \
            "logs/robustness/r3-robustness-vqa/${model_id}/robustness-vqa-question-negation.json" \
            "logs/robustness/r3-robustness-vqa/${model_id}/robustness-vqa-option-expansion.json" \
            "logs/robustness/r3-robustness-vqa/${model_id}/robustness-vqa-narrative-distraction.json" \
            --output_file \
            "logs/robustness/r3-robustness-vqa/${model_id}/robustness-vqa.json"
        continue

    else  
        config="ours/configs/task/robustness/r3-${dataset_id}.yaml"
        log_dir="logs/robustness/r3-robustness-vqa"
    fi

    python run_task.py \
        --config ${config} \
        --cfg-options \
            dataset_id=${dataset_id} \
            model_id=${model_id} \
            log_file="${log_dir}/${model_id}/${dataset_id}.json"
done