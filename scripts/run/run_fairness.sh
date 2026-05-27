#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    # "bias-vqa-text"
    "bias-vqa"
    # "preference-choice"
)



#frist round
python run_bias_ref.py --config ours/configs/task/fairness/f1-bias-ref.yaml --cfg-options \
    dataset_id="bias-ref" \
    model_id=${model_id} \
    log_file="logs/fairness/f1-bias-ref/${model_id}/bias-ref.json"




for dataset_id in "${dataset_ids[@]}"; do
    echo "Running $dataset_id ..."

    if [[ "$dataset_id" == "preference-choice" ]]; then
        config="ours/configs/task/fairness/f2-preference-choice.yaml"
        log_dir="logs/fairness/f2-preference-choice"
        
        python run_task.py \
        --config ${config} \
        --cfg-options \
            dataset_id=${dataset_id} \
            model_id=${model_id} \
            log_file="${log_dir}/${model_id}/${dataset_id}.json"
    else  
        config="ours/configs/task/fairness/f1-bias-vqa.yaml"
        log_dir="logs/fairness/f1-bias-ref"

        python run_bias_vqa.py \
        --config ${config} \
        --cfg-options \
            dataset_id=${dataset_id} \
            model_id=${model_id} \
            log_file="${log_dir}/${model_id}/${dataset_id}.json"
    fi


done