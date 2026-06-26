#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    # "bias-vqa-text-race"
    # "bias-vqa-text-language"
    # "bias-vqa-text-emotion"
    # "bias-vqa-text-cognitive"
    # "bias-vqa-text"

    "bias-vqa-race"
    "bias-vqa-language"
    "bias-vqa-emotion"
    "bias-vqa-cognitive"
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
    elif [[ "$dataset_id" == "bias-vqa" ]]; then
        
        python ours/utils/bias_utils.py \
            --json_files \
            "logs/fairness/f1-bias-ref/${model_id}/bias-vqa-race.json" \
            "logs/fairness/f1-bias-ref/${model_id}/bias-vqa-language.json" \
            "logs/fairness/f1-bias-ref/${model_id}/bias-vqa-emotion.json" \
            "logs/fairness/f1-bias-ref/${model_id}/bias-vqa-cognitive.json" \
            --output_file \
            "logs/fairness/f1-bias-ref/${model_id}/bias-vqa.json"

    else  
        config="ours/configs/task/fairness/f1-${dataset_id}.yaml"
        log_dir="logs/fairness/f1-bias-ref"

        python run_task.py \
        --config ${config} \
        --cfg-options \
            dataset_id=${dataset_id} \
            model_id=${model_id} \
            log_file="${log_dir}/${model_id}/${dataset_id}.json"


    fi
done