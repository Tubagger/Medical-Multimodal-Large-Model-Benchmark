# #!/bin/bash

# echo "===================================="
# echo "Step 1: Run safety evaluation local"
# echo "===================================="
# # cd /home/aiotlab/lhx/code/Ours2.0

# models=(
#     "lingshu-7b"
#     # "lingshu-32b"
#     # "qwen3-vl-32b-instruct"
# )

# for model_id in "${models[@]}"
# do
#     echo "=================================="
#     echo "Running model: $model_id"
#     echo "=================================="

#     # bash scripts/run/run_truthfulness.sh $model_id

#     bash scripts/run/run_robustness.sh $model_id

#     # bash scripts/run/run_fairness.sh $model_id

#     bash scripts/run/run_privacy.sh $model_id

#     bash scripts/run/run_safety.sh $model_id


# done

#!/bin/bash

LOG_FILE="runtime.log"

echo "===================================="
echo "Step 1: Run safety evaluation local"
echo "====================================" 

models=(
    "lingshu-7b"
    # "lingshu-32b"
    # "qwen3-vl-32b-instruct"
)

for model_id in "${models[@]}"
do
    echo "==================================" |
    echo "Running model: $model_id" | 
    echo "==================================" | 

    # ---------------- Robustness ----------------
    start=$(date +%s)
    echo "[START] robustness $model_id : $(date)" | tee -a "$LOG_FILE"

    bash scripts/run/run_robustness.sh "$model_id"

    end=$(date +%s)
    cost=$((end-start))
    echo "[END] robustness $model_id : $(date)" | tee -a "$LOG_FILE"
    echo "[TIME] robustness $model_id : ${cost}s" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # ---------------- Privacy ----------------
    start=$(date +%s)
    echo "[START] privacy $model_id : $(date)" | tee -a "$LOG_FILE"

    bash scripts/run/run_privacy.sh "$model_id"

    end=$(date +%s)
    cost=$((end-start))
    echo "[END] privacy $model_id : $(date)" | tee -a "$LOG_FILE"
    echo "[TIME] privacy $model_id : ${cost}s" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # ---------------- Safety ----------------
    start=$(date +%s)
    echo "[START] safety $model_id : $(date)" | tee -a "$LOG_FILE"

    bash scripts/run/run_safety.sh "$model_id"

    end=$(date +%s)
    cost=$((end-start))
    echo "[END] safety $model_id : $(date)" | tee -a "$LOG_FILE"
    echo "[TIME] safety $model_id : ${cost}s" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done