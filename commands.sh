#1 +60+a
#th2-readonly-inventory-diagnostic-data-20260902-a11
set -euo pipefail

echo '=== read-only diagnostic data/model inventory; burns are not modified ==='
date -u
hostname

for TASK_ROOT in \
    /mnt/local/_data/@PROJECT@ \
    /mnt/local/_models/@PROJECT@ \
    /mnt/local/_outputs/@PROJECT@; do
    echo "--- root: $TASK_ROOT ---"
    if [[ -d "$TASK_ROOT" ]]; then
        find "$TASK_ROOT" -maxdepth 7 \
            \( -type d -o -type f \) \
            \( -name eval -o -name ar -o -name de -o -name en -o -name ru \
               -o -name vi -o -name zh -o -name config.json \
               -o -name eval_ppl_bytoken.npz -o -name 'tokenized_data_*' \) \
            -print 2>/dev/null | sort | sed -n '1,500p'
    else
        echo 'MISSING ROOT'
    fi
done

echo '=== required checkpoint roots ==='
for TASK_ROOT in \
    /mnt/local/_outputs/@PROJECT@/product_code_hashed_h2048 \
    /mnt/local/_outputs/@PROJECT@/groupreduce_matched_nested_tied_t4 \
    /mnt/local/_outputs/@PROJECT@/frequency_binned_ppl_four_10k_20260831_a03/merged; do
    if [[ -d "$TASK_ROOT" ]]; then
        du -sh "$TASK_ROOT"
        find "$TASK_ROOT" -maxdepth 2 -type f \
            \( -name eval_ppl_bytoken.npz -o -name eval_ppl.json \
               -o -name trainer_state.json -o -name train_config.json \) \
            -print | sort
    else
        echo "MISSING: $TASK_ROOT"
    fi
done

echo '=== GPU state unchanged ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
echo 'TH2 READONLY DIAGNOSTIC DATA INVENTORY COMPLETE'
