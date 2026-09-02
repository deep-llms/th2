#1 +60+a
#th2-readonly-locate-diagnostic-assets-20260902-a12
set -euo pipefail

echo '=== read-only locate diagnostic assets across /mnt/local; burns unchanged ==='
date -u
hostname

echo '=== top-level mount namespaces ==='
find /mnt/local -mindepth 1 -maxdepth 2 -type d -print 2>/dev/null \
    | sort | sed -n '1,500p'

echo '=== exact experiment and model directories ==='
find /mnt/local -xdev -maxdepth 6 -type d \
    \( -name product_code_hashed_h2048 \
       -o -name groupreduce_matched_nested_tied_t4 \
       -o -name frequency_binned_ppl_four_10k_20260831_a03 \
       -o -name Qwen3-0.6B -o -name Qwen_Qwen3-0.6B \) \
    -print 2>/dev/null | sort

echo '=== diagnostic files and checkpoint-10000 parents ==='
find /mnt/local -xdev -maxdepth 8 \
    \( -type d -name checkpoint-10000 \
       -o -type f -name eval_ppl_bytoken.npz \) \
    -print 2>/dev/null | sort | sed -n '1,1000p'

echo '=== GPU state unchanged ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
echo 'TH2 READONLY DIAGNOSTIC ASSET LOCATION COMPLETE'
