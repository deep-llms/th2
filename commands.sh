#1 +60+a
#th2-ccm-seed29-result-only-export-20260915-a01
set -euo pipefail
test "$PWD" = /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-8mgy-worker-0
/usr/bin/python3 scripts/export_seed29_results.py --base /mnt/local/_outputs/deep-llms_th2 --output /mnt/local/_outputs/deep-llms_th2/ccm_seed29_results_export_20260915_a01
cat /mnt/local/_outputs/deep-llms_th2/ccm_replication_seed29_20260914_a01/status.json
/usr/bin/python3 scripts/gpu_status.py
echo CCM_SEED29_RESULT_EXPORT_OK
