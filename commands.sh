#1 +60+a
#th2-ccm-export-all-completed-results-20260914-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-8mgy-worker-0
cd /mnt/local/deep-llms_th2
/usr/bin/python3 scripts/export_pilot_results.py --base /mnt/local/_outputs/deep-llms_th2 --output /mnt/local/_outputs/deep-llms_th2/ccm_results_export_20260914_a01
/usr/bin/python3 -c 'import sys; sys.path.insert(0, "scripts"); from pilot_gpu_ops import inspect; inspect(list(range(8)), active=True); print("RESULT_EXPORT_COMPLETE_BURNS_UNTOUCHED")'
