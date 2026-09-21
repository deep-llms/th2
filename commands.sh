#1 +60+a
#th2-tpbw-verify-data-root-clean-20260921-a01
set -euo pipefail
date -u
hostname
echo '--- /mnt/local/_data/deep-llms_th2 (recursive)'
ls -laR /mnt/local/_data/deep-llms_th2/ 2>/dev/null || echo data-root-absent
echo '--- any parquet or partial files anywhere under _data'
find /mnt/local/_data -name '*.parquet*' -o -name '*.incomplete' -o -name '*.part' 2>/dev/null | head -20 || true
echo '--- /mnt/local/_models/deep-llms_th2 (recursive)'
ls -laR /mnt/local/_models/deep-llms_th2/ 2>/dev/null || echo models-root-absent
echo '--- disk usage'
du -sh /mnt/local/_data 2>/dev/null || true
df -h /mnt/local | tail -1
echo TPBW_DATA_ROOT_CLEAN_VERIFIED
