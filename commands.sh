#1 +30+a
#th2-swt-readonly-check-cache-tmp-leftovers-20260908-a01
set -euo pipefail
date -u
hostname
/mnt/local/conda-py311/envs/swt/bin/python - <<'PY'
import os
from pathlib import Path
deleted = [
    '/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02',
    '/mnt/local/_outputs/deep-llms_th2/swt/batch_benchmark_20260908_a01',
    '/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000',
]
for name in deleted:
    assert not os.path.lexists(name), name
    print('VERIFIED_ABSENT', name, flush=True)
for name in (
    '/mnt/local/_outputs/deep-llms_th2/swt',
    '/mnt/local/_data/deep-llms_th2/swt',
    '/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B',
):
    root = Path(name)
    matches = []
    if root.is_dir():
        def fail(error):
            raise error
        for base, dirs, files in os.walk(root, followlinks=False, onerror=fail):
            for item in dirs + files:
                if item.startswith(('cache-', 'tmp-')):
                    matches.append(str(Path(base) / item))
    print('SCAN_ROOT', name, 'EXISTS', root.exists(), 'CACHE_TMP_MATCHES', len(matches), flush=True)
    for item in sorted(matches)[:100]:
        print('MATCH', item, flush=True)
    if len(matches) > 100:
        print('LIST_TRUNCATED_AFTER_100_TOTAL_COUNT_ABOVE', flush=True)
print('SWT_READONLY_CACHE_TMP_CHECK_COMPLETE_NO_DELETIONS', flush=True)
PY
