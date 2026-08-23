#1 +60+a
#th2-locate-sparse-emb-conda-env-20260823-0214
set -euo pipefail

echo '=== locate sparse_emb environment ==='
date -u
hostname
echo "HOME=$HOME"
echo "PATH=$PATH"
command -v conda || true
command -v python || true

for TASK_CANDIDATE in \
    /mnt/local/conda/envs/sparse_emb/bin/python \
    /mnt/local/miniconda3/envs/sparse_emb/bin/python \
    /mnt/local/_conda/envs/sparse_emb/bin/python \
    /opt/conda/envs/sparse_emb/bin/python; do
    if [ -x "$TASK_CANDIDATE" ]; then
        echo "FOUND_CANDIDATE: $TASK_CANDIDATE"
        "$TASK_CANDIDATE" -c 'import sys, torch, datasets, transformers; print(sys.executable); print(torch.__version__, datasets.__version__, transformers.__version__)'
    else
        echo "NOT_FOUND: $TASK_CANDIDATE"
    fi
done

echo '=== bounded filesystem search ==='
find /mnt/local -maxdepth 7 -type f -path '*/sparse_emb/bin/python*' -print 2>/dev/null | sort
find /mnt/local -maxdepth 5 -type f -name conda -print 2>/dev/null | sort | head -50
echo 'TH2 SPARSE EMB ENV SEARCH COMPLETE'
