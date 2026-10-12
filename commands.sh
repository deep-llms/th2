#1 +60+a
#th2-q359-expanded-reduce-stop-20261013-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u -m scripts.stop_proxy_followup \
  --root /mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a03 \
  --session q359-expanded-paired-20k-20261012-a03 \
  --burn-session q359-expanded-paired-20k-20261012-a03-final-burn \
  --host thiennh-p6-q359-worker-0 \
  --output /mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a03/reduce-seed42-stop.json
