#!/bin/bash
# The shared JSON defaults to Arm A; ordinary training options are unchanged.
set -euo pipefail
if [ "$#" -eq 0 ]; then
    set -- deep_kv.b200.json
fi
exec bash scripts/train_deep_kv.sh "$@"
