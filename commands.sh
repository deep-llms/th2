#1 +30+a
#th2-6y3o-a2-stage1b-fwe-part2-20261030-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PART2_SPENT=$(/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'LEDGER'
from datetime import datetime,timezone
print(3299.9433463821188+(datetime.now(timezone.utc)-datetime.fromisoformat('2026-10-29T20:19:10.047762+00:00')).total_seconds())
LEDGER
)
printf 'PART2_PRIOR_LEDGER_SECONDS=%s\n' "$PART2_SPENT"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_stage1b_heals --phase part2 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-part2-20261030-a01 --inputs /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs --heal-data /mnt/local/_data/deep-llms_th2/a2_stage1b_fwe_shuffle2030_20261030 --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-downstream-part1-20261030-a01/source/resources/llm_pretrain_burn.py --preflight-root /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-preflight-20261030-a01 --preflight-sha256 0ddedf66933134338cc87f872b28e5df9d8a67ba0cf6460d482c90ebe0321b8b --session a2_stage1b_fwe_part2_20261030_a01 --code-commit 42f7820ce72d932ff3d75edc96e16538dbd85df5 --spent-seconds "$PART2_SPENT" --deadline 2026-10-31T04:06:27.182436+08:00 --over20h-reported --prior /mnt/local/deep-llms_th2/resources/a2_stage1b_part2_prior_20261030.json --prior-sha256 c126960e5f1632688da9347723e56b1aa1c8314dfe90c9d425de6043334f01b8
