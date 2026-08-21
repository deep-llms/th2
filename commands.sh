#1
#th2-setup-env-20260821-185655
set -euo pipefail

echo '=== th2 environment setup ==='
date -u
hostname

bash scripts/setup_env.sh

echo 'TH2 ENVIRONMENT SETUP DONE'
