#1 +60+a
#th2-run-system-ls-check-20260826-222500
set -euo pipefail

echo '=== TH2 RUN-SYSTEM LS CHECK START ==='
date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
hostname
printf 'pwd=%s\n' "$PWD"
printf 'project=%s\n' "${PROJECT:-unset}"

echo '=== repository root listing ==='
ls -la

test -f commands.sh
echo 'TH2 RUN-SYSTEM LS CHECK OK'
