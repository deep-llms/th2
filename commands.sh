#1 +30+a
#th2-commands-only-sync-probe-20260907-a01
set -euo pipefail

date -u
hostname
printf 'Runner working directory: '
pwd
TASK_PROJECT_DIR=/mnt/local/@PROJECT@
test -d "$TASK_PROJECT_DIR"

printf '\nProject code directory: %s\n' "$TASK_PROJECT_DIR"
ls -lah -- "$TASK_PROJECT_DIR"
printf '\nProject entries (two levels; excluding Git internals):\n'
find "$TASK_PROJECT_DIR" -mindepth 1 -maxdepth 2 \
    -not -path "$TASK_PROJECT_DIR/.git" \
    -not -path "$TASK_PROJECT_DIR/.git/*" -printf '%y %P\n' | sort

printf '\nOld sparse-embedding paths after synchronization:\n'
for TASK_OLD_PATH in prepare_data.py train.py train_compositional.py \
    run_experiments.py compositional eval finetune scripts resources docs; do
    if [ -e "$TASK_PROJECT_DIR/$TASK_OLD_PATH" ] || [ -L "$TASK_PROJECT_DIR/$TASK_OLD_PATH" ]; then
        printf 'STILL PRESENT: %s\n' "$TASK_OLD_PATH"
    else
        printf 'ABSENT: %s\n' "$TASK_OLD_PATH"
    fi
done

printf '\nTop-level entries other than commands.sh and .git:\n'
find "$TASK_PROJECT_DIR" -mindepth 1 -maxdepth 1 \
    -not -name commands.sh -not -name .git -printf '%f\n' | sort
printf '\nTH2 COMMANDS-ONLY SYNC PROBE COMPLETE\n'
