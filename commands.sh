#0
#th2-idle-after-old-project-code-cleanup-20260907-a01
set -euo pipefail

date -u
hostname
TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_ARCHIVE_BASE=/mnt/local/_project_archives
TASK_ARCHIVE_PARENT="$TASK_ARCHIVE_BASE/@PROJECT@"
TASK_ARCHIVE_DIR="$TASK_ARCHIVE_PARENT/before_stagewise_20260907_a01"

# Hard guards: this is ONLY the verified old th2 code folder, not /mnt/local,
# data, model, output, environment, or /tmp directories. Never signal processes.
test "$TASK_PROJECT_DIR" = /mnt/local/deep-llms_th2
test -d "$TASK_PROJECT_DIR" && test ! -L "$TASK_PROJECT_DIR"
test "$(readlink -f -- "$TASK_PROJECT_DIR")" = "$TASK_PROJECT_DIR"
test "$(pwd -P)" = "$TASK_PROJECT_DIR"
test -f "$TASK_PROJECT_DIR/commands.sh" && test ! -L "$TASK_PROJECT_DIR/commands.sh"
for TASK_PATH in "$TASK_ARCHIVE_BASE" "$TASK_ARCHIVE_PARENT" "$TASK_ARCHIVE_DIR"; do
    test ! -L "$TASK_PATH"
done
test ! -e "$TASK_ARCHIVE_DIR"

# Confirm the project is the old deployment before moving ANYTHING.
printf '%s  %s\n' \
    e436125c2f7d61fc2e205321d0635f5fa49c5e60c8aaf09b081d5bd5f058db30 "$TASK_PROJECT_DIR/train_compositional.py" \
    846965684f8914533258628708afaece160da52a0c18b14f7acb2b5fb521b27f "$TASK_PROJECT_DIR/run_experiments.py" \
    | sha256sum --check --strict

# This explicit inventory comes from the preceding read-only B200 listing.
TASK_OLD_NAMES=(
    .gitignore compositional crosslingual docs dummy.py eval eval.txt
    experiments.log finetune prepare_data.py requirements.txt resources
    run_experiments.py scripts sparse_emb.txt temp test_prepare_data.py
    test_run_experiments.py test_token_importance_quota.py train.py
    train_compositional.py train_original_ant.py wandb
)
declare -A TASK_OLD_SET=()
for TASK_NAME in "${TASK_OLD_NAMES[@]}"; do TASK_OLD_SET["$TASK_NAME"]=1; done

# Preserve the active log directory and runner state; deleting the log pathname
# could prevent the controller from retrieving this very cleanup result.
shopt -s dotglob nullglob
for TASK_PATH in "$TASK_PROJECT_DIR"/*; do
    TASK_NAME=${TASK_PATH##*/}
    case "$TASK_NAME" in
        commands.sh|_run_log_|_previous_run_status.log|_dl_active.txt) continue ;;
    esac
    if [ "${TASK_OLD_SET[$TASK_NAME]:-}" != 1 ]; then
        printf 'REFUSE: unexpected project entry: %s\n' "$TASK_PATH" >&2
        exit 1
    fi
done

mkdir -p -- "$TASK_ARCHIVE_PARENT"
test "$(readlink -f -- "$TASK_ARCHIVE_PARENT")" = "$TASK_ARCHIVE_PARENT"
mkdir -- "$TASK_ARCHIVE_DIR"
printf 'Recoverable archive: %s\n' "$TASK_ARCHIVE_DIR"
for TASK_NAME in "${TASK_OLD_NAMES[@]}"; do
    TASK_SOURCE="$TASK_PROJECT_DIR/$TASK_NAME"
    TASK_DEST="$TASK_ARCHIVE_DIR/$TASK_NAME"
    if [ -e "$TASK_SOURCE" ] || [ -L "$TASK_SOURCE" ]; then
        test ! -e "$TASK_DEST" && test ! -L "$TASK_DEST"
        printf 'ARCHIVE: %s\n' "$TASK_NAME"
        mv --no-clobber --no-target-directory -- "$TASK_SOURCE" "$TASK_DEST"
        test ! -e "$TASK_SOURCE" && test ! -L "$TASK_SOURCE"
        test -e "$TASK_DEST" || test -L "$TASK_DEST"
    fi
done

printf '\nRemaining project entries:\n'
ls -lah -- "$TASK_PROJECT_DIR"
for TASK_PATH in "$TASK_PROJECT_DIR"/*; do
    case "${TASK_PATH##*/}" in
        commands.sh|_run_log_|_previous_run_status.log|_dl_active.txt) ;;
        *) printf 'ERROR: unexpected remaining entry: %s\n' "$TASK_PATH" >&2; exit 1 ;;
    esac
done
printf '\nArchived entries:\n'
ls -lah -- "$TASK_ARCHIVE_DIR"
printf '\nTH2 OLD PROJECT CODE CLEANUP COMPLETE\n'
printf 'Kept commands.sh and runner-owned logs/state; old files are recoverable at %s\n' "$TASK_ARCHIVE_DIR"
