#1 +30+a
#th2-reverify-project-code-cleanup-20260907-a01
set -euo pipefail
date -u
hostname

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_ARCHIVE_DIR=/mnt/local/_project_archives/@PROJECT@/before_stagewise_20260907_a01
test "$TASK_PROJECT_DIR" = /mnt/local/deep-llms_th2
test "$(pwd -P)" = "$TASK_PROJECT_DIR"
test -d "$TASK_PROJECT_DIR" && test ! -L "$TASK_PROJECT_DIR"
test -f "$TASK_PROJECT_DIR/commands.sh"
printf '\nCurrent project directory:\n'
ls -lah -- "$TASK_PROJECT_DIR"

shopt -s dotglob nullglob
for TASK_PATH in "$TASK_PROJECT_DIR"/*; do
    case "${TASK_PATH##*/}" in
        commands.sh|_run_log_|_previous_run_status.log|_dl_active.txt) ;;
        *) printf 'FAIL: unexpected project entry: %s\n' "$TASK_PATH" >&2; exit 1 ;;
    esac
done

TASK_OLD_NAMES=(
    .gitignore compositional crosslingual docs dummy.py eval eval.txt
    experiments.log finetune prepare_data.py requirements.txt resources
    run_experiments.py scripts sparse_emb.txt temp test_prepare_data.py
    test_run_experiments.py test_token_importance_quota.py train.py
    train_compositional.py train_original_ant.py wandb
)
test -d "$TASK_ARCHIVE_DIR" && test ! -L "$TASK_ARCHIVE_DIR"
printf '\nConfirm every old entry is archived, not in the project folder:\n'
for TASK_NAME in "${TASK_OLD_NAMES[@]}"; do
    test ! -e "$TASK_PROJECT_DIR/$TASK_NAME" && test ! -L "$TASK_PROJECT_DIR/$TASK_NAME"
    test -e "$TASK_ARCHIVE_DIR/$TASK_NAME" || test -L "$TASK_ARCHIVE_DIR/$TASK_NAME"
    printf 'VERIFIED: %s\n' "$TASK_NAME"
done
printf '%s  %s\n' \
    e436125c2f7d61fc2e205321d0635f5fa49c5e60c8aaf09b081d5bd5f058db30 "$TASK_ARCHIVE_DIR/train_compositional.py" \
    846965684f8914533258628708afaece160da52a0c18b14f7acb2b5fb521b27f "$TASK_ARCHIVE_DIR/run_experiments.py" \
    | sha256sum --check --strict

printf '\nExternal data/model/output/environment directories still present:\n'
for TASK_PATH in \
    /mnt/local/_data/@PROJECT@ \
    /mnt/local/_models/@PROJECT@ \
    /mnt/local/_outputs/@PROJECT@ \
    /mnt/local/conda-py311/envs/sparse_emb \
    /mnt/local/conda-py311/envs/eval; do
    test -d "$TASK_PATH"
    ls -ld -- "$TASK_PATH"
done
printf '\nRead-only GPU usage snapshot:\n'
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

printf '\nTH2 PROJECT CODE CLEANUP REVERIFIED\n'
