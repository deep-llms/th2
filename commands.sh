#1 +60+a
#th2-clean-failed-culturax-download-20260822
set -euo pipefail

echo '=== remove failed CulturaX download only ==='
date -u
hostname

TASK_DATA_DIR="/mnt/local/_data/@PROJECT@/data"
case "$TASK_DATA_DIR" in
    /mnt/local/_data/*/data) ;;
    *)
        echo "REFUSE: unexpected data path: $TASK_DATA_DIR"
        exit 1
        ;;
esac

if [ -L "$TASK_DATA_DIR" ]; then
    echo "REFUSE: data path is a symlink: $TASK_DATA_DIR"
    exit 1
fi

if [ -e "$TASK_DATA_DIR" ]; then
    test -d "$TASK_DATA_DIR"
    echo "deleting=$TASK_DATA_DIR"
    du -sh "$TASK_DATA_DIR"
    rm -rf -- "$TASK_DATA_DIR"
else
    echo "already_absent=$TASK_DATA_DIR"
fi

test ! -e "$TASK_DATA_DIR"
echo 'TH2 FAILED CULTURAX DATA CLEANUP OK'
