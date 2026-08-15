#1 +120+a
#th2-dump-ft-results
echo '=== summary.md ==='
cat /opt/dlami/nvme/sparse_emb_outputs/finetune/summary.md
echo '=== all result JSONs ==='
for f in /opt/dlami/nvme/sparse_emb_outputs/finetune/*.json; do
    echo "===JSON $(basename $f)==="
    cat "$f"
    echo
done
