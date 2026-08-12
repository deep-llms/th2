#1 +120+a
#th2-probe-xling
O=/opt/dlami/nvme/sparse_emb_outputs
pgrep -af run_crosslingual || echo "no xling processes"
ls -la $O/crosslingual/ $O/crosslingual/*/ 2>/dev/null
echo '=== original_ant log tail ==='
tail -c 1200 $O/crosslingual/original_ant.log 2>/dev/null
echo '=== v2_attn log tail ==='
tail -c 1200 $O/crosslingual/v2_attn.log 2>/dev/null
