#1 +120+a
#th2-probe-bytoken-progress
O=/opt/dlami/nvme/sparse_emb_outputs
echo '=== procs ==='
pgrep -af "ppl_bytoken" || echo none
echo '=== gpu ==='
nvidia-smi | grep -E "MiB /" | head -4
echo '=== original_ant log tail ==='
tail -c 1500 $O/original_ant/checkpoint-10000/ppl_bytoken.log
echo
echo '=== v2_attn log tail ==='
tail -c 1500 $O/v2_attn/checkpoint-10000/ppl_bytoken.log
