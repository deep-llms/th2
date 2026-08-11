#1 +120+a
#th2-probe-anchor-results
O=/opt/dlami/nvme/sparse_emb_outputs
pgrep -af analyze_anchor || echo "no analysis process"
ls -la $O/original_ant/checkpoint-10000/anchor_usage* $O/v2_attn/checkpoint-10000/anchor_usage* 2>/dev/null || echo "no outputs yet"
echo '=== original_ant json ==='
cat $O/original_ant/checkpoint-10000/anchor_usage.json 2>/dev/null
echo '=== v2_attn json ==='
cat $O/v2_attn/checkpoint-10000/anchor_usage.json 2>/dev/null
