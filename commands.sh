#1 +20+a
#th2-pq5r-dot-data-prep-status-20261128-a01
set -u
date -u
prep=/mnt/local/_outputs/deep-llms_th2/dot_prep_20261128
data=/mnt/local/_data/deep-llms_th2/dot_data_20261128
tmux ls 2>&1 | head -5
cat "$prep/status.json" 2>&1 | grep -v '"raw/dclm\|"tokenizer/' | head -40
ls "$prep" 2>&1
tail -4 "$prep/driver.log" 2>&1
tail -6 "$prep/e2.log" 2>&1 | cut -c1-300
ls "$data/norm" "$data/norm/dot" 2>&1 | head -12
