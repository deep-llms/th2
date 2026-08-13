#1 +120+a
#th2-list-run-logs
echo '=== all Aug 13 run logs ==='
ls -la _run_log_/_run-2026-08-13* 2>/dev/null
echo '=== any eval log? ==='
ls -la _run_log_/*eval* 2>/dev/null || echo "none"
echo '=== last 3 run logs tail ==='
for f in $(ls -t _run_log_/*.log 2>/dev/null | head -3); do
    echo "--- $f ($(wc -c < $f) bytes) ---"
    tail -30 "$f"
done
