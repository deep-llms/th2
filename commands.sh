#1 +240+a
#th2-ohmv-backup-discover-20261126-a01
set -u
date -u
# Read-only: find the persistent backup folder under /mnt and size what needs backing up before the B200 restart.
echo '== /mnt'; ls -la /mnt/
for d in /mnt/*/; do echo "== $d"; ls -la "$d" 2>/dev/null | head -25; done
echo '== backup dirs'; find /mnt -maxdepth 4 -iname '*backup*' -type d 2>/dev/null | head -20
echo '== mounts'; df -hT /mnt /mnt/* 2>/dev/null; findmnt -lo TARGET,SOURCE,FSTYPE,SIZE,AVAIL 2>/dev/null | grep -E '^/mnt|^/ ' | head -20
echo '== outputs size'; du -sh /mnt/local/_outputs/deep-llms_th2 2>/dev/null
du -sh /mnt/local/_outputs/deep-llms_th2/* 2>/dev/null | sort -h | tail -60
echo '== checkpoints'; find /mnt/local/_outputs/deep-llms_th2 -name '*.pt' -size +1M 2>/dev/null | wc -l
find /mnt/local/_outputs/deep-llms_th2 -name '*.pt' -size +1M -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.1f GB\n", s/1e9}'
find /mnt/local/_outputs/deep-llms_th2 -path '*backup*' -name '*.pt' -printf '%s\n' 2>/dev/null | awk '{s+=$1; n++} END {printf "in-run backup copies: %d files %.1f GB\n", n, s/1e9}'
echo '== other local trees'; du -sh /mnt/local/* 2>/dev/null | sort -h | tail -20
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | head -8
