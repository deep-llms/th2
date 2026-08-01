#1
#check-disks
echo "=== Disk usage ==="
df -h | grep -v tmpfs | grep -v snap
echo ""

echo "=== Mount points ==="
lsblk -o NAME,SIZE,MOUNTPOINT,FSTYPE | grep -v loop
echo ""

echo "=== /opt/ ==="
ls -la /opt/ 2>/dev/null || echo "/opt not found"
echo ""

echo "=== /opt/dlami/ ==="
ls -la /opt/dlami/ 2>/dev/null || echo "/opt/dlami not found"
echo ""

echo "=== /opt/dlami/nvme/ ==="
ls -la /opt/dlami/nvme/ 2>/dev/null || echo "/opt/dlami/nvme not found"
echo ""

echo "=== Home dir ==="
df -h ~ | tail -1
ls ~ | head -20
