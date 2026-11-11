#1 +20+a
#th2-wc69-backup-inventory-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
date -u
hostname
df -B1 /mnt/local/_backup/deep-llms_th2 /mnt/local/_outputs/deep-llms_th2
findmnt -T /mnt/local/_backup/deep-llms_th2
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CODE'
import os,json
from pathlib import Path
out=Path('/mnt/local/_outputs/deep-llms_th2');bak=Path('/mnt/local/_backup/deep-llms_th2')
assert out.is_dir() and bak.is_dir()
for root in (out,bak):
 print('ROOT',str(root),flush=True)
 for top in sorted(root.iterdir()):
  if top.is_symlink():print('LINK',str(top),os.readlink(top));continue
  count=size=keep=keepbytes=weights=0;links=[]
  if not top.is_dir():continue
  for d,ds,fs in os.walk(top,followlinks=False):
   for n in fs:
    p=Path(d)/n
    if p.is_symlink():links.append(str(p));continue
    s=p.stat().st_size;count+=1;size+=s
    parts=p.relative_to(top).parts
    skipped=any(x=='trainer' or x.startswith('checkpoint-') for x in parts)
    if not skipped:keep+=1;keepbytes+=s
    if n.endswith(('.safetensors','.bin','.pt')) and not skipped:weights+=s
  print('TREE',json.dumps(dict(name=top.name,files=count,bytes=size,keep_files=keep,keep_bytes=keepbytes,weight_bytes=weights,backup_exists=(bak/top.name).exists(),links=links[:5],link_count=len(links))),flush=True)
print('BACKUP_INVENTORY_COMPLETE',flush=True)
CODE
