#1 +20+a
#th2-6y3o-a2-checkpoint-size-audit-20261031-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python3 -u - <<'PYCODE'
import os,json,socket,struct
from pathlib import Path
from collections import defaultdict
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
roots=sorted(Path('/mnt/local/_outputs/deep-llms_th2').glob('6y3o-a2-*'))+[Path('/mnt/local/_models/deep-llms_th2/a2')]
groups=defaultdict(lambda:dict(bytes=0,files=0));special=[]
for root in roots:
 for directory,dirs,files in os.walk(root,followlinks=False):
  for name in files:
   p=Path(directory)/name
   if p.is_symlink():print('SYMLINK',str(p));continue
   size=p.stat().st_size;n=name.lower()
   if 'optimizer' in n or 'scheduler' in n or n.startswith('rng_state'):kind='training_state'
   elif p.suffix=='.safetensors':kind='model_weights'
   elif name=='first_gradients.pt':kind='profiling_gradients'
   elif p.suffix in ('.pt','.pth') or name=='pytorch_model.bin':kind='other_tensor'
   elif size>=10000000:kind='other_large'
   else:kind='small_metadata_code_results'
   groups[kind]['bytes']+=size;groups[kind]['files']+=1
   if kind not in ('model_weights','small_metadata_code_results'):special.append(dict(path=str(p),bytes=size,kind=kind))
   if kind=='model_weights':
    with p.open('rb') as f:
     nhead=struct.unpack('<Q',f.read(8))[0];assert nhead<10000000;header=json.loads(f.read(nhead))
    print('MODEL_FILE',json.dumps(dict(path=str(p),bytes=size,dtypes=sorted({v['dtype'] for k,v in header.items() if k!='__metadata__'}))),flush=True)
print('FILE_GROUPS',json.dumps(dict(groups)),flush=True)
print('NONMODEL_LARGE_AND_STATE',json.dumps(special),flush=True)
print('CHECKPOINT_SIZE_AUDIT_COMPLETE',flush=True)
PYCODE
