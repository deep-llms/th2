#1 +60+a
#th2-ohmv-attention-shift-cpu-diagnose-20261122-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd /mnt/local/_outputs/deep-llms_th2/attention_shift_20261121_a01/source
export PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,socket
from pathlib import Path
import torch,numpy as np
from efficient_llm.warm_order.attention_shift_train import read
from efficient_llm.warm_order.adaptation_train import load_parent
from efficient_llm.warm_order.attention_shift import freeze,shifted,PATHS
from efficient_llm.warm_order.adaptation_runtime import measurement_runtime
from efficient_llm.warm_order.data import Rows,sha
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0'
base=Path('/mnt/local/_outputs/deep-llms_th2/attention_shift_20261121_a01')
assert not (base/'prepared/prepared.json').exists()
pins=read(base/'source.json')
for p,h in pins.items():assert sha(p)==h,p
c=read('resources/attention_shift_trial_20261121.json')['seed_panels']['0']
torch.set_num_threads(4);m,o,s,_=load_parent(c['parent'],c['parent_hashes']);del o,s
freeze(m);m.eval();rows=Rows(c['data'],m.c.length);ids=np.load(c['calibration_indices'])[:1];x=rows.batch(ids,'cpu')
zero=torch.zeros(m.c.d,requires_grad=True)
modules={'stem':m.stem,**{'middle.'+str(i):b for i,b in enumerate(m.middle)},'coda':m.coda,'norm':m.norm}
def trace(bias):
 captured={};handles=[]
 for name,module in modules.items():
  def save(mod,args,out,name=name):captured[name]=out.detach().clone()
  handles.append(module.register_forward_hook(save))
 try:
  with torch.no_grad(),measurement_runtime('cpu'):
   if bias is None:h=m.hidden(x,PATHS['C'])
   else:h=shifted(m,x,bias,'C',hidden=True)
  return h,captured
 finally:
  for handle in handles:handle.remove()
report={'row':int(ids[0]),'torch':torch.__version__,'trials':{}}
for threads in (4,1):
 torch.set_num_threads(threads)
 a,ta=trace(None);b,tb=trace(None);z,tz=trace(zero)
 report['trials'][str(threads)]={'native_repeat':float((a-b).abs().max()),'zero_delta':float((z-b).abs().max()),
  'layers':{k:{'repeat':float((ta[k]-tb[k]).abs().max()),'zero':float((tz[k]-tb[k]).abs().max()),'dtype':str(tb[k].dtype)} for k in tb}}
print('ATTENTION_SHIFT_CPU_DIAGNOSE',json.dumps(report),flush=True)
PY
