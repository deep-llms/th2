#1 +60+a
#th2-ohmv-attention-shift-cpu-diagnose-20261122-a02
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
from efficient_llm.warm_order.attention_shift_node import references
from efficient_llm.warm_order.attention_shift_eval import load_means
from efficient_llm.warm_order.attention_shift import fingerprint
import inspect
means=load_means(c['mean_reference'],c,m.c.d);before=fingerprint(m)
original_assert=torch.testing.assert_close;checks=[]
def traced_assert(a,b,**kw):
 frame=inspect.currentframe().f_back
 record=dict(branch=frame.f_locals.get('branch'),name=frame.f_locals.get('name'),max_abs=float((a-b).abs().max()),kwargs=kw)
 value=frame.f_locals.get('b')
 if isinstance(value,torch.Tensor):record['bias_max_abs']=float(value.detach().abs().max())
 checks.append(record);print('SHIFT_REFERENCE_CHECK',json.dumps(record),flush=True)
 return original_assert(a,b,**kw)
torch.testing.assert_close=traced_assert
error=None
try:
 with measurement_runtime('cpu'):ref=references(m,x,means)
except Exception as e:error=str(e)
finally:torch.testing.assert_close=original_assert
report=dict(checks=checks,reference_error=error,body_unchanged=before==fingerprint(m),post_checks={})
for branch,path in PATHS.items():
 with torch.no_grad(),measurement_runtime('cpu'):
  a=m.hidden(x,path);b=shifted(m,x,torch.zeros(m.c.d),branch,hidden=True);d=m.hidden(x,path)
 report['post_checks'][branch]=dict(zero=float((a-b).abs().max()),repeat=float((a-d).abs().max()))
print('ATTENTION_SHIFT_CPU_DIAGNOSE_SEQUENCE',json.dumps(report),flush=True)
PY
