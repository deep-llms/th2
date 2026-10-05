#1 +60+a
#th2-tjx3-p1-alpha1-final-gates-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-p1-alpha1-2500-20261005-a01/supervised/run/training/seed-42/P1-block')
checkpoint=root/'checkpoint-2500'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert state['global_step']==2500
config=json.loads((root/'train_config.json').read_text())
assert config['pilot']['proxy_alpha_init']==1. and config['pilot']['arm']=='P1-block'
result=json.loads((root/'result.json').read_text())
assert result['global_step']==2500
rows=[];values=[]
with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as weights:
    names=sorted((n for n in weights.keys() if n.startswith('heads.') and n.endswith('.alpha')),key=lambda n:int(n.split('.')[1]))
    for name in names:
        gate=weights.get_tensor(name).float();assert torch.isfinite(gate).all()
        assert gate.shape==(1,1024)
        values.append(gate.flatten())
        rows.append(dict(layer=int(name.split('.')[1]),mean=gate.mean().item(),mean_abs=gate.abs().mean().item(),
            std=gate.std(unbiased=False).item(),min=gate.min().item(),max=gate.max().item(),
            mean_abs_change_from_init=(gate-1).abs().mean().item()))
assert len(rows)==12
all_values=torch.cat(values)
logged=next(row for row in reversed(state['log_history']) if row.get('step')==2500 and 'loss' in row)
print('FINAL_GATES',json.dumps(dict(arm='P1-block',step=2500,alpha_init=1.,source=str(checkpoint),
    layers=rows,overall=dict(mean=all_values.mean().item(),mean_abs=all_values.abs().mean().item(),
        min=all_values.min().item(),max=all_values.max().item(),std=all_values.std(unbiased=False).item()),
    logged_gates={k:v for k,v in logged.items() if 'alpha' in k},
    evaluation=result['evaluation'])),flush=True)
PY
