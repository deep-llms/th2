#1 +30+a
#th2-6y3o-dot-C1-launch-20261022-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
SUMMARY_CODE="import json,math,sys\nfrom pathlib import Path\npred,root=map(Path,sys.argv[1:])\nread=lambda p:json.loads(p.read_text())\nrun=root/'supervised/run'\nc=read(run/'C1-s0/result.json')\nv=read(pred/'supervised/run/V1-s0/result.json')\nd=read(pred.parent/'6y3o-dot-pilot-20261020-a01/prepared/runs/D/final_eval.json')\nverified=read(run/'C1-s0.verified.json')\nfinal=read(root/'prepared/runs/C1-s0/final_eval.json')\nassert c['status']==v['status']==verified['status']=='ok'\nassert c['global_step']==v['global_step']==verified['global_step']==4768\nassert (c['num_tokens'],c['num_sequences'])==(d['eval_num_tokens'],d['eval_num_sequences'])==(9915698,9704)\nfor key in ['num_tokens','num_sequences','selection','data_recipe','attention_backend','precision','world_size']:\n    assert c[key]==v[key],key\nassert abs(c['loss']-final['eval_loss'])<1e-5\nassert math.isfinite(c['loss']) and math.isfinite(c['perplexity'])\nassert abs(c['perplexity']-math.exp(c['loss']))<1e-8\nfor name in ['C2-s0','V1-s0']:\n    assert read(root/'prepared/runs/C1-s0/data_recipe.json')==read(pred/'prepared/runs'/name/'data_recipe.json')\nresult={'status':'ok','step':4768,'num_sequences':c['num_sequences'],'num_tokens':c['num_tokens'],\n    'C1':{'loss':c['loss'],'ppl':c['perplexity']},\n    'V1':{'loss':v['loss'],'ppl':v['perplexity']},\n    'D':{'loss':d['eval_loss'],'ppl':d['eval_perplexity']},\n    'V1_gain_over_C1':c['loss']-v['loss'],\n    'V1_beats_C1_by_001':c['loss']-v['loss']>=.01,\n    'C1_gap_to_D':c['loss']-d['eval_loss'],\n    'scope':'Seed 0 conditional control only; does not authorize second seeds or new architectures.'}\nwith (run/'C1-summary.json').open('x') as f:json.dump(result,f,indent=2)\nprint('C1_COMPARISON',json.dumps(result),flush=True)\n"
import copy,json
from pathlib import Path
from bdot.routing import RoutingConfig
from run_experiments import load_jobs

def prepare(pred,root):
    read=lambda p:json.loads(p.read_text())
    old_prepared=pred/'prepared'
    prepared=root/'prepared'
    old=read(old_prepared/'C2-s0.train.json')
    model=read(old_prepared/'C2-s0.model.json')
    cfg=RoutingConfig(**model['bdot'])
    assert (cfg.routing,cfg.reuse,cfg.r,cfg.curriculum,cfg.placement_film,cfg.seed,cfg.expert_backend)==('random',False,1,'none',True,0,'grouped')
    assert (old['max_steps'],old['warmup_steps'],old['block_size'],old['seed'],old['data_seed'])==(4768,96,1024,0,0)
    assert (old['per_device_train_batch_size'],old['gradient_accumulation_steps'],old['attention_backend'])==(16,4,'fa4')
    assert old.get('stop_after') is None
    new_model=copy.deepcopy(model)
    new_model['bdot']['placement_film']=False
    RoutingConfig(**new_model['bdot'])
    new=copy.deepcopy(old)
    new['config_name']=str(prepared/'C1-s0.model.json')
    new['output_dir']=str(prepared/'runs/C1-s0')
    # Copy the proven C2 jobs verbatim except experiment-owned names/paths.
    source=read(pred/'gated-jobs.json')['jobs']
    jobs=[copy.deepcopy(j) for j in source if j['name'] in ('C2-s0','C2-s0-eval')]
    assert [j['name'] for j in jobs]==['C2-s0','C2-s0-eval']
    def replace(x):
        if isinstance(x,str):return x.replace(str(old_prepared),str(prepared)).replace('C2-s0','C1-s0')
        if isinstance(x,list):return [replace(v) for v in x]
        if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
        return x
    jobs=replace(jobs)
    assert all(j['gpus']==list(range(8)) for j in jobs)
    assert jobs[0]['argv'][-1]==str(prepared/'C1-s0.train.json')
    assert jobs[1]['argv'][jobs[1]['argv'].index('--checkpoint')+1]==str(prepared/'runs/C1-s0/checkpoint-4768')
    assert str(pred) not in json.dumps(jobs)
    summary_code=SUMMARY_CODE
    jobs.append(dict(name='C1-report',timeout_seconds=300,
        argv=['{python}','-c',summary_code,str(pred),str(root)],
        required_outputs=[dict(path='C1-summary.json',json_equals=dict(status='ok'))]))
    prepared.mkdir(parents=True,exist_ok=False)
    for name,obj in [('C1-s0.train.json',new),('C1-s0.model.json',new_model),('jobs.json',{'jobs':jobs})]:
        with (prepared/name).open('x') as f:json.dump(obj,f,indent=2)
    load_jobs(prepared/'jobs.json')
    contract={'status':'ok','control':'C2-s0 with placement_film=False',
        'model_changes':{'placement_film':[True,False]},
        'training_changes':{k:[old[k],new[k]] for k in old if old[k]!=new[k]},
        'source_prepared':str(old_prepared),'jobs':[j['name'] for j in jobs]}
    assert set(contract['training_changes'])=={'config_name','output_dir'}
    with (root/'C1-contract.json').open('x') as f:json.dump(contract,f,indent=2)
    return prepared/'jobs.json'

import os,subprocess,sys
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect
from scripts.train_then_burn import digest,burn_progress
base=Path('/mnt/local/_outputs/deep-llms_th2')
pred=base/'6y3o-dot-v3-20261021-a01'
root=base/'6y3o-dot-C1-20261022-a01'
initial=json.loads((root/'initial-preflight.json').read_text())
assert initial['status']=='ok' and initial['live']['host']=='thiennh-p6-6y3o-worker-0'
for name,expected in initial['training_code_sha256'].items():
    assert digest(name)==expected,name
burn_gate(pred)
assert burn_progress((pred/'supervised/burn.log').read_text())
jobs=prepare(pred,root)
burn_gate(pred)
live=inspect()
assert live['host']=='thiennh-p6-6y3o-worker-0'
assert len(live['gpus'])==8 and all('B200' in g['name'] for g in live['gpus'])
with (root/'preflight.json').open('x') as f:json.dump(live,f,indent=2)
subprocess.run([sys.executable,'scripts/bdot_b200_launch.py',
    '--root',str(root),'--host',live['host'],'--session','dot_C1_20261022_a01',
    '--jobs-config',str(jobs)],check=True,
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
print('C1_QUEUE_SUBMITTED',flush=True)
PY
