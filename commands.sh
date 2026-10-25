#1 +60+a
#th2-6y3o-partial-router-full-validation-20261025-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import hashlib,json,math,os,shutil,socket,subprocess,sys
from pathlib import Path

base=Path('/mnt/local/_outputs/deep-llms_th2')
previous=base/'6y3o-dot-stop-optimized-20261025-a03'
supervisor_source=previous/'source'
router=base/'6y3o-dot-router-20261022-a01'
source=router/'source'
root=base/'6y3o-dot-partial-route-full-20261025-a01'
host='thiennh-p6-6y3o-worker-0'
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
assert socket.gethostname()==host
assert not root.exists()
assert sha(source/'SOURCE_MANIFEST.json')=='65afe4d86c3e154638d512f2e63c38303d9659b57ef1c4ebf96ff7e857c096c9'
for tree in (source,supervisor_source):
    for name,expected in read(tree/'SOURCE_MANIFEST.json')['files'].items():
        p=tree/name
        assert not p.is_symlink() and p.resolve().is_relative_to(tree.resolve()) and sha(p)==expected,name
sys.path.insert(0,str(supervisor_source))
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect
from scripts.train_then_burn import burn_progress
from run_experiments import load_jobs
burn_gate(previous)
assert read(previous/'supervised/run/complete.json')['status']=='ok'
assert burn_progress((previous/'supervised/burn.log').read_text())
root.mkdir()
write(root/'preflight-initial.json',inspect())
print('EVALUATION_ONLY_PREFLIGHT '+json.dumps(read(root/'preflight-initial.json')),flush=True)

cases=[
 ('Partial-25-cosine-s0-4768',router,'Partial-25-cosine-s0',4768,0,3.495407258695272,0.1444795739031585),
 ('Partial-25-cosine-s1-4768',base/'6y3o-dot-router-confirm-20261023-a01','Partial-25-cosine-s1',4768,1,3.4965066446610065,0.12365575477493172),
 ('Partial-25-cosine-s0-10000',base/'6y3o-dot-long-pair-20261023-a01','Partial-25-cosine-s0',10000,0,3.2970712571608005,0.202274218729873),
 ('Partial-50-cosine-detach-s0-4768',router,'Partial-50-cosine-detach-s0',4768,0,3.4972656017974293,0.08886740134947572),
]
prepared=root/'prepared';prepared.mkdir()
manifest=[];jobs=[]
for name,parent,run_name,step,seed,loss,off in cases:
    recipe=read(parent/'prepared'/(run_name+'.train.json'))
    cp=parent/'prepared/runs'/run_name/f'checkpoint-{step}'
    assert Path(recipe['output_dir'])==cp.parent
    assert recipe['seed']==seed and recipe['data_seed']==0 and recipe['block_size']==1024
    assert read(cp/'trainer_state.json')['global_step']==step
    options=read(cp/'config.json')['bdot']
    assert options['routing']=='learned' and options['router_metric']=='cosine'
    assert options['routed_mlp_fraction']==(.5 if name.startswith('Partial-50') else .25)
    assert options['router_detach_input']==name.startswith('Partial-50')
    assert options['S']==8 and options['route_scope']=='mlp' and options['r']==1 and options['reuse']
    old=read(cp.parent/'final_eval.json')
    assert abs(old['eval_loss']-loss)<1e-12 and old['eval_off_default_share']==off
    checkpoint_files=[*cp.glob('*.safetensors'),*cp.glob('pytorch_model*.bin')]
    assert checkpoint_files
    checkpoint_files += [cp/x for x in ('config.json','trainer_state.json','data_recipe.json','training_recipe.json')]
    before={p.name:sha(p) for p in checkpoint_files}
    config=dict(checkpoint=str(cp),eval_data_dir=recipe['eval_data_dir'],tokenizer_name=recipe['tokenizer_name'],
        output_file=str(root/'supervised/run'/name/'result.json'),attention_backend='fa4',precision='bf16',batch_size=16,
        data_cache_dir=recipe.get('data_cache_dir'),route_study=True,route_sequences=9704,route_artifact_sequences=256,
        routing_seed=seed,max_sequences=None,pilot_diagnostics=False,router_gradient_check=False,overwrite_cache=False)
    cfg=prepared/(name+'.eval.json');write(cfg,config)
    manifest.append(dict(name=name,checkpoint=str(cp),step=step,seed=seed,reference_loss=loss,reference_off_default=off,
                         checkpoint_sha256=before,original_result=str(cp.parent/'final_eval.json')))
    jobs.append(dict(name=name,gpus=list(range(8)),timeout_seconds=1800,
        argv=['{python}','-m','accelerate.commands.launch','--config_file',str(source/'resources/accelerate_b200.yaml'),
              '--num_processes','8','--main_process_port','0',str(source/'eval/packed_ppl.py'),str(cfg)],
        required_outputs=[dict(path=name+'/result.json',json_equals=dict(status='ok',global_step=step,num_sequences=9704,num_tokens=9915698,world_size=8))]))
write(root/'manifest.json',dict(scope='Evaluation only: four unchanged checkpoints, full held-out routing diagnostics',
    source=str(source),supervisor_source=str(supervisor_source),predecessor=str(previous),cases=manifest,
    source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),supervisor_manifest_sha256=sha(supervisor_source/'SOURCE_MANIFEST.json')))

validator=r'''
import hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=read(root/'manifest.json');results={}
for case in manifest['cases']:
    cp=Path(case['checkpoint'])
    assert all(sha(cp/name)==value for name,value in case['checkpoint_sha256'].items()),'Checkpoint changed'
    path=root/'supervised/run'/case['name']/'result.json'
    r=read(path);s=r['route_study'];d=s['route_diagnostics']
    assert r['status']=='ok' and r['global_step']==case['step'] and Path(r['checkpoint'])==cp
    assert r['selection']['max_sequences'] is None and r['selection']['full_sequences']==9704
    assert r['world_size']==8 and r['batch_size_per_rank']==16 and r['routing_seed']==case['seed']
    assert r['attention_backend']=='fa4' and r['precision']=='bf16'
    assert abs(r['loss']-case['reference_loss'])<1e-5
    assert abs(r['off_default_share']-case['reference_off_default'])<1e-12
    assert s['diagnostic_sequences']==9704 and d['num_sequences']==9704
    assert d['input_tokens']==9936896 and d['routing_decisions']==79495168
    counts=d['module_counts_by_step'];assert len(counts)==8 and all(len(row)==8 for row in counts)
    assert all(sum(row)==9936896 for row in counts)
    assert all(counts[i][j]==0 for i in range(8) for j in range(8) if abs(i-j)>1)
    loads=[sum(row[j] for row in counts) for j in range(8)]
    off=1-sum(counts[i][i] for i in range(8))/79495168
    assert abs(off-r['off_default_share'])<1e-12 and abs(d['off_default_share']-off)<1e-12
    assert all(abs(loads[j]/79495168-d['module_load_share'][j])<1e-12 for j in range(8))
    for policy,v in s['interventions'].items():
        assert (v['num_tokens'],v['num_sequences'])==(9915698,9704)
        assert math.isfinite(v['loss']) and abs(math.exp(v['loss'])-v['perplexity'])<1e-8
    assert s['interventions']['fixed']['off_default_share']==0
    results[case['name']]=dict(checkpoint=str(cp),step=case['step'],seed=case['seed'],
        off_default_share=r['off_default_share'],interventions=s['interventions'],usage=d,module_load_counts=loads,
        unchanged_checkpoint_sha256=case['checkpoint_sha256'],result_sha256=sha(path),
        loss_replay_delta=r['loss']-case['reference_loss'],
        data_recipe=r['data_recipe'],training_recipe_sha256=r['training_recipe_sha256'],
        selection=r['selection'],world_size=8,batch_size_per_rank=16,routing_seed=case['seed'],
        evaluator_sha256=r['evaluator_sha256'],exporter_sha256=s['exporter_sha256'])
for tree in (Path(manifest['source']),Path(manifest['supervisor_source'])):
    assert all(sha(tree/k)==v for k,v in read(tree/'SOURCE_MANIFEST.json')['files'].items())
output=dict(status='ok',scope=manifest['scope'],source=manifest['source'],results=results,
    checkpoint_files_unchanged=True,source_unchanged=True,
    counts=dict(sequences=9704,scored_targets=9915698,routing_input_tokens=9936896,routing_decisions=79495168))
with (root/'supervised/run/summary.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
print('FULL_ROUTE_DIAGNOSTICS_COMPLETE '+json.dumps({k:dict(own=v['interventions']['as_trained']['loss'],fixed=v['interventions']['fixed']['loss'],random=v['interventions']['random']['loss'],off=v['off_default_share']) for k,v in results.items()}),flush=True)
'''
driver=root/'validate.py';driver.write_text(validator)
compile(validator,str(driver),'exec')
jobs.append(dict(name='validate',timeout_seconds=600,argv=['{python}',str(driver),str(root)],
    required_outputs=[dict(path='summary.json',json_equals=dict(status='ok',checkpoint_files_unchanged=True,source_unchanged=True))]))
write(prepared/'jobs.json',dict(jobs=jobs));load_jobs(prepared/'jobs.json')
burn_gate(previous)
write(root/'preflight.json',inspect())
subprocess.run([sys.executable,str(supervisor_source/'scripts/bdot_b200_launch.py'),
    '--root',str(root),'--host',host,'--session','partial_route_full_20261025_a01',
    '--jobs-config',str(prepared/'jobs.json'),
    '--predecessor-burn-script',str(supervisor_source/'resources/llm_pretrain_burn.py')],
    cwd=supervisor_source,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
print('EVAL_ONLY_QUEUE_SUBMITTED '+str(root),flush=True)
REMOTE
