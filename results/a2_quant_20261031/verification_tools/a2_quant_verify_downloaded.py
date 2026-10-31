"""Audit downloaded frozen artifacts and independently recompute paired statistics."""
import argparse,hashlib,json,math,re,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from eval.a2_request_bridge import digest
from scripts.a2_quant_run import downstream_statistics
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r=a.root;b=r/'results/a2_quant_20261031'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(b/'config/plan.json');done=read(b/'complete.json');tests=read(b/'tests.json');down=read(b/'downstream.json')
assert cfg['spec_sha256']==sha(Path('docs/A2_quant_spec.md'))==done['spec_sha256']
assert sha(b/'decision_quant.json')==done['decision']['sha256'] and sha(b/'tests.json')==done['tests']['sha256']
sm=read(r/'source-manifest.json');sv=read(r/'verification/source_verification.json')
assert sm['code_commit']==cfg['code_commit'] and sv['code_commit']==sm['code_commit'] and sv['changed']==[] and sv['count']==len(sm['files'])==144
for name,h in sm['files'].items():assert hashlib.sha256(subprocess.check_output(['git','show',sm['code_commit']+':'+name])).hexdigest()==h,name
runs=[json.loads(l) for l in (b/'runs.jsonl').read_text().splitlines()];assert len(runs)==160
remote_root=read(r/'retrieval_manifest.json')['root']
for row in runs:
 lock=row['per_sequence_file'];local=r/Path(lock['path']).relative_to(remote_root)
 assert sha(local)==lock['sha256']
 data=read(local);assert float(np.asarray(data['per_sequence_nll'],dtype=np.float64).mean())==row['nll']
flags=Counter(x.get('test',x.get('check','unknown')) for xs in (tests['workers'],tests['coordinator']) for x in xs if not x['passed'])
assert sum(flags.values())==done['flagged_checks'] and done['calibration_disjointness_passed'] is False
assert all(x['passed'] for x in tests['coordinator']),[x for x in tests['coordinator'] if not x['passed']]
paired=read(b/'downstream_paired.json')['datasets'];verified={};model_records={};reference=None
for ds in ('dclm','fwe'):
 primary={};independent={}
 for model in [p+s for s in ('-parent','-quantized') for p in ('S72','P300','S188','P1T')]:
  name=ds+'/'+model
  values={}
  for group in ('scores','replays','independent_replays'):
   path=r/group/ds/(model+'.json');assert sha(path)==down[group][name]['sha256'];values[group]=read(path)
  score=values['scores'];rep=values['replays'];fresh=values['independent_replays']
  parent=model.split('-')[0]
  if model.endswith('-parent') and parent in ('S72','P300','P1T'):
   assert down['scores'][name]==cfg['prior_downstream'][ds]['scores'][model]
   assert down['replays'][name]==cfg['prior_downstream'][ds]['replays'][model]
  if model.endswith('-quantized'):
   identity=score['checkpoint_identity'];assert identity['dataset']==ds and identity['parent']==parent and identity['compression']=='gptq3'
   assert identity['spec_sha256']==cfg['spec_sha256'] and identity['assets']==cfg['assets']
   weights_path=r/Path(identity['quantized']['path']).relative_to(remote_root)
   assert sha(weights_path)==identity['quantized']['sha256']
   weights=read(weights_path);assert identity['full_state_sha256']==weights['full_state_sha256']
  assert score['batch_size']==2 and len(score['scores'])==score['likelihood_requests']==78259
  assert score['checkpoint_identity_sha256']==digest(score['checkpoint_identity'])
  assert score['precision']=='fp32_parameters_bf16_cuda_autocast'
  assert all(math.isfinite(ll) and ll<=0 and type(g) is bool for ll,g in score['scores'])
  identity=(score['prepared_sha256'],score['request_sha256'],score['packages'])
  if reference is None:reference=identity
  assert reference==identity
  for v in (rep,fresh):
   assert v['a2_metadata']['scored_file_sha256']==down['scores'][name]['sha256']
   assert v['a2_metadata']['prepared_sha256']==score['prepared_sha256'] and v['a2_metadata']['request_sha256']==score['request_sha256']
  for key in ('samples','results','n-shot','n-samples'):assert rep[key]==fresh[key],(name,key)
  primary[model]=rep;independent[model]=fresh
  model_records[name]={'scores':len(score['scores']),'batch_size':score['batch_size'],'primary_equals_independent':True,'score_sha256':down['scores'][name]['sha256']}
 assert downstream_statistics(primary)==paired[ds],ds
 assert downstream_statistics(independent)==paired[ds],ds
 verified[ds]=True
 print('DOWNSTREAM_VERIFIED',ds,flush=True)
samples=read(r/'verification/terminal_samples.json');assert len(samples)==2
supervisor=read(r/'supervised/supervisor.json');expected=supervisor['burn']['gpus']
progress=[]
for sample in samples:
 snap=sample['inspection'];assert snap['guard_disabled'] is False
 assert [(g['index'],g['uuid'],g['pids']) for g in snap['gpus']]==[(g['index'],g['uuid'],g['pids']) for g in expected]
 assert len(snap['workers'])==8 and len(snap['stop_roots'])==1
 launcher=snap['processes'][str(snap['stop_roots'][0])];assert launcher['launcher_kind']=='gpu_burn'
 for pid in snap['workers']:assert snap['processes'][str(pid)]['ppid']==launcher['pid']
 progress.append([(int(x),float(y)) for x,y in re.findall(r'completed_cycles=(\d+).*?completed_collective_payload_gib=([\d.]+)',sample['burn_tail'])][-1])
assert samples[0]['inspection']['processes']==samples[1]['inspection']['processes']
assert progress[1][0]>progress[0][0] and progress[1][1]>progress[0][1]
assert supervisor['training_returncode']==0
out={'spec_sha256':cfg['spec_sha256'],'source_commit':sm['code_commit'],'source_files_verified':144,'run_records_hash_verified':160,'downstream_models_verified':model_records,'paired_bootstrap_exact':verified,'native_handoff_verified':True,'burn_progress':progress,'flag_counts':dict(flags),'flagged_checks':done['flagged_checks'],'calibration_disjointness_passed':False,'execution_complete':True,'coverage_complete':True,'scientific_flags_retained':True}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
