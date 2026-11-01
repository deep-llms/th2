from pathlib import Path
import json,gzip,shutil,ast
r=Path('temp/a2_depth_partc_health07');raw=Path('temp/a2_depth_partc_health07.log');base=r/'results/a2_depth_20261031'
def read(p):return json.loads(p.read_text())
e=read(r/'extraction.json');cfg=read(base/'config/plan.json');dt6=read(base/'DT6.json')
assert e['source_commit']=='dd55c8ba96eafabfed3de63a94e249da674c4f19' and e['source_files_verified']==185
parsed={}
for l in raw.read_text().splitlines():
 if l.startswith(('PROCESS_CHAINS ','SOURCE_VERIFICATION ','NONFINITE_LOG_FLAGS ')):
  k,v=l.split(' ',1);parsed[k]=json.loads(v)
assert parsed['SOURCE_VERIFICATION']['changed']==[]
assert dt6['passed'] and not dt6['flags'] and dt6['selected_sequences']==3124992 and dt6['prefix_sequences']==781248
assert dt6['tokens']['full_sha256']==cfg['stream_history']['data_manifest']['streams']['heal']['token_sha256']
assert dt6['reader_prefix_sha256']==dt6['tokens']['prefix_sha256']
assert read(r/'supervised/reclaim.json')['identities']==cfg['predecessor']['known_burn_identities']
free=read(r/'supervised/gpus-free-before-training.json')['gpus'];live=e['live'];assert len(free)==8 and all(not g['pids'] for g in free)
assert [(g['index'],g['uuid']) for g in free]==[(g['index'],g['uuid']) for g in live['gpus']]
assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
assert json.loads(e['guard'])['owner_pid']==767788 and e['session']=='0 767788'
for pid in live['workers']:
 chain=parsed['PROCESS_CHAINS'][str(pid)];assert [p['pid'] for p in chain[:4]]==[pid,767899,767828,767788]
 assert all(p['start_ticks'] and p['command_sha256'] for p in chain[:4])
checks=[]
for n in ['preparation_checks.json','runtime_checks.json','DT6_checks.json']:
 checks+=read(base/n)['checks']
worker=[]
for p in base.glob('heals/*/tests-rank*.json'):worker+=read(p)['checks']
assert len(worker)==32 and all(c['passed'] for c in checks+worker)
assert all(v==0 for v in parsed['NONFINITE_LOG_FLAGS'].values())
assert read(base/'progress.json')['active']=='P300'
summary=dict(health_at=e['health_at'],raw_sha256=e['raw_sha256'],source=e['source_commit'],source_files=185,command_commit='1f7b66cc163a08bc6e8acbc4e761f2cd5622182d',status='DT6 PASS and first C arm active',active='P300',updates_from_log_epoch=7510,total_updates=48828,worker_checks=32,other_checks=len(checks),owner=767788,coordinator=767828,workers=live['workers'],nonfinite_log_flags=parsed['NONFINITE_LOG_FLAGS'],actual_eight_gpu_resume_verified=False)
from datetime import datetime,timedelta
rows=[ast.literal_eval(l) for k,v in e['logs'].items() if 'C-P300-attempt001.log' in k for l in v.splitlines() if l.startswith("{'loss'")]
step=round(rows[-1]['epoch']*48828);assert step==7510
previous=read(Path('results/a2_depth_20261031/partc_health_1702/verification.json'))
seconds=(datetime.fromisoformat(e['health_at'])-datetime.fromisoformat(previous['health_at'])).total_seconds()
rate=seconds/(step-previous['updates_from_log_epoch'])
markers={str(p.relative_to(r)):read(p) for p in base.glob('heals/*/trainer/**/depth-checkpoint.json')}
assert len(markers)==3 and all(m['complete'] and m['state_finite'] and m['world_size']==8 for m in markers.values())
for m in markers.values():
 assert m['identity']['steps']==48828
 assert all(m['files'].get(f'rng_state_{rank}.pth',0)>0 for rank in range(8))
 assert all(m['files'].get(f'depth-audit-rank{rank}.json',0)>0 for rank in range(8))
 assert all(m['files'].get(n,0)>0 for n in ['pytorch_model.bin','optimizer.pt','scheduler.pt','trainer_state.json'])
clean=[m for n,m in markers.items() if '/clean-state/' in n];assert len(clean)==1 and clean[0]['global_step']==7000
sizes=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('CHECKPOINT_MEMBER_SIZES ')]
assert len(sizes)==3 and all(x['match'] and x['expected']==x['actual'] and len(x['actual'])==20 for x in sizes)
summary.update(checkpoint_member_sizes_match=True,checkpoint_marker_steps=[m['global_step'] for m in markers.values()],clean_checkpoint_step=7000,checkpoint_verification_scope='Saved marker and actual 20 member sizes match for each of three directories; no external tensor-load/resume performed',seconds_per_update_observed=rate,first_arm_training_eta_utc=(datetime.fromisoformat(e['health_at'])+timedelta(seconds=(48828-step)*rate)).isoformat(),all_training_eta_utc=(datetime.fromisoformat(e['health_at'])+timedelta(seconds=(3*48828-step)*rate)).isoformat(),eta_scope='Extrapolated from latest 1810 updates including saves; later arms/evaluation/downstream may differ')
out=Path('results/a2_depth_20261031/partc_health_1732');out.mkdir(exist_ok=True)
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'raw_receipt.log.gz').write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
for rel in ['source-manifest.json','results/a2_depth_20261031/DT6.json','results/a2_depth_20261031/DT6_checks.json','results/a2_depth_20261031/progress.json']:
 p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r/rel,p)
(out/'checkpoint_markers.json').write_text(json.dumps(markers,indent=2)+'\n');shutil.copyfile(__file__,out/'verifier.py');print(json.dumps(summary))
