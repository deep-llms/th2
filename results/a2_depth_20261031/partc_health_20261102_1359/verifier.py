from pathlib import Path
import json,gzip,shutil,ast
r=Path('temp/a2_depth_partc_health27');raw=Path('temp/a2_depth_partc_health27.log');base=r/'results/a2_depth_20261031'
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
 chain=parsed['PROCESS_CHAINS'][str(pid)];assert [p['pid'] for p in chain[:4]]==[pid,773611,767828,767788]
 assert all(p['start_ticks'] and p['command_sha256'] for p in chain[:4])
checks=[]
for n in ['preparation_checks.json','runtime_checks.json','DT6_checks.json']:
 checks+=read(base/n)['checks']
worker=[]
for p in base.glob('heals/*/tests-rank*.json'):worker+=read(p)['checks']
assert len(worker)==120 and all(c['passed'] for c in checks+worker)
assert all(v==0 for v in parsed['NONFINITE_LOG_FLAGS'].values())
assert read(base/'progress.json')['active']=='P1T'

from datetime import datetime,timedelta
progress=read(base/'progress.json')
assert len(progress['completed'])==1 and progress['completed'][0]['arm']=='P300' and progress['completed'][0]['returncode']==0
results=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('HEAL_RESULT ')]
assert len(results)==1
p300=results[0]
assert p300['arm']=='P300' and p300['status']=='complete' and p300['lr_updates']==48828 and p300['roundtrip_difference']==0 and not p300['unverified'] and not p300['flags']
decision=read(base/'decision_depth_partc.json')
assert decision['parents']['P300']['val']==p300['scores']['final']['loss'] and decision['parents']['P300']['delta_post']==p300['delta_post']
assert not decision['coverage_complete'] and len(decision['flags'])==2
assert {x['run_id'].split('-')[2] for x in decision['flags']}=={'P1T','X72'}
rows=[ast.literal_eval(l) for k,v in e['logs'].items() if 'C-P1T-attempt001.log' in k for l in v.splitlines() if l.startswith("{'loss'")]
step=round(rows[-1]['epoch']*48828);assert step==33300
markers={str(p.relative_to(r)):read(p) for p in base.glob('heals/*/trainer/**/depth-checkpoint.json')}
assert len(markers)==6 and all(m['complete'] and m['state_finite'] and m['world_size']==8 for m in markers.values())
for m in markers.values():
 assert m['identity']['steps']==48828
 assert all(m['files'].get(f'rng_state_{rank}.pth',0)>0 for rank in range(8))
 assert all(m['files'].get(f'depth-audit-rank{rank}.json',0)>0 for rank in range(8))
 assert all(m['files'].get(n,0)>0 for n in ['pytorch_model.bin','optimizer.pt','scheduler.pt','trainer_state.json'])
clean={('P1T' if 'ST1-dclm-P1T-' in n else 'P300'):m['global_step'] for n,m in markers.items() if '/clean-state/' in n}
assert clean=={'P1T':33000,'P300':48828}
sizes=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('CHECKPOINT_MEMBER_SIZES ')]
assert len(sizes)==6 and all(x['match'] and x['expected']==x['actual'] and len(x['actual'])==20 for x in sizes)
rate=p300['training_seconds']/48828
at=datetime.fromisoformat(e['health_at'])
summary=dict(health_at=e['health_at'],raw_sha256=e['raw_sha256'],source=e['source_commit'],source_files=185,command_commit='2cc85f58be4be352dd8b29efb95bc34e2b562f7e',status='P300 complete; P1T active',active='P1T',updates_from_log_epoch=step,total_updates=48828,worker_checks=len(worker),other_checks=len(checks),owner=767788,coordinator=767828,launcher=773611,workers=live['workers'],nonfinite_log_flags=parsed['NONFINITE_LOG_FLAGS'],completed_results=results,actual_eight_gpu_resume_verified=False,checkpoint_member_sizes_match=True,clean_checkpoint_steps=clean,checkpoint_verification_scope='Saved marker and actual20member sizes for6directories; no external tensor-load/resume',seconds_per_update_from_completed_P300=rate,active_training_eta_utc=(at+timedelta(seconds=(48828-step)*rate)).isoformat(),all_training_eta_utc=(at+timedelta(seconds=(2*48828-step)*rate)).isoformat(),eta_scope='Extrapolated using completed P300 training runtime; P1T/X72 speed and evaluation/downstream may differ',partial_readout_flags=decision['flags'],independent_saved_array_replay='pending')
out=Path('results/a2_depth_20261031/partc_health_20261102_1359');out.mkdir(exist_ok=True)
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'raw_receipt.log.gz').write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
for rel in ['source-manifest.json','results/a2_depth_20261031/DT6.json','results/a2_depth_20261031/DT6_checks.json','results/a2_depth_20261031/progress.json','results/a2_depth_20261031/attempts.json','results/a2_depth_20261031/coordinator_checks.json','results/a2_depth_20261031/decision_depth_partc.json']:
 p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r/rel,p)
(out/'checkpoint_markers.json').write_text(json.dumps(markers,indent=2)+'\n');shutil.copyfile(__file__,out/'verifier.py')
note=dict(spec_sha256=cfg['spec_sha256'],posted_at=datetime.now().astimezone().isoformat(),arm='P300',result=p300,evidence=str(out/'verification.json'),independent_saved_array_replay='pending',E=None,A_init=None,limitations=['P1T and X72 results incomplete; no Part C comparative conclusion','Actual eight GPU resume unverified','Breadth-before-Depth ordering deviation retained'])
(out/'P300_note.json').write_text(json.dumps(note,indent=2)+'\n')
print(json.dumps(summary))
