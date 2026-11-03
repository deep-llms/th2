from pathlib import Path
import json,ast,gzip,shutil
r=Path('temp/a2_depth_partc_health29');raw=Path('temp/a2_depth_partc_health29.log');base=r/'results/a2_depth_20261031'
def read(p): return json.loads(p.read_text())
e=read(r/'extraction.json');cfg=read(base/'config/plan.json');dt=read(base/'DT6.json')
assert e['source_commit']=='dd55c8ba96eafabfed3de63a94e249da674c4f19' and e['source_files_verified']==185
parsed={k:json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() for k in ['PROCESS_CHAINS','SOURCE_VERIFICATION','NONFINITE_LOG_FLAGS'] if l.startswith(k+' ')}
assert not parsed['SOURCE_VERIFICATION']['changed']
assert dt['passed'] and not dt['flags'] and dt['selected_sequences']==3124992 and dt['prefix_sequences']==781248
assert dt['tokens']['full_sha256']==cfg['stream_history']['data_manifest']['streams']['heal']['token_sha256'] and dt['reader_prefix_sha256']==dt['tokens']['prefix_sha256']
free=read(r/'supervised/gpus-free-before-training.json')['gpus'];live=e['live']
assert [(g['index'],g['uuid']) for g in free]==[(g['index'],g['uuid']) for g in live['gpus']]
assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
assert json.loads(e['guard'])['owner_pid']==767788 and e['session']=='0 767788'
for pid in live['workers']:
 chain=parsed['PROCESS_CHAINS'][str(pid)];assert [x['pid'] for x in chain[:4]]==[pid,778223,767828,767788]
 assert all(x['start_ticks'] and x['command_sha256'] for x in chain[:4])
checks=[c for n in ['preparation_checks.json','runtime_checks.json','DT6_checks.json'] for c in read(base/n)['checks']]
worker=[c for p in base.glob('heals/*/tests-rank*.json') for c in read(p)['checks']]
assert len(worker)==208 and all(c['passed'] for c in checks+worker)
assert all(v==0 for v in parsed['NONFINITE_LOG_FLAGS'].values())
progress=read(base/'progress.json');assert progress['active']=='X72' and {c['arm'] for c in progress['completed']}=={'P300','P1T'} and all(c['returncode']==0 and c['attempt']==1 for c in progress['completed'])
results=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('HEAL_RESULT ')]
assert len(results)==2
for v in results:
 assert v['status']=='complete' and v['lr_updates']==48828 and v['roundtrip_difference']==0 and not v['unverified'] and not v['flags']
 decision=read(base/'decision_depth_partc.json');note=read(base/'notes'/f"{v['arm']}.json")
 assert note['result']==decision['parents'][v['arm']] and note['spec_sha256']==cfg['spec_sha256']
 assert abs(note['result']['val']-v['scores']['final']['loss'])<=1e-14 and note['result']['delta_post']==v['delta_post']
markers={str(p.relative_to(r)):read(p) for p in base.glob('heals/*/trainer/**/depth-checkpoint.json')}
assert len(markers)==9 and all(m['complete'] and m['state_finite'] and m['world_size']==8 and m['identity']['steps']==48828 for m in markers.values())
sizes=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('CHECKPOINT_MEMBER_SIZES ')]
assert len(sizes)==9 and all(x['match'] and x['expected']==x['actual'] and len(x['actual'])==20 for x in sizes)
rows=[ast.literal_eval(l) for k,v in e['logs'].items() if 'C-X72-attempt001.log' in k for l in v.splitlines() if l.startswith("{'loss'")]
step=round(rows[-1]['epoch']*48828);assert step==16480
summary=dict(health_at=e['health_at'],raw_sha256=e['raw_sha256'],command_commit='a0666c57957bc1016a09a05525ce149c195bff84',source=e['source_commit'],source_files=185,active='X72',updates_from_log_epoch=step,total_updates=48828,completed_results=results,worker_checks=len(worker),other_checks=len(checks),owner=767788,coordinator=767828,launcher=778223,workers=live['workers'],nonfinite_log_flags=parsed['NONFINITE_LOG_FLAGS'],checkpoint_member_sizes_match=True,actual_eight_gpu_resume_verified=False,independent_saved_array_replay='pending',observation_scope='Historical snapshot02Nov22:46SGT; retrieved03Nov10:36SGT; no current progress/ETA inference')
out=Path('results/a2_depth_20261031/partc_health_20261102_2246');out.mkdir(exist_ok=True)
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'raw_receipt.log.gz').write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
for n in ['progress.json','decision_depth_partc.json','notes/P1T.json','notes/P300.json','DT6.json']:
 p=out/n;p.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(base/n,p)
shutil.copyfile(__file__,out/'verifier.py')
print(json.dumps(summary))
