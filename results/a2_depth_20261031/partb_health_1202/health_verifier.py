from pathlib import Path
import json,hashlib,gzip,shlex,ast
raw=Path('temp/a2_depth_partb_health02.log');root=Path('temp/a2_depth_partb_health02')
def read(p):return json.loads((root/p).read_text())
e=read('extraction.json');parsed={}
for line in raw.read_text().splitlines():
 if line.startswith(('PROCESS_CHAINS ','SOURCE_VERIFICATION ','LAUNCH_PROCESSES ')):
  k,v=line.split(' ',1);parsed[k]=json.loads(v)
assert e['source_commit']=='a9cbd433d1fb91642c266319463e91c7a931afc4' and e['source_files_verified']==176
assert parsed['SOURCE_VERIFICATION']['changed']==[]
assert read('supervised/supervisor.json')['training_status']=='running'
assert read('supervised/run/run.json')['status']=='running'
guard=json.loads(e['guard']);assert guard['owner_pid']==762457 and e['session']=='0 762457'
live=e['live'];assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
commands={int(r.split()[0]):r.split(None,5)[5] for r in parsed['LAUNCH_PROCESSES']}
for worker in live['workers']:
 chain=parsed['PROCESS_CHAINS'][str(worker)]
 assert [p['pid'] for p in chain[:4]]==[worker,762849,762497,762457]
 for row in chain[:4]:
  assert row['start_ticks']>0
  args=shlex.split(commands[row['pid']]);assert hashlib.sha256(('\0'.join(args)+'\0').encode()).hexdigest()==row['command_sha256']
 assert '--dataset dclm --arm P300 --block W9' in commands[worker]
free=read('supervised/gpus-free-before-training.json');assert len(free['gpus'])==8 and not any(g['pids'] for g in free['gpus'])
assert [(g['index'],g['uuid']) for g in free['gpus']]==[(g['index'],g['uuid']) for g in live['gpus']]
cfg=read('results/a2_depth_20261031/config/plan.json');reclaim=read('supervised/reclaim.json')
assert reclaim['signal_targets']==list(range(761449,761457))
assert reclaim['identities']==cfg['predecessor']['known_burn_identities']
checks=read('results/a2_depth_20261031/preparation_checks.json')['checks'];assert len(checks)==43 and all(c['passed'] for c in checks)
workers=[]
for path in root.glob('results/a2_depth_20261031/heals/*/tests-rank*.json'):workers.extend(json.loads(path.read_text())['checks'])
assert workers and all(c['passed'] for c in workers)
progress=read('results/a2_depth_20261031/progress.json');assert progress['active']=='D-dclm-P300-W9-0.2B-lr1e-3'
log=e['logs']['results/a2_depth_20261031/logs/D-dclm-P300-W9-0.2B-lr1e-3.log']
rows=[ast.literal_eval(l) for l in log.splitlines() if l.startswith("{'loss':")]
step=round(rows[-1]['epoch']*762);assert step==1
out=Path('results/a2_depth_20261031/partb_health_1202');out.mkdir(parents=True,exist_ok=True)
summary=dict(spec_sha256=cfg['spec_sha256'],health_at=e['health_at'],raw_sha256=e['raw_sha256'],health_commit='b2099f68ba4e5bd7f0f53c47cbe9a24406701598',source=e['source_commit'],source_files=176,started_at=read('supervised/run/run.json')['started_at'],phase=progress['active'],updates=step,total_updates=762,preparation_checks=len(checks),initial_worker_checks=len(workers),full_ownership_verified=True,guard=guard,live=live,chains=parsed['PROCESS_CHAINS'],exact_known_burn_reclaimed=reclaim['signal_targets'])
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
for rel in ['source-manifest.json','admission.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','results/a2_depth_20261031/preparation_checks.json']:
 p=out/(rel.replace('/','__')+'.gz');p.write_bytes(gzip.compress((root/rel).read_bytes(),mtime=0))
(out/'active.log').write_text(log)
heal=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('HEAL_RESULT ')]
assert len(heal)==1 and heal[0]['status']=='complete' and heal[0]['lr_updates']==762 and not heal[0]['flags']
diff=heal[0]['scores']['final']['loss']-2.942554079
assert abs(diff)<=.002
summary['DT4']={'result_summary':heal[0],'reference':2.942554079,'difference':diff,'passed':True,'full_saved_array_replay':'pending'}
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'DT4.log').write_text(e['logs']['results/a2_depth_20261031/logs/DT4.log'])
(out/'health_verifier.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in summary.items() if k not in ('live','chains')}))
