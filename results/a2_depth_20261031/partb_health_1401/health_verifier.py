from pathlib import Path
import json,hashlib,shlex,ast,gzip
root=Path('temp/a2_depth_partb_health06');raw=Path('temp/a2_depth_partb_health06.log')
def read(p):return json.loads((root/p).read_text())
e=read('extraction.json');parsed={};heals=[]
for line in raw.read_text().splitlines():
 if line.startswith(('PROCESS_CHAINS ','SOURCE_VERIFICATION ','LAUNCH_PROCESSES ')):
  k,v=line.split(' ',1);parsed[k]=json.loads(v)
 if line.startswith('HEAL_RESULT '):heals.append(json.loads(line.split(' ',1)[1]))
assert e['source_commit']=='a9cbd433d1fb91642c266319463e91c7a931afc4' and e['source_files_verified']==176
assert parsed['SOURCE_VERIFICATION']['changed']==[]
assert read('supervised/supervisor.json')['training_status']=='running'
assert read('supervised/run/run.json')['status']=='running'
assert json.loads(e['guard'])['owner_pid']==762457 and e['session']=='0 762457'
live=e['live'];assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
commands={int(r.split()[0]):r.split(None,5)[5] for r in parsed['LAUNCH_PROCESSES']}
for worker in live['workers']:
 chain=parsed['PROCESS_CHAINS'][str(worker)]
 assert [p['pid'] for p in chain[:4]]==[worker,765359,762497,762457]
 for row in chain[:4]:
  assert row['start_ticks']>0
  args=shlex.split(commands[row['pid']]);assert hashlib.sha256(('\0'.join(args)+'\0').encode()).hexdigest()==row['command_sha256']
 assert '--dataset dclm --arm X72 --block W1' in commands[worker]
free=read('supervised/gpus-free-before-training.json');assert [(g['index'],g['uuid']) for g in free['gpus']]==[(g['index'],g['uuid']) for g in live['gpus']]
checks=read('results/a2_depth_20261031/preparation_checks.json')['checks'];assert len(checks)==43 and all(c['passed'] for c in checks)
workers=[]
for path in root.glob('results/a2_depth_20261031/heals/*/tests-rank*.json'):workers.extend(json.loads(path.read_text())['checks'])
assert workers and all(c['passed'] for c in workers)
progress=read('results/a2_depth_20261031/progress.json');assert progress['active']=='D-dclm-X72-W1-0.2B-lr1e-3'
log=e['logs']['results/a2_depth_20261031/logs/'+progress['active']+'.log'];rows=[ast.literal_eval(l) for l in log.splitlines() if l.startswith("{'loss':")];step=round(rows[-1]['epoch']*762);assert step==270
assert len(heals)==9 and all(h['status']=='complete' and h['lr_updates']==762 and h['roundtrip_difference']==0 and not h['flags'] for h in heals)
prior=json.loads(Path('results/a2_depth_20261031/partb_health_1202/verification.json').read_text())['DT4']['result_summary'];assert next(h for h in heals if h['block']=='S1')==prior
byarm={h['arm']:h for h in heals if h['block']=='W9' and h['dataset']=='dclm'};E=byarm['P1T']['delta_post']-byarm['P300']['delta_post']
nonfinite=json.loads(next(l.split(' ',1)[1] for l in raw.read_text().splitlines() if l.startswith('NONFINITE_LOG_FLAGS ')))
assert len(nonfinite)==10 and all(v==0 for v in nonfinite.values())
previous=json.loads(Path('results/a2_depth_20261031/partb_health_1332/verification.json').read_text())
for old in previous['completed']:assert next(h for h in heals if h['run_id']==old['run_id'])==old
A=byarm['X72']['scores']['final']['loss']-byarm['P300']['scores']['final']['loss']
out=Path('results/a2_depth_20261031/partb_health_1401');out.mkdir(exist_ok=True)
summary=dict(spec_sha256=progress['spec_sha256'],health_at=e['health_at'],raw_sha256=e['raw_sha256'],health_commit='aab3d00f672494e122408e295462b826deac9c30',source=e['source_commit'],source_files=176,phase=progress['active'],updates=step,total_updates=762,completed_B=8,preparation_checks=len(checks),worker_checks=len(workers),full_ownership_verified=True,workers=live['workers'],guard=762457,gpu_utilization=[g['utilization_percent'] for g in live['gpus']],completed=heals,DCLM_W9_E_from_summary=E,DCLM_W9_A_init_from_summary=A,nonfinite_log_counts=nonfinite,full_array_replay='pending',raw_receipt='raw_receipt.log.gz')
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'raw_receipt.log.gz').write_bytes(gzip.compress(raw.read_bytes(),mtime=0));(out/'health_verifier.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in summary.items() if k!='completed'}))
