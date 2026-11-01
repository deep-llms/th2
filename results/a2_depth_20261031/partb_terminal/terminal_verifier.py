from pathlib import Path
import hashlib,json,re,shutil,gzip
raw=Path('temp/a2_depth_partb_health08.log');root=Path('temp/a2_depth_partb_health08')
def read(rel):return json.loads((root/rel).read_text())
e=read('extraction.json');parsed={}
for line in raw.read_text().splitlines():
 if line.startswith(('NATIVE_TERMINAL_SAMPLES ','PROCESS_CHAINS ','SOURCE_VERIFICATION ')):
  k,v=line.split(' ',1);parsed[k]=json.loads(v)
sv=read('supervised/supervisor.json');done=read('supervised/run/depth-partb-complete.json');native=read('supervised/run/complete.json')
assert parsed['SOURCE_VERIFICATION']['changed']==[] and e['source_files_verified']==176 and e['source_commit']=='a9cbd433d1fb91642c266319463e91c7a931afc4'
assert sv['training_status']=='ok' and sv['training_returncode']==0 and sv['finished_at']
assert len(sv['free_after_training'])==8 and not any(g['pids'] for g in sv['free_after_training'])
assert native['status']=='ok' and all(j['returncode']==0 for j in native['jobs'])
assert done['execution_complete'] and done['coverage_complete'] and done['readout_complete']
assert len(done['events'])==13 and all(x['returncode']==0 for x in done['events'])
assert e['guard']=='released' and (not e['session'] or e['session'].split()[0]=='1')
burn=read('supervised/burn-verified.json');assert burn['collective_progress_verified']
samples=parsed['NATIVE_TERMINAL_SAMPLES'];assert len(samples)==2
original=read('supervised/gpus-free-before-training.json')
progress=[]
for sample in samples:
 live=sample['inspection'];procs=sample['processes']
 assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
 assert [(g['index'],g['uuid']) for g in live['gpus']]==[(g['index'],g['uuid']) for g in original['gpus']]
 assert {(g['index'],g['uuid'],tuple(g['pids'])) for g in live['gpus']}=={(g['index'],g['uuid'],tuple(g['pids'])) for g in burn['gpus']}
 parentids={procs[str(p)]['ppid'] for p in live['workers']};assert len(parentids)==1
 parent=procs[str(next(iter(parentids)))]
 assert any('6y3o-a2-depth-partb-20261101-a01/source/resources/llm_pretrain_burn.py' in s for s in parent['command_prefix'])
 for pid in live['workers']:assert procs[str(pid)]['start_ticks'] and procs[str(pid)]['command_sha256']
 rows=re.findall(r'rank=0 .*?completed_cycles=(\d+).*?completed_collective_payload_gib=([\d.]+)',sample['burn_tail']);assert rows
 progress.append(rows[-1])
assert samples[0]['processes']==samples[1]['processes']
assert int(progress[1][0])>int(progress[0][0]) and float(progress[1][1])>float(progress[0][1])
rows=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('HEAL_RESULT ')];assert len(rows)==13
assert all(r['status']=='complete' and r['lr_updates']==762 and r['roundtrip_difference']==0 and not r['flags'] for r in rows)
checks=[]
for p in root.glob('results/a2_depth_20261031/heals/*/tests-rank*.json'):checks.extend(json.loads(p.read_text())['checks'])
assert len(list(root.glob('results/a2_depth_20261031/heals/*/tests-rank*.json')))==104
assert checks and all(c['passed'] for c in checks)
nonfinite=json.loads(next(l.split(' ',1)[1] for l in raw.read_text().splitlines() if l.startswith('NONFINITE_LOG_FLAGS ')))
assert len(nonfinite)==13 and all(v==0 for v in nonfinite.values())
prior=json.loads(Path('results/a2_depth_20261031/partb_health_1432/verification.json').read_text())
for old in prior['completed']:assert next(r for r in rows if r['run_id']==old['run_id'])==old
out=Path('results/a2_depth_20261031/partb_terminal');out.mkdir(exist_ok=True,parents=True)
summary=dict(spec_sha256=done['spec_sha256'],status='terminal_and_metadata_verified',health_at=e['health_at'],raw_sha256=e['raw_sha256'],command_commit='a63084c41d137ce323745d98767bd59ed508da55',source=e['source_commit'],source_files=176,completed_B=12,DT4_complete=True,workerchecks=len(checks),nonfinite_log_counts=nonfinite,training_end=native['finished_at'],native_finished=sv['finished_at'],burn_progress=progress,burn_identities=samples[1]['processes'],perseq_replay_pending=True)
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'heal_summaries.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'checks.json.gz').write_bytes(gzip.compress(json.dumps(checks,indent=2).encode(),mtime=0))
for rel in ['source-manifest.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json','supervised/run/depth-partb-complete.json','results/a2_depth_20261031/decision_depth_partb.json','results/a2_depth_20261031/REPORT.md','results/a2_depth_20261031/ledger.json']:
 p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,p)
print(json.dumps({k:v for k,v in summary.items() if k!='burn_identities'}))
