from pathlib import Path
import hashlib,json,re,shutil,gzip
raw=Path('temp/a2_depth_parta_health05.log');root=Path('temp/a2_depth_parta_health05')
def read(rel):return json.loads((root/rel).read_text())
e=read('extraction.json');parsed={}
for line in raw.read_text().splitlines():
 if line.startswith(('NATIVE_TERMINAL_SAMPLES ','PROCESS_CHAINS ','SOURCE_VERIFICATION ')):
  k,v=line.split(' ',1);parsed[k]=json.loads(v)
sv=read('supervised/supervisor.json');done=read('supervised/run/depth-parta-complete.json');native=read('supervised/run/complete.json')
assert parsed['SOURCE_VERIFICATION']['changed']==[] and e['source_files_verified']==163 and e['source_commit']=='1be9ccb33fb146e6e6712aff140e8c3346f0fbdb'
assert sv['training_status']=='ok' and sv['training_returncode']==0 and sv['finished_at']
assert len(sv['free_after_training'])==8 and not any(g['pids'] for g in sv['free_after_training'])
assert native['status']=='ok' and all(j['returncode']==0 for j in native['jobs'])
assert done['execution_complete'] and done['coverage_complete'] and done['readout_complete']
assert len(done['events'])==12 and all(x['returncode']==0 for x in done['events'])
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
 assert any('6y3o-a2-depth-parta-20261101-a01/source/resources/llm_pretrain_burn.py' in s for s in parent['command_prefix'])
 for pid in live['workers']:assert procs[str(pid)]['start_ticks'] and procs[str(pid)]['command_sha256']
 rows=re.findall(r'rank=0 .*?completed_cycles=(\d+).*?completed_collective_payload_gib=([\d.]+)',sample['burn_tail']);assert rows
 progress.append(rows[-1])
assert samples[0]['processes']==samples[1]['processes']
assert int(progress[1][0])>int(progress[0][0]) and float(progress[1][1])>float(progress[0][1])
rows=[json.loads(l.split(' ',1)[1]) for l in raw.read_text().splitlines() if l.startswith('PERSEQ ')];assert len(rows)==156
assert all(r['sequences']==2461 and r['tokens']==10077795 for r in rows)
checks=[]
for p in root.glob('results/a2_depth_20261031/part_a/*/*/tests.json'):checks.extend(json.loads(p.read_text())['checks'])
assert len(checks)==180 and all(c['passed'] for c in checks)
assert len([c for c in checks if c['test']=='DT1'])==12
out=Path('results/a2_depth_20261031/parta_terminal');out.mkdir(exist_ok=True,parents=True)
summary=dict(spec_sha256=done['spec_sha256'],status='terminal_and_metadata_verified',health_at=e['health_at'],raw_sha256=e['raw_sha256'],command_commit='70a57564eb0c668f1480effe52d6aa08f108c735',source=e['source_commit'],source_files=163,arrays=156,workerchecks=180,training_end=native['finished_at'],native_finished=sv['finished_at'],burn_progress=progress,burn_identities=samples[1]['processes'],perseq_replay_pending=True)
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'perseq_hashes.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'checks.json.gz').write_bytes(gzip.compress(json.dumps(checks,indent=2).encode(),mtime=0))
for rel in ['source-manifest.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json','supervised/run/depth-parta-complete.json','results/a2_depth_20261031/decision_depth.json','results/a2_depth_20261031/REPORT.md','results/a2_depth_20261031/ledger.json']:
 p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,p)
print(json.dumps({k:v for k,v in summary.items() if k!='burn_identities'}))
