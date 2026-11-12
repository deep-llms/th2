"""CPU-only additive report recovery; original runs and GPU work stay untouched."""
import argparse,csv,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from efficient_llm.dot_pool.data import sha,write
from scripts.gpu_process import inspect,process


def check_burn(expected):
 current=inspect()
 assert current['host']==expected['host'] and not current['guard_disabled']
 assert {g['uuid'] for g in current['gpus']}==set(expected['gpu_uuids'])
 assert set(current['workers'])==set(expected['workers'])
 for pid,old in expected['identities'].items():
  new=process(int(pid))
  for key in ['start_ticks','command_sha256','exe','ppid']:assert new[key]==old[key],(pid,key)
 return current


def recover(root,output,expected):
 assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU only'
 assert root.is_dir() and not output.exists()
 recovery_source=Path(__file__).resolve().parent.parent
 for rel,digest in expected.get('recovery_files',{}).items():assert sha(recovery_source/rel)==digest,rel
 before=check_burn(expected['burn'])
 source=Path(expected['source']);manifest=json.loads(Path(expected['source_manifest']).read_text())
 assert sha(Path(expected['source_manifest']))==expected['source_manifest_sha256']
 for rel,value in manifest['files'].items():assert sha(source/rel)==value,rel
 for path,value in expected['data_locks'].items():assert sha(Path(path))==value,path
 output.mkdir(parents=True);report=output/'report';report.mkdir();delta=output/'analysis_delta';delta.mkdir()
 artifact_rows=[];input_hashes={};counts={};run_names=[]
 for study,arms in [('E1',['D','L','SO','DOT','SR']),('E2',['D','PL','SO','DOT','SR'])]:
  for arm in arms:
   for seed in [0,1]:
    name=f'{study}-{arm}-s{seed}';run=root/name;run_names.append(name)
    export=expected['analysis_archives'][name]
    archive=Path(export['path']);assert archive.stat().st_size==export['bytes'] and sha(archive)==export['sha256']
    with tarfile.open(archive,'r:gz') as tar:archived_manifest=tar.extractfile('artifact_manifest.json').read()
    assert archived_manifest==(run/'artifact_manifest.json').read_bytes()
    records=json.loads(archived_manifest)['artifacts'];dest=report/name;dest.mkdir();dd=delta/name;dd.mkdir()
    for r in records:
     path=Path(r['path']);assert path.resolve().is_relative_to(run.resolve())
     assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],str(path)
     artifact_rows.append(r)
     if r['tier']==1:
      assert path.parent==run
      shutil.copy2(path,dest/path.name);input_hashes[str(path)]=r['sha256']
      if path.name in ['analysis_complete.json','downstream.json','reentry.json','shuffle.json','taylor_validity.json','usage.json']:
       shutil.copy2(path,dd/path.name)
    shutil.copy2(run/'artifact_manifest.json',dest/'artifact_manifest.json');shutil.copy2(run/'artifact_manifest.json',dd/'artifact_manifest.json')
    input_hashes[str(run/'artifact_manifest.json')]=sha(run/'artifact_manifest.json')
    cfg=json.loads((dest/'config.json').read_text());done=json.loads((dest/'train_complete.json').read_text());analysis=json.loads((dest/'analysis_complete.json').read_text())
    assert cfg['study']==study and cfg['arm']==arm and cfg['seed']==seed
    assert done['step']==cfg['steps']==(154000 if study=='E1' else 5722) and not done['flags']
    assert analysis['complete'] and not analysis['flags']
    if study=='E2':assert (dest/'downstream.json').is_file()
    counts[name]=len(records);print('HASH_VERIFIED',name,len(records),flush=True)
 for path in root.iterdir():
  if path.is_file() and path.suffix in ['.json','.jsonl'] and path.name not in ['requests.json','REPORT_SMALL.json','readout.json','report_complete.json']:
   shutil.copy2(path,report/path.name);input_hashes[str(path)]=sha(path)
 phase4=json.loads((report/'phase4.json').read_text());assert phase4['extra_epochs']==0 and phase4['elapsed_seconds']>=41400
 subprocess.run([sys.executable,'-m','scripts.dot_pool_report','--root',str(report)],check=True)
 readout=json.loads((report/'readout.json').read_text());checks=[]
 for study in ['E1','E2']:
  assert len(readout['tables'][study])==10
  for row in readout['tables'][study]:
   run=root/f"{study}-{row['arm']}-s{row['seed']}";final=json.loads((run/'final_eval.json').read_text());cfg=json.loads((run/'config.json').read_text());done=json.loads((run/'train_complete.json').read_text())
   if study=='E1':
    for key in ['id_test','ood_test','train','atoms','ood_atoms']:assert row[key]==final['metrics'][key]['accuracy']
   else:
    assert row['final_nll']==final['metrics']['val']['nll'] and row['tokens']==cfg['steps']*262144
    assert row['pre_decay_nll']==json.loads((run/f"eval-{int(cfg['steps']*.8)}.json").read_text())['metrics']['val']['nll']
   assert row['seconds']==done['elapsed'];checks.append(run.name)
 assert len(readout['downstream'])==10
 for row in readout['downstream']:
  actual=json.loads((root/f"E2-{row['arm']}-s{row['seed']}"/'downstream.json').read_text())['metrics']
  assert all(row[k]==v for k,v in actual.items())
 shuffle=json.loads((report/'shuffle_summary.json').read_text())['rows']
 assert len(shuffle)==4 and {(r['arm'],r['seed']) for r in shuffle}=={(a,s) for a in ['SO','DOT'] for s in [0,1]}
 assert all(r['diagnostic_seed']==0 for r in shuffle)
 # The original paths remain authoritative; do not relabel them as off-node backups.
 write(report/'artifact_index.json',dict(artifacts=artifact_rows,locations_note='Original node locations; Tier2/3 not yet off-node'))
 with (report/'REPORT.md').open('a') as f:
  f.write('\n\n## CPU report recovery\n\nOriginal report failed on duplicate seed metadata. This additive CPU recovery preserves run seed and diagnostic_seed separately; all original failure logs/status remain unchanged. No model execution was repeated. Phase4 was not triggered.\n\n## Artifact inventory\n\n| Path | Bytes | SHA256 | Locations |\n|---|---:|---|---|\n')
  for r in artifact_rows:f.write('| '+str(r['path'])+' | '+str(r['bytes'])+' | '+r['sha256']+' | '+', '.join(r['locations'])+' |\n')
 completion=json.loads((report/'report_complete.json').read_text());completion.update(report_sha256=sha(report/'REPORT.md'),recovery='CPU-only additive; original failure retained');write(report/'report_complete.json',completion)
 for path,digest in input_hashes.items():assert sha(Path(path))==digest,path
 for rel,value in manifest['files'].items():assert sha(source/rel)==value,rel
 after=check_burn(expected['burn'])
 proof=dict(original_root=str(root),output=str(output),verified_artifacts=len(artifact_rows),verified_bytes=sum(r['bytes'] for r in artifact_rows),runs=counts,report_rows_replayed=checks,downstream_rows_replayed=10,shuffle_seed_pairs_verified=4,original_inputs_unchanged=True,frozen_source_unchanged=True,recovery_files=expected.get('recovery_files',{}),native_burn_unchanged=True,native_burn_before=before,native_burn_after=after,phase4=phase4,report_sha256=sha(report/'REPORT.md'),original_status=json.loads((root/'status.json').read_text()),flags=readout['flags'],utc=time.time())
 write(report/'recovery_verification.json',proof)
 # Only compact reports, diagnostics and manifests; no raw scores, route arrays or weights.
 bundle=output/'REPORT_RECOVERED_SMALL.tar.gz'
 with tarfile.open(bundle,'w:gz') as tar:
  for path in sorted(report.iterdir()):
   if path.is_file() and path.suffix in ['.json','.csv','.md','.png'] and path.name not in ['requests.json','REPORT_SMALL.json']:tar.add(path,arcname=path.name,recursive=False)
  for path in sorted(delta.rglob('*')):
   if path.is_file():tar.add(path,arcname='analysis_delta/'+str(path.relative_to(delta)),recursive=False)
 assert bundle.stat().st_size<20*1024*1024
 export=dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),verified=True,report_sha256=sha(report/'REPORT.md'),original_inputs_unchanged=True)
 write(output/'complete.json',export);print('DOT_RECOVERY_COMPLETE',json.dumps(export),flush=True)
 return export


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--expected',required=True);a=p.parse_args()
 recover(Path(a.root),Path(a.output),json.loads(Path(a.expected).read_text()))
if __name__=='__main__':main()
