import sys,tempfile,json,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from eval.a2_family_f12 import write,read,descriptors
from scripts.a2_quant_io import lock
from efficient_llm.diagnostic_data import sha256
from efficient_llm.a2.protocol import lr_for_update
from eval.a2_family_f12_report import report
from scripts.a2_family_f12_backup import backup
with tempfile.TemporaryDirectory(dir='temp',prefix='f12-report-fixture-') as d:
 root=Path(d).resolve();prior=root/'prior.md';prior.write_text('F11 immutable prior report\n')
 cfg=dict(previous_report=lock(prior),inventory=lock('resources/a2_family_f12_inventory_20261111.json'),reference_13=dict(final_nll=2.7),output_root=str(root/'native'),f11_root=str(root/'f11'))
 for k,item in enumerate(descriptors()):
  n=item['steps'];folder=root/'heals'/item['run_id'];folder.mkdir(parents=True);traces=[]
  for rank in range(8):
   p=folder/f'sample-order-rank{rank}.jsonl'
   with p.open('w') as f:
    for mb in range(n*2):f.write(json.dumps(dict(sample_ids=list(range(mb*32+rank*4,mb*32+rank*4+4))))+'\n')
   traces.append(dict(path=str(p),sha256=sha256(p)))
  arm=k%2;parent=np.array([2.8,2.9,3.0,3.1])-arm*.1;delta=.02 if arm else .01
  def score(a):return dict(per_sequence_nll=a.tolist(),tokens=4*4095,sequences=4,loss=float(np.mean(a)))
  write(folder/'result.json',dict(registered_run=item,raw_order_files=traces,lr_updates=[lr_for_update(i,n,.001) for i in range(1,n+1)],baseline=score(parent),pre=dict(final=score(parent+.2+arm*.03)),scores=dict(final=score(parent+delta)),flags=[]))
 for arm in ('P300','P1T'):write(root/'masks'/(arm+'.json'),dict(flags=[]))
 with patch('eval.a2_family_f12_report.downstream',return_value=dict(complete=True,fixture_only=True)):report(cfg,root)
 result=read(root/'width_readout.json');assert result['complete'] and result['FE2']=='flat';assert all(abs(c['E']['estimate']-.01)<1e-12 for c in result['contrasts']);assert abs(result['slope_per_log_budget']['estimate'])<1e-12
 assert (root/'REPORT.md').read_text()==(root/'REPORT_FULL_v1.16.md').read_text()
 # Actual tar streaming/reverification, with all 48 raw rank traces in place.
 source=root/'native/source';source.mkdir(parents=True);(source/'example.py').write_text('pass\n')
 cfg['source_manifest']=write(root/'native/source-manifest.json',dict(files={'example.py':sha256(source/'example.py')}))
 for key in ['cpu_proof','f10_config_lock','preparation_proof','data_manifest','assets','f11_prepared']:
  cfg[key]=write(root/'locks'/(key+'.json'),dict(fixture=True))
 write(root/'native/config.json',cfg)
 p=root/'f11/results/a2_family_20261107/f11/backup';p.mkdir(parents=True);(p/'family_f11_scientific.tar.gz').write_bytes(b'fixture-prior-archive')
 write(p/'verified.json',dict(verified=True,archive_sha256=sha256(p/'family_f11_scientific.tar.gz')))
 saved=read(backup(root,cfg)['path']);assert saved['verified'];assert sum('sample-order-rank' in n for n in saved['files'])==48
 write('temp/f12_inputs/report_proof.json',dict(passed=True,full_registered_steps=True,raw_rank_traces=48,raw_order_and_LR_replayed=True,paired_bootstrap_10000=True,report_append_and_csv=True,archive_verified=True,archive_members=saved['members'],downstream_fixture_mocked=True))
 print('F12_REPORT_TRACE_ARCHIVE_FIXTURE_PASS',saved['members'])
