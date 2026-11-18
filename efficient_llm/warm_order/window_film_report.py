"""Replay causal-window controls without selecting on final scores."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .window_film_prepare import NAMES
from .recursion_eval import check_train
from .strength_refinement import check
from .evaluate import paired_ci
from .train import write_json
from .data import sha
FLAGS=['canonical_fixed_shape_eval','reference_evaluates_all_experts','grouped_training_numeric_caveat','continued_from_window_parents','window_restricted_attention','historical_document_boundary_convention','two_seeds','reused_final_exploratory','conditional_prefix32_suffix480','unequal_attention_compute','fullcontext_timing_not_generation','weights_node_only']

def calculate(arrays):
 rows=[];ci={};passes=[]
 for name in NAMES:
  for d in ('dclm','fwe'):
   a=arrays[name,d];row=dict(name=name,domain=d,nll=float(a['nll'].mean(dtype=np.float64)),sequences=len(a['nll']))
   if 'routes' in a:
    r=a['routes'];hist=np.bincount(r.flatten(),minlength=4);prob=hist/hist.sum();row.update(route_histogram=hist.tolist(),route_entropy=float(-(prob[prob>0]*np.log(prob[prob>0])).sum()),within_depth_window_switch_fraction=float((r[:,1:]!=r[:,:-1])[:,a['route_depth'][1:]==a['route_depth'][:-1]].mean()))
   row.setdefault('route_histogram',[]);row.setdefault('route_entropy',None);row.setdefault('within_depth_window_switch_fraction',None);rows.append(row)
 for s in (0,1):
  for d in ('dclm','fwe'):
   a=arrays[f'LF-s{s}',d]
   for other in ('G0','GF','L0','D6C'):
    z=arrays[f'{other}-s{s}',d];np.testing.assert_array_equal(a['ids'],z['ids']);np.testing.assert_array_equal(a['clusters'],z['clusters']);ci[f'LF-s{s}/{d}/minus-{other}']=paired_ci(a['nll']-z['nll'],a['clusters'])
   passes.append(all(ci[f'LF-s{s}/{d}/minus-{k}']['mean']<=-v for k,v in [('GF',.005),('L0',.005),('D6C',.01)]))
 return dict(rows=rows,comparisons=ci,continuation_all_seeds_domains=all(passes),flags=FLAGS)

def replay_run(out):
 out=Path(out);c=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text());check(c['hashes'])
 if done['train'] is not None:check_train(out/'train')
 if done['status']!='complete':raise ValueError('Incomplete')
 if done['proof']['canonical_causal_max']!=0 or not done['proof']['route_parity']:raise ValueError('Causal reference proof failure')
 for path,h in done['files'].items():
  if sha(out/path)!=h:raise ValueError('Result changed')
 for d in ('dclm','fwe'):
  a=np.load(out/d/'summary.npz')
  if not np.isfinite(a['nll']).all() or len(a['nll'])!=len(a['ids']):raise ValueError('Invalid NLL summary')
  np.testing.assert_array_equal(a['grouped_abs_delta'],np.abs(a['nll']-a['grouped_nll']))
  if float(a['grouped_abs_delta'].max())>2e-4:raise ValueError('Reference/grouped parity failure')
  if 'routes' in a and (a['routes'].min()<0 or a['routes'].max()>3 or len(a['routes'])!=len(a['ids'])):raise ValueError('Invalid routes')
 return done

def report(base,backup):
 base=Path(base);backup=Path(backup);arrays={};done={};files=[]
 for name in NAMES:
  out=base/'runs'/name;done[name]=replay_run(out)
  for d in ('dclm','fwe'):arrays[name,d]=dict(np.load(out/d/'summary.npz'))
  files += [p for p in out.rglob('*') if p.is_file() and (p.suffix in ('.json','.jsonl','.npz') or p.name=='router.pt')]
 result=calculate(arrays);result.update(runs=done,node_raw_replay='PASS',replay_scope='Canonical exact causality; full-set grouped/canonical route-NLL parity; per-sequence NLL aggregation and hashes; independent single-sequence forward checks; no downloaded token logits');write_json(base/'REPORT.json',result)
 with (base/'RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(result['rows'][0]),lineterminator='\n');w.writeheader();w.writerows(result['rows'])
 lines=['# Slot-conditioned window-order screen','','| Run | Domain | NLL |','|---|---|---:|']+[f"| {r['name']} | {r['domain']} | {r['nll']:.6f} |" for r in result['rows']]+['','Continuation: '+str(result['continuation_all_seeds_domains']),'','Flags: '+', '.join(FLAGS)]
 (base/'REPORT.md').write_text('\n'.join(lines)+'\n');files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')];members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json');archive=base/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
 if archive.stat().st_size>=25_000_000:raise ValueError('Small archive cap')
 backup.mkdir(parents=True,exist_ok=False);copies={};heads=[]
 for p in [archive,*sorted((base/'runs').glob('*/train/*.pt')),*heads]:
  dest=backup/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  if sha(dest)!=sha(p):raise ValueError('Backup mismatch')
  copies[str(dest)]=sha(dest)
 if len(copies)!=21+len(heads):raise ValueError('Backup inventory mismatch')
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies; original D6 retains prior backup'))
 return result
