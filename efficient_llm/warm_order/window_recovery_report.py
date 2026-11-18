"""Replay causal-window controls without selecting on final scores."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .window_order_prepare import NAMES
from .recursion_eval import check_train
from .strength_refinement import check
from .evaluate import paired_ci
from .train import write_json
from .data import sha
FLAGS=['window_restricted_attention','historical_document_boundary_convention','two_seeds','reused_final_exploratory','conditional_prefix32_suffix480','unequal_attention_compute','fullcontext_timing_not_generation','weights_node_only']

def calculate(arrays):
 from .window_order_report import calculate as original
 result=original(arrays);result['flags']=result['flags']+['canonical_fixed_shape_eval','original_grouped_causal_gate_failure_preserved','reference_evaluates_all_experts']
 return result

def replay_run(out):
 out=Path(out);c=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text());check(c['hashes'])
 if done['train'] is not None:check_train(Path(c['parent'])/'train')
 if done['status']!='complete':raise ValueError('Incomplete')
 for path,h in done['files'].items():
  if sha(out/path)!=h:raise ValueError('Result changed')
 for d in ('dclm','fwe'):
  a=np.load(out/d/'summary.npz')
  if not np.isfinite(a['nll']).all() or len(a['nll'])!=len(a['ids']):raise ValueError('Invalid NLL summary')
  if 'grouped_nll' not in a or not np.isfinite(a['grouped_nll']).all():raise ValueError('Missing parity data')
  np.testing.assert_array_equal(a['grouped_abs_delta'],np.abs(a['nll']-a['grouped_nll']))
  if float(a['grouped_abs_delta'].max())>2e-4:raise ValueError('Parity gate')
  if 'routes' in a and (a['routes'].min()<0 or a['routes'].max()>3 or len(a['routes'])!=len(a['ids'])):raise ValueError('Invalid routes')
 return done

def report(base,backup):
 base=Path(base);backup=Path(backup);arrays={};done={};files=[]
 for name in NAMES:
  out=base/'runs'/name;done[name]=replay_run(out)
  for d in ('dclm','fwe'):arrays[name,d]=dict(np.load(out/d/'summary.npz'))
  if done[name]['train'] is not None:
   parent=Path(json.loads((out/'config.json').read_text())['parent']);meta=out/'parent-train';meta.mkdir(exist_ok=False)
   for f in ('recipe.json','checkpoint.json','trace.jsonl','complete.json'):shutil.copyfile(parent/'train'/f,meta/f)
  files += [p for p in out.rglob('*') if p.is_file() and (p.suffix in ('.json','.jsonl','.npz') or p.name=='router.pt')]
 lineage=json.loads((base/'runs'/NAMES[0]/'config.json').read_text())['lineage']
 for name,path in lineage.items():
  dest=base/('parent-'+name+'.json');shutil.copyfile(path,dest);files.append(dest)
 result=calculate(arrays);result.update(runs=done,node_raw_replay='PASS',replay_scope='Full-set grouped/canonical NLL and route parity; canonical exact causal gate; per-sequence aggregation and hashes');write_json(base/'REPORT.json',result)
 with (base/'RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(result['rows'][0]),lineterminator='\n');w.writeheader();w.writerows(result['rows'])
 lines=['# Window-order canonical evaluation recovery','','| Run | Domain | NLL |','|---|---|---:|']+[f"| {r['name']} | {r['domain']} | {r['nll']:.6f} |" for r in result['rows']]+['','Continuation: '+str(result['continuation_all_seeds_domains']),'','Flags: '+', '.join(result['flags'])]
 (base/'REPORT.md').write_text('\n'.join(lines)+'\n');files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')];members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json');archive=base/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
 if archive.stat().st_size>=25_000_000:raise ValueError('Small archive cap')
 backup.mkdir(parents=True,exist_ok=False);copies={};heads=[]
 sources=[(archive,Path('results.tar.gz'))]
 for name in NAMES:
  c=json.loads((base/'runs'/name/'config.json').read_text())
  if c['train']['arm']!='D6':
   for filename in ('weights.pt','fullstate.pt'):sources.append((Path(c['parent'])/'train'/filename,Path('parents')/name/'train'/filename))
 for p,rel in sources:
  dest=backup/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  if sha(dest)!=sha(p):raise ValueError('Backup mismatch')
  copies[str(dest)]=sha(dest)
 if len(copies)!=17+len(heads):raise ValueError('Backup inventory mismatch')
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies; original D6 retains prior backup'))
 return result
