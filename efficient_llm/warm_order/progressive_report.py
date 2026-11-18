"""Replay four teacher/order controls without selecting on final scores."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .progressive_prepare import NAMES
from .recursion_eval import check_train
from .strength_refinement import check
from .evaluate import paired_ci
from .train import write_json
from .data import sha
FLAGS=['bounded_warmup_not_convergence','two_seeds','reused_final_exploratory','conditional_prefix32_suffix480','equal_input_tokens_not_compute','scratch_common_rows_init','posthoc_frozen_prefix_router','multiple_comparisons','weights_node_only']

def calculate(arrays):
 rows=[];ci={};quality=[];curriculum=[]
 for name in NAMES:
  for d in ('dclm','fwe'):
   a=arrays[name,d]
   for metric in ('identity','uniform','learned','constant','random','oracle','actual'):rows.append(dict(name=name,domain=d,metric=metric,nll=float(a[metric].mean(dtype=np.float64)),sequences=len(a[metric])))
   for key in ('identity','constant','random'):ci[f'{name}/{d}/learned-minus-{key}']=paired_ci(a['learned']-a[key],a['clusters'])
 for s in (0,1):
  for d in ('dclm','fwe'):
   a=arrays[f'WL-s{s}',d]
   for arm in ('D','R','WS'):
    z=arrays[f'{arm}-s{s}',d];np.testing.assert_array_equal(a['ids'],z['ids']);np.testing.assert_array_equal(a['clusters'],z['clusters'])
    for metric in ('identity','learned'):ci[f'WL-s{s}/{d}/minus-{arm}-{metric}']=paired_ci(a['learned']-z[metric],a['clusters'])
   ok=ci[f'WL-s{s}/{d}/minus-WS-learned']['mean']<=-.005;curriculum.append(ok)
   quality.append(ok and ci[f'WL-s{s}/{d}/minus-D-identity']['mean']<=-.005 and ci[f'WL-s{s}/{d}/learned-minus-constant']['mean']<=-.001)
 return dict(rows=rows,comparisons=ci,curriculum_all_seeds_domains=all(curriculum),continuation_all_seeds_domains=all(quality),flags=FLAGS)

def replay_run(out):
 out=Path(out);c=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text());check(c['hashes']);check_train(out/'train')
 if done['status']!='complete' or done['proof']['causal_max']!=0 or not done['proof']['prefix_independent']:raise ValueError('Incomplete/causality failure')
 for path,h in done['files'].items():
  if sha(out/path)!=h:raise ValueError('Result changed')
 policy=json.loads((out/'policy.json').read_text());v=np.load(out/'validation.npz');np.testing.assert_array_equal(v['choices'],v['scores'].argmax(1));assert policy['constant']==int(v['costs'].mean(0).argmin());np.testing.assert_array_equal(np.bincount(v['choices'],minlength=8)/len(v['choices']),policy['route_probabilities'])
 for d in ('dclm','fwe'):
  a=np.load(out/d/'summary.npz');ix=np.arange(len(a['ids']));np.testing.assert_array_equal(a['choices'],a['scores'].argmax(1));np.testing.assert_array_equal(a['random_choices'],np.random.default_rng(941+c['train']['seed']).choice(8,len(ix),p=policy['route_probabilities']))
  for k,z in [('learned',a['costs'][ix,a['choices']]),('identity',a['costs'][:,0]),('uniform',a['costs'].mean(1)),('constant',a['costs'][:,policy['constant']]),('random',a['costs'][ix,a['random_choices']]),('oracle',a['costs'].min(1))]:np.testing.assert_array_equal(a[k],z)
  np.testing.assert_allclose(a['actual'],a['learned'],atol=2e-4,rtol=0)
 return done

def report(base,backup):
 base=Path(base);backup=Path(backup);arrays={};done={};files=[]
 for name in NAMES:
  out=base/'runs'/name;done[name]=replay_run(out)
  for d in ('dclm','fwe'):arrays[name,d]=dict(np.load(out/d/'summary.npz'))
  files += [p for p in out.rglob('*') if p.is_file() and (p.suffix in ('.json','.jsonl','.npz') or p.name=='router.pt')]
 result=calculate(arrays);result.update(runs=done,node_raw_replay='PASS');write_json(base/'REPORT.json',result)
 with (base/'RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
 lines=['# Longer progressive warm-up screen','','| Run | Domain | Metric | NLL |','|---|---|---|---:|']+[f"| {r['name']} | {r['domain']} | {r['metric']} | {r['nll']:.6f} |" for r in result['rows']]+['','Continuation: '+str(result['continuation_all_seeds_domains']),'','Flags: '+', '.join(FLAGS)]
 (base/'REPORT.md').write_text('\n'.join(lines)+'\n');files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')];members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json');archive=base/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
 if archive.stat().st_size>=25_000_000:raise ValueError('Small archive cap')
 backup.mkdir(parents=True,exist_ok=False);copies={};heads=sorted((base/'runs').glob('*/head/fullstate.pt'))
 for p in [archive,*sorted((base/'runs').glob('*/train/*.pt')),*heads]:
  dest=backup/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  if sha(dest)!=sha(p):raise ValueError('Backup mismatch')
  copies[str(dest)]=sha(dest)
 if len(copies)!=17+len(heads):raise ValueError('Backup inventory mismatch')
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies only; selected router parameters included in archive'))
 return result
