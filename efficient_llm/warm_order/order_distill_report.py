"""Replay four teacher/order controls without selecting on final scores."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .order_distill_prepare import NAMES
from .recursion_eval import check_train
from .strength_refinement import check
from .evaluate import paired_ci
from .train import write_json
from .data import sha
FLAGS=['two_seeds','reused_final_exploratory','conditional_prefix32_suffix480','teacher_and_dual_pass_training_compute','prefix_router_overhead','fullcontext_timing_not_generation','weights_node_only']

def calculate(arrays):
 rows=[];ci={};passes=[]
 for name in NAMES:
  for d in ('dclm','fwe'):
   a=arrays[name,d]
   for metric in ('identity','swap','learned','constant','random','oracle','actual'):rows.append(dict(name=name,domain=d,metric=metric,nll=float(a[metric].mean(dtype=np.float64)),sequences=len(a[metric])))
   for key in ('identity','constant','random'):ci[f'{name}/{d}/learned-minus-{key}']=paired_ci(a['learned']-a[key],a['clusters'])
 for s in (0,1):
  for d in ('dclm','fwe'):
   pk=arrays[f'PK2-s{s}',d]
   for other in ('C2','K2','P2'):
    a=arrays[f'{other}-s{s}',d];np.testing.assert_array_equal(pk['ids'],a['ids']);np.testing.assert_array_equal(pk['clusters'],a['clusters']);ci[f'PK2-s{s}/{d}/minus-{other}']=paired_ci(pk['learned']-a['learned'],a['clusters'])
   k=arrays[f'K2-s{s}',d];cc=arrays[f'C2-s{s}',d];ci[f'K2-s{s}/{d}/minus-C2']=paired_ci(k['identity']-cc['identity'],k['clusters'])
   passes.append(ci[f'PK2-s{s}/{d}/minus-C2']['mean']<=-.01 and ci[f'PK2-s{s}/{d}/minus-K2']['mean']<=-.005 and ci[f'PK2-s{s}/{d}/learned-minus-constant']['mean']<=-.001)
 return dict(rows=rows,comparisons=ci,continuation_all_seeds_domains=all(passes),flags=FLAGS)

def replay_run(out):
 out=Path(out);c=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text());check(c['hashes']);check_train(out/'train')
 if done['status']!='complete':raise ValueError('Incomplete')
 for path,h in done['files'].items():
  if sha(out/path)!=h:raise ValueError('Result changed')
 policy=json.loads((out/'policy.json').read_text());v=np.load(out/'validation.npz');np.testing.assert_array_equal(v['choices'],v['scores'].argmin(1));assert policy['constant']==int(v['costs'].mean(0).argmin());np.testing.assert_array_equal(np.bincount(v['choices'],minlength=2)/len(v['choices']),policy['route_probabilities'])
 for d in ('dclm','fwe'):
  a=np.load(out/d/'summary.npz');ix=np.arange(len(a['ids']));np.testing.assert_array_equal(a['choices'],a['scores'].argmin(1));np.testing.assert_array_equal(a['random_choices'],np.random.default_rng(941+c['train']['seed']).choice(2,len(ix),p=policy['route_probabilities']))
  for k,z in [('learned',a['costs'][ix,a['choices']]),('identity',a['costs'][:,0]),('swap',a['costs'][:,1]),('constant',a['costs'][:,policy['constant']]),('random',a['costs'][ix,a['random_choices']]),('oracle',a['costs'].min(1))]:np.testing.assert_array_equal(a[k],z)
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
 lines=['# Teacher-anchored order screen','','| Run | Domain | Metric | NLL |','|---|---|---|---:|']+[f"| {r['name']} | {r['domain']} | {r['metric']} | {r['nll']:.6f} |" for r in result['rows']]+['','Continuation: '+str(result['continuation_all_seeds_domains']),'','Flags: '+', '.join(FLAGS)]
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
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies; original teacher retains prior backup'))
 return result
