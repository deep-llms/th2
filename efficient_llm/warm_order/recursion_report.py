"""Replay depth decisions and compare parameter/compute controls without final selection."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .recursion_prepare import NAMES
from .recursion_eval import check_train
from .recursion_head import choose,calibrate
from .strength_refinement import check
from .evaluate import paired_ci
from .train import write_json
from .data import sha
FLAGS=['adaptive_depth_not_dynamic_order','two_seeds','reused_final_exploratory','conditional_prefix32_suffix480','block_tokens_not_measured_FLOPs','prefix_router_overhead_counted','cross_job_fullcontext_timing_not_generation','weights_node_only']


def calculate(arrays):
 rows=[];ci={};work={};passes=[]
 for name in NAMES:
  for domain in ('dclm','fwe'):
   a=arrays[name,domain];metrics={k:a[k] for k in ('dense','learned','random','constant','depth2','oracle','actual') if k in a}
   if name.startswith('RV'):
    metrics.update({f'depth{i+1}':a['costs'][:,i] for i in (0,2)})
    work[f'{name}/{domain}']=dict(mean_blocks=float(a['block_work'].mean()),route_counts=np.bincount(a['choices'],minlength=3).tolist())
   for metric,v in metrics.items():rows.append(dict(name=name,domain=domain,metric=metric,nll=float(v.mean(dtype=np.float64)),sequences=len(v)))
 for seed in (0,1):
  name=f'RV-s{seed}'
  for d in ('dclm','fwe'):
   a=arrays[name,d]
   compare={'depth2':a['depth2'],'constant':a['constant'],'random':a['random']}
   for arm in ('D6','D4','RF'):
    b=arrays[f'{arm}-s{seed}',d];np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['clusters'],b['clusters']);compare[arm]=b['dense']
   for key,v in compare.items():ci[f'{name}/{d}/minus-{key}']=paired_ci(a['learned']-v,a['clusters'])
   passes.append(ci[f'{name}/{d}/minus-D6']['mean']<=-.01 and ci[f'{name}/{d}/minus-depth2']['mean']<=-.005 and work[f'{name}/{d}']['mean_blocks']<=6)
 return dict(rows=rows,comparisons=ci,work=work,continuation_all_seeds_domains=all(passes),flags=FLAGS)


def replay_run(out):
 out=Path(out);c=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text());check(c['hashes'])
 if done['status']!='complete':raise ValueError('Incomplete')
 for p,h in done['files'].items():
  if sha(out/p)!=h:raise ValueError('Result changed')
 if c['train']['arm']!='D6':check_train(out/'train')
 if done['arm']=='RV':
  policy=json.loads((out/'policy.json').read_text());v=np.load(out/'validation.npz');limit=(6-2-4*c['train']['model']['prefix']/c['train']['model']['length'])/2
  if calibrate(v['scores'],v['costs'],limit)!=policy['policy']:raise ValueError('Policy calibration changed')
  np.testing.assert_array_equal(v['choices'],choose(v['scores'],policy['policy']))
  for d in ('dclm','fwe'):
   a=np.load(out/d/'summary.npz');n=np.arange(len(a['ids']));np.testing.assert_array_equal(a['choices'],choose(a['scores'],policy['policy']))
   for key,value in [('learned',a['costs'][n,a['choices']]),('random',a['costs'][n,a['random_choices']]),('constant',a['costs'][:,policy['constant']]),('depth2',a['costs'][:,1]),('oracle',a['costs'].min(1))]:np.testing.assert_array_equal(a[key],value)
   np.testing.assert_allclose(a['actual'],a['learned'],atol=2e-4,rtol=0)
   np.testing.assert_array_equal(a['block_work'],2+2*(a['choices']+1)+4*c['train']['model']['prefix']/c['train']['model']['length'])
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
 lines=['# Shared-block adaptive depth','','Six new fits, two reused D6 evaluations; prefix-only router calibrated on validation.','','| Run | Domain | Metric | NLL |','|---|---|---|---:|']
 lines += [f"| {r['name']} | {r['domain']} | {r['metric']} | {r['nll']:.6f} |" for r in result['rows']]
 lines += ['','Continuation criterion: '+str(result['continuation_all_seeds_domains']),'','Actual route work: '+json.dumps(result['work']),'','Flags: '+', '.join(FLAGS)]
 (base/'REPORT.md').write_text('\n'.join(lines)+'\n');files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')]
 members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json');archive=base/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
 if archive.stat().st_size>=25_000_000:raise ValueError('Small archive cap')
 backup.mkdir(parents=True,exist_ok=False);copies={}
 for p in [archive,*sorted((base/'runs').glob('*/train/*.pt')),*sorted((base/'runs').glob('*/head/fullstate.pt'))]:
  dest=backup/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  if sha(dest)!=sha(p):raise ValueError('Backup mismatch')
  copies[str(dest)]=sha(dest)
 if len(copies)!=15:raise ValueError('Backup inventory mismatch')
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies; existing D6 keeps prior backups'))
 return result
