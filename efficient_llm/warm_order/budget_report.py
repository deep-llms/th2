"""Counterfactual replay and fixed budget comparisons; preserve all prior results."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .data import sha
from .train import write_json
from .evaluate import paired_ci
from .budget_eval import BUDGETS,check_hashes
NAMES=[f'{arm}-s{s}' for arm in ('F7','J7') for s in (0,1)]
FLAGS=['posthoc_exploratory_reused_final','two_seeds','fixed_global_validation_thresholds','three_budgets_no_final_selection','padded_queries_and_full_KV_counted','fullcontext_not_generation_speed','body_weights_unchanged_node_only']

def replay(run):
 run=Path(run);cfg=json.loads((run/'config.json').read_text());done=json.loads((run/'complete.json').read_text());check_hashes(cfg['hashes'])
 if done['status']!='complete':raise ValueError('Incomplete')
 for p,h in done['files'].items():
  if sha(run/p)!=h:raise ValueError('Changed output')
 for domain in ('dclm','fwe'):
  a=np.load(run/domain/'summary.npz');prior=Path(cfg['prior'])/domain;c=np.load(prior/'costs.npy',mmap_mode='r');old=np.load(prior/'summary.npz');score=np.load(run/domain/'scores.npy')
  for k in ('ids','clusters','base','refine','oracle'):np.testing.assert_array_equal(a[k],old[k])
  np.testing.assert_array_equal(a['c6'],np.load(cfg['c6'][domain])['dense'])
  for q in BUDGETS:
   k=str(q);ch=np.load(run/domain/f'choices-{k}.npy');random=np.load(run/domain/f'random-{k}.npy')
   np.testing.assert_array_equal(ch,score>done['calibration'][k]['threshold'])
   for label,mask in [('learned',ch),('random',random)]:np.testing.assert_array_equal(a[label+'-'+k],np.where(mask,c[:,:,1],c[:,:,0]).mean(1,dtype=np.float64))
   np.testing.assert_array_equal(a['count-'+k],ch.sum(1));np.testing.assert_allclose(a['actual-'+k],a['learned-'+k],atol=2e-4,rtol=0)
 return done


def calculate(arrays,done):
 rows=[];ci={};criteria={}
 for name in NAMES:
  for d in ('dclm','fwe'):
   a=arrays[name,d]
   for key,v in a.items():
    if key in ('ids','clusters') or key.startswith('count-'):continue
    rows.append(dict(name=name,domain=d,metric=key,nll=float(v.mean()),sequences=len(v)))
   for q in BUDGETS:
    k=str(q)
    for label in ('c6','refine','random-'+k):ci[f'{name}/{d}/{k}/minus-{label}']=paired_ci(a['learned-'+k]-a[label],a['clusters'])
 for arm in ('F7','J7'):
  for q in BUDGETS:
   k=str(q);gain=[];quality=[];adaptive=[];speed=[]
   for seed in (0,1):
    name=f'{arm}-s{seed}'
    for d in ('dclm','fwe'):
     vals=ci[f'{name}/{d}/{k}/minus-random-'+k]['mean'];gain.append(vals<=-.001);quality.append(ci[f'{name}/{d}/{k}/minus-c6']['mean']<=-.01)
     work=done[name]['work'][d][k];rate=work['logical_queries']/work['router_tokens'];adaptive.append(ci[f'{name}/{d}/{k}/minus-refine']['mean']<=.01 and rate<=.75)
     timing=work['latency']['seconds']['body'];speed.append(timing['sparse']<=.95*timing['dense'])
   criteria[f'{arm}/{k}']=dict(learned_gain_vs_random=all(gain),quality_vs_c6=all(quality),adaptive_tradeoff=all(adaptive),body_latency_signal=all(speed))
 return dict(rows=rows,comparisons=ci,criteria=criteria,flags=FLAGS)


def report(base,backup):
 base=Path(base);backup=Path(backup);arrays={};done={};files=[]
 for name in NAMES:
  r=base/'runs'/name;done[name]=replay(r)
  for d in ('dclm','fwe'):arrays[name,d]=dict(np.load(r/d/'summary.npz'))
  files += [p for p in r.rglob('*') if p.is_file() and (p.name in ('summary.npz','router.pt') or p.suffix=='.json' and p.name!='progress.json')]
 result=calculate(arrays,done);result.update(runs=done,node_raw_replay='PASS');write_json(base/'REPORT.json',result)
 with (base/'RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
 lines=['# Frozen-body budgeted refinement','','Four gain routers; thresholds calibrated on validation for25/50/75% target-query budgets. No backbone updates. All budgets reported; final sets are reused and exploratory.','','| Run | Domain | Metric | NLL |','|---|---|---|---:|']
 lines += [f"| {r['name']} | {r['domain']} | {r['metric']} | {r['nll']:.6f} |" for r in result['rows']]
 lines += ['','Criteria: '+json.dumps(result['criteria']),'','Body-only and CE-inclusive full-context latency, padded query/KV counts and raw replay are recorded in REPORT.json. Adaptive tradeoff alone is not proof that routing beats random or dense C6. No generation-speed claim.','', 'Flags: '+', '.join(FLAGS)]
 (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
 files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')]
 members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json');archive=base/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
 if archive.stat().st_size>=25_000_000:raise ValueError('Small archive cap')
 backup.mkdir(parents=True,exist_ok=False);copies={}
 for p in [archive,*sorted((base/'runs').glob('*/head/fullstate.pt'))]:
  dst=backup/p.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
  if sha(p)!=sha(dst):raise ValueError('Backup mismatch')
  copies[str(dst)]=sha(dst)
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same node; unchanged body checkpoints retain verified previous backups'))
 return result
