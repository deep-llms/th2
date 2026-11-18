"""Report all LR settings; select exclusively by validation and replay raw costs."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .strength_refinement import NAMES,check
from .refinement_eval import check_train
from .evaluate import paired_ci
from .train import write_json
from .data import sha
FLAGS=['posthoc_LR_screen','reused_final_exploratory','two_seeds','equal_tokens_not_compute','validation_only_LR_selection','oracle_not_achievable_router','weights_fullstate_node_only']


def calculate(arrays):
 rows=[];comparisons={};selected={};selection=[]
 for s in (0,1):
  choices={'low_lr':float(arrays[f'H3-s{s}','valid']['low_lr'].mean())}
  choices.update({f'{arm}-s{s}':float(arrays[f'{arm}-s{s}','valid']['refine'].mean()) for arm in ('H3','H10')})
  selected[str(s)]={'validation_nll':choices,'name':min(choices,key=choices.get)}
 for name in NAMES:
  for domain in ('valid','dclm','fwe'):
   a=arrays[name,domain]
   for metric,v in a.items():
    if metric not in ('ids','clusters'):rows.append(dict(name=name,domain=domain,metric=metric,nll=float(v.mean()),sequences=len(v)))
   if domain!='valid':
    for metric in ('c6','base','low_lr'):comparisons[f'{name}/{domain}/minus-{metric}']=paired_ci(a['refine']-a[metric],a['clusters'])
 for s in (0,1):
  name=selected[str(s)]['name']
  for domain in ('dclm','fwe'):
   a=arrays[f'H3-s{s}' if name=='low_lr' else name,domain];value=a['low_lr'] if name=='low_lr' else a['refine']
   q=paired_ci(value-a['c6'],a['clusters']);comparisons[f'selected-s{s}/{domain}/minus-c6']=q;selection.append(q['mean']<=-.01)
 return dict(rows=rows,comparisons=comparisons,selected=selected,quality_signal_all_seeds_domains=all(selection),flags=FLAGS)


def report(base,backup):
 base=Path(base);backup=Path(backup);arrays={};runs={};files=[]
 for name in NAMES:
  out=base/'runs'/name;c=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text());check(c['hashes']);check_train(out/'train')
  if done['status']!='complete':raise ValueError('Incomplete job')
  for p,h in done['files'].items():
   if sha(out/p)!=h:raise ValueError('Changed result')
  for d in ('valid','dclm','fwe'):
   a=dict(np.load(out/d/'summary.npz'));v=np.load(out/d/'costs.npy',mmap_mode='r')
   for key,val in [('base',v[:,:,0]),('refine',v[:,:,1]),('oracle',v.min(-1))]:np.testing.assert_array_equal(a[key],val.mean(1,dtype=np.float64))
   old=Path(c['prior'])/d
   low=np.load(old/'costs.npy',mmap_mode='r')[:,:,1].mean(1,dtype=np.float64) if d=='valid' else np.load(old/'summary.npz')['refine']
   np.testing.assert_array_equal(a['low_lr'],low)
   if d!='valid':np.testing.assert_array_equal(a['c6'],np.load(c['c6'][d])['dense'])
   arrays[name,d]=a
  runs[name]=done
  files += [p for p in out.rglob('*') if p.is_file() and (p.name=='summary.npz' or p.suffix in ('.json','.jsonl'))]
 result=calculate(arrays);result.update(runs=runs,node_raw_replay='PASS');write_json(base/'REPORT.json',result)
 with (base/'RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
 lines=['# Frozen seventh-block LR screen','','All new LRs reported; selection only by512 validation rows. Old low-LR and C6 controls reused without rerunning.','','| Run | Split | Metric | NLL |','|---|---|---|---:|']
 lines += [f"| {r['name']} | {r['domain']} | {r['metric']} | {r['nll']:.6f} |" for r in result['rows']]
 lines+=['','Selection: '+json.dumps(result['selected']),'','Quality signal all seeds/domains: '+str(result['quality_signal_all_seeds_domains']),'','Flags: '+', '.join(FLAGS)]
 (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
 files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json')]
 members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json');archive=base/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
 if archive.stat().st_size>=25_000_000:raise ValueError('Compact export limit')
 backup.mkdir(parents=True,exist_ok=False);copies={}
 for p in [archive,*sorted((base/'runs').glob('*/train/*.pt'))]:
  dst=backup/p.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
  if sha(dst)!=sha(p):raise ValueError('Backup failure')
  copies[str(dst)]=sha(dst)
 if len(copies)!=9:raise ValueError('Incomplete backup inventory')
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node verified copies; weights not exported'))
 return result
