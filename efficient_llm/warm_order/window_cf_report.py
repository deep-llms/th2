"""Replay validation-only single-intervention oracle; never call it a causal policy."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .window_cf_prepare import NAMES
from .evaluate import paired_ci
from .strength_refinement import check
from .train import write_json
from .data import sha
FLAGS=['window_attention_differs_from_dense','label_using_single_intervention_oracle_not_deployable','validation_only_reused_exploratory','two_seeds','no_new_training','canonical_all_expert_compute','weights_node_only']

def validate(a):
 for v in a.values():
  if not np.isfinite(v).all():raise ValueError('Nonfinite costs')
 if a['costs'].shape!=(len(a['ids']),len(a['actions'])) or len(a['nll'])!=len(a['ids']):raise ValueError('Cost shape')
 decision=[]
 for d,j,k in a['actions']:
  if (int(d),int(j)) not in decision:decision.append((int(d),int(j)))
 for index,(d,j,k) in enumerate(a['actions']):
  col=decision.index((int(d),int(j)));mask=a['routes'][:,col]==k
  np.testing.assert_array_equal(a['costs'][mask,index],a['nll'][mask])

def calculate(arrays):
 rows=[];headroom={}
 for name in NAMES:
  a=arrays[name];validate(a);oracle=np.minimum(a['nll'],a['costs'].min(1));gain=a['nll']-oracle;win=gain>1e-4;best=a['costs'].argmin(1)
  rows.append(dict(name=name,sequences=len(gain),factual_nll=float(a['nll'].mean(dtype=np.float64)),dense_nll=float(a['dense_nll'].mean(dtype=np.float64)),single_edit_oracle_nll=float(oracle.mean(dtype=np.float64)),gain_ci=paired_ci(gain,a['clusters']),oracle_minus_dense_ci=paired_ci(oracle-a['dense_nll'],a['clusters']),gain_gt_1e4_fraction=float(win.mean()),best_action_counts=np.bincount(best[win],minlength=len(a['actions'])).tolist()))
  seed=int(name.split('-')[1][1:]);control=arrays[f'GF-s{seed}-hold'];np.testing.assert_array_equal(a['ids'],control['ids']);np.testing.assert_array_equal(a['clusters'],control['clusters']);np.testing.assert_array_equal(a['dense_nll'],control['dense_nll'])
 for arm in ('GF','LF'):
  for seed in (0,1):np.testing.assert_array_equal(arrays[f'{arm}-s{seed}-hold']['nll'],arrays[f'{arm}-s{seed}-rescore']['nll'])
 for mode in ('hold','rescore'):headroom[mode]=all(next(r for r in rows if r['name']==f'LF-s{s}-{mode}')['gain_ci']['mean']>.005 for s in (0,1))
 return dict(rows=rows,LF_both_seeds_headroom_gt_005=headroom,flags=FLAGS)

def replay_run(out):
 out=Path(out);c=json.loads((out/'config.json').read_text());d=json.loads((out/'complete.json').read_text());check(c['hashes'])
 if d['status']!='complete' or not d['identity_exact'] or d['probe']['canonical_causal_max']!=0:raise ValueError('Incomplete diagnostic')
 for p,h in d['files'].items():
  if sha(out/p)!=h:raise ValueError('Changed artifact')
 a=dict(np.load(out/'costs.npz'));validate(a);return a,d

def report(base,backup):
 base=Path(base);backup=Path(backup);arrays={};done={};files=[]
 for n in NAMES:
  arrays[n],done[n]=replay_run(base/'runs'/n);files += [p for p in (base/'runs'/n).iterdir() if p.is_file()]
 result=calculate(arrays);result.update(runs=done,node_replay='PASS');write_json(base/'REPORT.json',result)
 with (base/'RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['name','sequences','factual_nll','dense_nll','single_edit_oracle_nll','mean_gain','gain_gt_1e4_fraction'],lineterminator='\n');w.writeheader()
  for r in result['rows']:w.writerow({**{k:r[k] for k in w.fieldnames if k!='mean_gain'},'mean_gain':r['gain_ci']['mean']})
 (base/'REPORT.md').write_text('# Single-window counterfactual diagnostic\n\nLabel-using oracle, not a deployable router. Validation-only.\n\n'+json.dumps(result['LF_both_seeds_headroom_gt_005'])+'\n\nFlags: '+', '.join(FLAGS)+'\n')
 files += [base/p for p in ['REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','execution/source.json','execution/admission.json']];members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json');archive=base/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
 if archive.stat().st_size>=25000000:raise ValueError('Small archive cap')
 backup.mkdir(parents=True,exist_ok=False);target=backup/'results.tar.gz';shutil.copyfile(archive,target)
 if sha(target)!=sha(archive):raise ValueError('Backup mismatch')
 write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies={str(target):sha(target)},durability='Same-node archive copy; frozen parent weights retain their prior verified backups'))
 return result
