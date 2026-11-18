"""Replay constant/learned validation policies and paired final-set comparisons."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
import torch
from .causal_cost_fit import select
from .evaluate import paired_ci
from .train import write_json
from .data import sha

NAMES=[f'{arm}-s{seed}' for arm in ('GF','LF') for seed in (0,1)]
FLAGS=['repeated_final_sets_exploratory','two_seeds','window_attention_differs_from_dense','conditional_suffix480','held_gate_single_edit_fixed_depth','canonical_all_expert_compute_not_generation','body_weights_node_only']


def calculate(base):
    base=Path(base);rows=[];validation={}
    for name in NAMES:
        root=base/'runs'/name;c=json.loads((root/'head/config.json').read_text());a=dict(np.load(root/'head/validation.npz'));stored=json.loads((root/'head/selection.json').read_text())
        for kind in ('learned','constant'):
            got=select(torch.as_tensor(a[kind+'_cost']),a,c['length'],c['window'],c['stride'])
            if got!=stored[kind]:raise ValueError('Validation policy replay mismatch')
        validation[name]=stored
        for domain in ('dclm','fwe'):
            a=dict(np.load(root/'eval'/domain/'summary.npz'))
            if not all(np.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite evaluation')
            if max(a['constant_edits'].max(),a['learned_edits'].max())>1:raise ValueError('Multiple edits')
            for kind in ('learned','constant'):
                if stored[kind]['selected']['depth'] is None:np.testing.assert_array_equal(a[kind],a['factual'])
            row=dict(name=name,domain=domain,sequences=len(a['ids']),means={k:float(a[k].mean(dtype=np.float64)) for k in ('factual','constant','learned','dense')},edit_fraction={k:float(a[k+'_edits'].mean()) for k in ('constant','learned')})
            row['learned_minus_factual']=paired_ci(a['learned']-a['factual'],a['clusters'])
            row['learned_minus_constant']=paired_ci(a['learned']-a['constant'],a['clusters'])
            row['learned_minus_dense']=paired_ci(a['learned']-a['dense'],a['clusters']);rows.append(row)
    criterion={arm:all(r['learned_minus_factual']['mean']<=-.001 and r['learned_minus_constant']['mean']<=-.001 for r in rows if r['name'].startswith(arm)) for arm in ('GF','LF')}
    return dict(rows=rows,validation=validation,criterion=criterion,flags=FLAGS)


def report(base,backup):
    base=Path(base);backup=Path(backup);files=[];copies={}
    for name in NAMES:
        root=base/'runs'/name
        for folder in ('cache-0','cache-1','head','eval/dclm','eval/fwe'):
            path=root/folder;done=json.loads((path/'complete.json').read_text())
            if done['status']!='complete':raise ValueError('Stage incomplete')
            for rel,h in done['files'].items():
                if sha(path/rel)!=h:raise ValueError('Changed artifact')
            if folder=='head' and sha(path/'fit/fullstate.pt')!=done['fullstate_sha256']:raise ValueError('Changed head fullstate')
            if folder.startswith('eval') and any(q['canonical_causal_max']!=0 for q in done['proof'].values()):raise ValueError('Causal gate failed')
            files += [p for p in path.iterdir() if p.is_file() and not (folder.startswith('cache') and p.suffix=='.npz')]
    result=calculate(base);result['node_replay']='PASS';write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['name','domain','sequences','factual','constant','learned','dense','learned_edit_fraction'],lineterminator='\n');w.writeheader()
        for r in result['rows']:w.writerow(dict(name=r['name'],domain=r['domain'],sequences=r['sequences'],**r['means'],learned_edit_fraction=r['edit_fraction']['learned']))
    (base/'REPORT.md').write_text('# Causal cost router\n\nFrozen LLM, one causal edit at a validation-selected fixed depth.\n\nCriterion: '+json.dumps(result['criterion'])+'\n\nFlags: '+', '.join(FLAGS)+'\n')
    files += [base/p for p in ('REPORT.json','RESULTS.csv','REPORT.md','inputs/complete.json','execution/source.json','execution/admission.json')]
    members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json')
    archive=base/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if archive.stat().st_size>=25_000_000:raise ValueError('Small archive cap')
    backup.mkdir(parents=True,exist_ok=False)
    for name in NAMES:
        for rel in ('head/router.pt','head/fit/fullstate.pt'):
            src=base/'runs'/name/rel;dst=backup/name/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            if sha(src)!=sha(dst):raise ValueError('Backup mismatch')
            copies[str(dst)]=sha(dst)
    shutil.copyfile(archive,backup/'results.tar.gz');copies[str(backup/'results.tar.gz')]=sha(backup/'results.tar.gz')
    if copies[str(backup/'results.tar.gz')]!=sha(archive):raise ValueError('Archive copy mismatch')
    write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node copies; parent LLM weights retain verified FiLM backups'))
    return result
