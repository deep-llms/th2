"""Raw replay, fixed criteria, compact export and exact checkpoint backups."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .data import sha
from .train import write_json
from .evaluate import paired_ci
from .refinement_eval import check_train,summarize
NAMES=[f'{a}-s{s}' for a in ('C6','F7','J7') for s in (0,1)]
FLAGS=['conditional_32_prefix_480_targets','two_seeds','equal_input_tokens_not_equal_compute','weights_fullstate_features_node_only','full_context_latency_not_generation_speed','training_documents_may_recur']


def replay(run):
    run=Path(run);done=json.loads((run/'complete.json').read_text());cfg=json.loads((run/'config.json').read_text())
    if done['status']!='complete':raise ValueError('Incomplete eval')
    check_train(cfg['run'])
    for p,h in cfg['hashes'].items():
        if sha(p)!=h:raise ValueError('Changed input')
    for p,h in done['files'].items():
        if sha(run/p)!=h:raise ValueError('Changed output '+p)
    for d in ('dclm','fwe'):
        a=np.load(run/d/'summary.npz');c=np.load(run/d/'costs.npy',mmap_mode='r')
        if done['arm']=='C6':np.testing.assert_array_equal(a['dense'],c)
        else:
            ch=np.load(run/d/'choices.npy');rnd=np.load(run/d/'random.npy')
            for k,v in summarize(c,ch,rnd).items():np.testing.assert_allclose(a[k],v,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(a['actual_sparse'],np.load(run/d/'actual.npy'))
            np.testing.assert_allclose(a['learned'],a['actual_sparse'],atol=2e-4,rtol=0)
    return done,cfg


def calculate(arrays):
    result=dict(rows=[],comparisons={},criteria={})
    for name in NAMES:
        for d in ('dclm','fwe'):
            a=arrays[name,d];dense=arrays['C6-s'+name[-1],d]
            for key in ('ids','clusters'):np.testing.assert_array_equal(a[key],dense[key])
            for k,v in a.items():
                if k in ('ids','clusters','refine_count','target_count'):continue
                result['rows'].append(dict(name=name,domain=d,metric=k,nll=float(np.mean(v)),sequences=len(v)))
            if name.startswith('C6'):continue
            for label,v in [('C6',dense['dense']),('refine',a['refine']),('base',a['base']),('random',a['random'])]:result['comparisons'][f'{name}/{d}/learned-minus-{label}']=paired_ci(a['learned']-v,a['clusters'])
    for arm in ('F7','J7'):
        quality=[];adaptive=[]
        for s in (0,1):
            for d in ('dclm','fwe'):
                a=arrays[f'{arm}-s{s}',d];q=result['comparisons'][f'{arm}-s{s}/{d}/learned-minus-C6']['mean'];r=result['comparisons'][f'{arm}-s{s}/{d}/learned-minus-refine']['mean']
                quality.append(q<=-.01);adaptive.append(r<=.01 and float(a['refine_count'].mean())<=.75*float(a['target_count']))
        result['criteria'][arm]=dict(quality_all_seeds_domains=all(quality),adaptive_all_seeds_domains=all(adaptive))
    return result


def report(base,backup):
    base=Path(base);backup=Path(backup);arrays={};files=[];train={};evals={}
    cfg=json.loads((base/'inputs/eval-config.json').read_text())
    for name in NAMES:
        run=base/'runs'/name;ev=base/'eval'/name;train[name],c=check_train(run);evals[name],_=replay(ev)
        for d in ('dclm','fwe'):
            arrays[name,d]=dict(np.load(ev/d/'summary.npz'));arrays[name,d]['target_count']=np.asarray(c['model']['length']-c['model']['prefix'])
        files += [p for p in ev.rglob('*') if p.is_file() and (p.name in ('summary.npz','router.pt') or p.suffix=='.json' and p.name!='progress.json')]
        files += [run/p for p in ('complete.json','recipe.json','checkpoint.json','trace.jsonl')]
    result=calculate(arrays);result.update(train=train,eval=evals,flags=FLAGS+(['reused_final_exploratory'] if not all(cfg['final_fresh_documents'].values()) else []),final_fresh_documents=cfg['final_fresh_documents'],node_raw_replay='PASS')
    write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    lines=['# Causal adaptive final refinement','','Six fixed-token fits, two seeds; routing selected only on fit/validation. Input tokens are matched; FLOPs and wall time are not.','','| Arm/seed | Domain | Metric | NLL |','|---|---|---|---:|']
    lines += [f"| {r['name']} | {r['domain']} | {r['metric']} | {r['nll']:.6f} |" for r in result['rows']]
    lines += ['','Registered criteria: '+json.dumps(result['criteria']), '', 'Per-seed paired document intervals, sparse dispatch parity, measured full-context latency, query/KV counts and train times are in REPORT.json. No generation-speed or equal-compute claim.','', 'Flags: '+', '.join(result['flags'])]
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','inputs/eval-config.json','inputs/eval.npz','execution/source.json','execution/admission.json')]
    files += sorted(base.glob('gpu-probe-*.json'))
    members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json')
    archive=base/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if archive.stat().st_size>=25_000_000:raise ValueError('Small export limit')
    backup.mkdir(parents=True,exist_ok=False);copies={}
    artifacts=[archive,*sorted((base/'runs').glob('*/weights.pt')),*sorted((base/'runs').glob('*/fullstate.pt')),*sorted((base/'eval').glob('*/head/fullstate.pt'))]
    for p in artifacts:
        dst=backup/p.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);h=sha(p)
        if sha(dst)!=h:raise ValueError('Backup mismatch')
        copies[str(dst)]=h
    write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copies,durability='Same-node backup; only small archive/proof export'))
    return result
