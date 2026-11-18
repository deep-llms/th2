"""Node-side replay from raw token costs; compact per-sequence scientific export."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .data import sha
from .train import write_json
from .evaluate import paired_ci

NAMES=[f'{arm}-s{seed}' for arm in ('D','R','W') for seed in (0,1)]
FLAGS=['exploratory_reused_final_sets','two_seeds','conditional_32_prefix_480_targets',
       'all_eight_route_forwards_not_efficient_dispatch','unequal_total_compute',
       'token_oracle_not_deployable','raw_features_costs_backbone_weights_node_only']


def replay_run(run):
    run=Path(run);done=json.loads((run/'complete.json').read_text())
    if done['status']!='complete':raise ValueError('Incomplete job')
    for rel,want in done['files'].items():
        if sha(run/rel)!=want:raise ValueError('Changed run artifact '+rel)
    cfg=json.loads((run/'config.json').read_text())
    if sha(cfg['weights'])!=done['frozen_weights_sha256']:raise ValueError('Backbone changed')
    constant=json.loads((run/'selection.json').read_text())['constant'];reports={}
    for domain in ('dclm','fwe'):
        folder=run/domain;raw=np.load(folder/'costs.npy',mmap_mode='r');data=np.load(folder/'summary.npz')
        c=np.asarray(raw,dtype=np.float64);n,t,k=c.shape
        expected={'canonical':c[:,:,0].sum(1)/t}
        if k==8:
            expected.update(constant=c[:,:,constant].sum(1)/t,uniform=c.sum((1,2))/(t*k),oracle=c.min(2).sum(1)/t)
            low=c.min(2);expected['ensemble']=(low-np.log(np.exp(low[:,:,None]-c).sum(2)/8)).sum(1)/t
            for key in ('learned','prefix','token_id'):
                ch=np.load(folder/(key+'.npy'));selected=c[np.arange(n)[:,None],np.arange(t)[None,:],ch]
                expected[key]=selected.sum(1)/t
                counts=np.stack([np.sum(ch==j,axis=1) for j in range(8)],1)
                np.testing.assert_array_equal(counts,data[key+'_counts'])
                np.testing.assert_array_equal((ch[:,1:]!=ch[:,:-1]).sum(1),data[key+'_switches'])
        proof=json.loads((folder/'complete.json').read_text())
        for key,v in expected.items():
            np.testing.assert_allclose(v,data[key],atol=1e-12,rtol=0)
            if abs(float(v.mean())-proof['metrics'][key])>1e-12:raise ValueError('Metric mismatch')
        if proof['targets']!=n*t:raise ValueError('Wrong denominator')
        reports[domain]=proof
    return done,reports


def report(base,backup):
    base=Path(base);backup=Path(backup);rows=[];arrays={};timings={};files=[]
    for name in NAMES:
        done,reports=replay_run(base/'runs'/name);timings[name]={k:done[k] for k in ('seconds','full_path_sequence_forwards','extra_stem_sequence_forwards')}
        for domain,r in reports.items():
            arrays[name,domain]=np.load(base/'runs'/name/domain/'summary.npz')
            for metric,value in r['metrics'].items():rows.append(dict(name=name,domain=domain,metric=metric,nll=value,sequences=r['sequences'],tokens=r['targets']))
        run=base/'runs'/name
        files.extend(p for p in run.rglob('*') if p.is_file() and (p.suffix=='.json' and p.name!='progress.json' or p.name in ('summary.npz','router.pt','token_id_lookup.npy')))
    comparisons={};promising={}
    for arm in ('R','W'):
        checks=[]
        for seed in (0,1):
            for domain in ('dclm','fwe'):
                a=arrays[f'{arm}-s{seed}',domain];d=arrays[f'D-s{seed}',domain]
                np.testing.assert_array_equal(a['ids'],d['ids']);np.testing.assert_array_equal(a['clusters'],d['clusters'])
                for other,v in [('dense',d['canonical']),('constant',a['constant']),('prefix',a['prefix']),('token_id',a['token_id'])]:
                    ci=paired_ci(a['learned']-v,a['clusters']);comparisons[f'{arm}-s{seed}/{domain}/learned-minus-{other}']=ci
                    if other in ('dense','constant'):checks.append(ci['mean']<=-.01)
        promising[arm]=all(checks)
    means={}
    for arm in ('D','R','W'):
        for domain in ('dclm','fwe'):
            for metric in sorted({r['metric'] for r in rows if r['name'].startswith(arm) and r['domain']==domain}):
                means[f'{arm}/{domain}/{metric}']=float(np.mean([r['nll'] for r in rows if r['name'].startswith(arm) and r['domain']==domain and r['metric']==metric]))
    result=dict(rows=rows,means=means,comparisons=comparisons,promising_screen=promising,timings=timings,flags=FLAGS,node_raw_replay='PASS')
    write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Token-wise causal order-router screen','','Frozen backbones; hard router selects each next-token prediction from eight full causal order experts. All-route compute, not an efficient mixed-path implementation.','','| Model | Domain | Dense/canonical | Constant | Prefix | Token ID | Learned | Oracle | Ensemble |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('D','R','W'):
        for domain in ('dclm','fwe'):
            lines.append('| '+arm+' | '+domain+' | '+' | '.join(f"{means[f'{arm}/{domain}/{m}']:.6f}" if f'{arm}/{domain}/{m}' in means else '—' for m in ['canonical','constant','prefix','token_id','learned','oracle','ensemble'])+' |')
    lines+=['','Promising screen (both seeds and domains, learned at least0.01nats better than dense AND constant): '+json.dumps(promising)+'.','','Paired document-cluster intervals, exact per-seed numbers and timings: REPORT.json. Oracle sees the true target and cannot be deployed. Ensemble is a probability mixture, not hard routing.','','Flags: '+', '.join(FLAGS)+'.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files += [base/'REPORT.json',base/'REPORT.md',base/'RESULTS.csv',base/'inputs/complete.json',base/'inputs/indices.npz',base/'execution/source.json',base/'execution/admission.json']
    manifest={str(p.relative_to(base)):sha(p) for p in files}
    write_json(base/'export-members.json',manifest);files.append(base/'export-members.json')
    archive=base/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if archive.stat().st_size>=25_000_000:raise ValueError('Export exceeds small-results ceiling; do not pull')
    backup.mkdir(parents=True,exist_ok=False);copied={}
    for src in [archive,*sorted((base/'runs').glob('*/head/fullstate.pt'))]:
        rel=str(src.relative_to(base));dst=backup/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        if sha(src)!=sha(dst):raise ValueError('Backup mismatch')
        copied[str(dst)]=sha(dst)
    write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=manifest,node_copies=copied,
        backbone_references={name:json.loads((base/'runs'/name/'complete.json').read_text())['frozen_weights_sha256'] for name in NAMES},
        durability='Same-node backup; original backbone checkpoints already preserved; raw cost/feature caches remain node-only'))
    return result
