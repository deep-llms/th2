"""Independent node raw replay and small result export for component/feature diagnostic."""
import csv,json,shutil,tarfile
from pathlib import Path
import numpy as np
from .data import sha
from .train import write_json
from .evaluate import paired_ci

NAMES=[f'{k}-s{s}' for k in ('attention','ffn','block') for s in (0,1)]+[f'{k}-deep-s{s}' for k in ('R','W') for s in (0,1)]
FLAGS=['exploratory_reused_final_sets','two_seeds','conditional_32_prefix_480_targets','deep_features_extra_full_canonical_compute','inherited_all8_expert_cost','token_oracle_not_deployable','weights_features_costs_node_only']


def replay(run):
    run=Path(run);done=json.loads((run/'complete.json').read_text());cfg=json.loads((run/'config.json').read_text())
    if done['status']!='complete':raise ValueError('Incomplete')
    for rel,want in done['files'].items():
        if sha(run/rel)!=want:raise ValueError('Changed artifact '+rel)
    for p,want in cfg['hashes'].items():
        if sha(p)!=want:raise ValueError('Changed input '+p)
    for domain in ('dclm','fwe'):
        a=np.load(run/domain/'summary.npz');prior=Path(cfg['prior'])/domain
        if cfg['kind']=='deep':
            c=np.load(prior/'costs.npy',mmap_mode='r');ch=np.load(run/domain/'choices.npy');old=np.load(prior/'summary.npz')
            expected=np.asarray(c[np.arange(len(c))[:,None],np.arange(c.shape[1])[None,:],ch],dtype=np.float64).sum(1)/c.shape[1]
            np.testing.assert_allclose(a['deep'],expected,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(a['deep_counts'],np.stack([np.sum(ch==j,axis=1) for j in range(8)],1))
            np.testing.assert_array_equal(a['deep_switches'],(ch[:,1:]!=ch[:,:-1]).sum(1))
            for key in old.files:np.testing.assert_array_equal(a[key.replace('learned','stem')],old[key])
        else:
            c=np.load(run/domain/'costs.npy',mmap_mode='r')
            for j in range(4):np.testing.assert_allclose(a[f'path{j}'],np.asarray(c[:,:,j],dtype=np.float64).sum(1)/c.shape[1],rtol=0,atol=1e-12)
            old=np.load(prior/'summary.npz')
            np.testing.assert_array_equal(a['ids'],old['ids']);np.testing.assert_array_equal(a['clusters'],old['clusters'])
            np.testing.assert_allclose(a['path0'],old['canonical'],rtol=0,atol=2e-4)
    return done,cfg


def calculate(arrays):
    rows=[];comparisons={};checks={'deep_R':[],'deep_W':[],'ffn':[],'beats_dense_R':[],'beats_dense_W':[]}
    for name in NAMES:
        for domain in ('dclm','fwe'):
            a=arrays[name,domain];n=len(a['ids'])
            keys=[k for k in a if k not in ('ids','clusters') and not k.endswith(('_counts','_switches'))]
            for k in keys:rows.append(dict(name=name,domain=domain,metric=k,nll=float(np.mean(a[k])),sequences=n))
            if '-deep-' in name:
                arm=name[0];seed=int(name[-1]);dense=arrays[f'block-s{seed}',domain]
                np.testing.assert_array_equal(a['ids'],dense['ids']);np.testing.assert_array_equal(a['clusters'],dense['clusters'])
                for k in ('stem','constant','prefix','token_id','dense'):
                    delta=a['deep']-(dense['path0'] if k=='dense' else a[k]);ci=paired_ci(delta,a['clusters']);comparisons[f'{name}/{domain}/deep-minus-{k}']=ci
                    if k=='stem':checks['deep_'+arm].append(ci['mean']<=-.01)
                    if k=='dense':checks['beats_dense_'+arm].append(ci['mean']<=-.01)
            else:
                for j in (1,2,3):comparisons[f'{name}/{domain}/path{j}-minus-canonical']=paired_ci(a[f'path{j}']-a['path0'],a['clusters'])
    for s in (0,1):
        for domain in ('dclm','fwe'):
            penalties={}
            for k in ('attention','ffn','block'):
                a=arrays[f'{k}-s{s}',domain];penalties[k]=float(np.mean([np.mean(a[f'path{j}']-a['path0']) for j in (1,2,3)]))
            checks['ffn'].append(penalties['ffn']<.02 and penalties['ffn']<=.5*min(penalties['attention'],penalties['block']))
    return dict(rows=rows,comparisons=comparisons,promising={k:all(v) for k,v in checks.items()},flags=FLAGS)


def report(base,backup):
    base=Path(base);backup=Path(backup);arrays={};files=[];timings={}
    for name in NAMES:
        run=base/'runs'/name;done,cfg=replay(run);timings[name]={k:done[k] for k in ('seconds','head_seconds','main_sequence_forwards','probe_note')}
        for domain in ('dclm','fwe'):arrays[name,domain]=dict(np.load(run/domain/'summary.npz'))
        files += [p for p in run.rglob('*') if p.is_file() and (p.name in ('summary.npz','router.pt') or p.suffix=='.json' and p.name!='progress.json')]
    result=calculate(arrays);result.update(timings=timings,node_raw_replay='PASS');write_json(base/'REPORT.json',result)
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    lines=['# Component and deep-feature diagnostic','','Frozen backbones; no backbone updates. Canonical deep features require an extra full path and inherited eight-expert computation. Exploratory reused final sets.','','| Job | Domain | Metric | NLL |','|---|---|---|---:|']
    lines += [f"| {r['name']} | {r['domain']} | {r['metric']} | {r['nll']:.6f} |" for r in result['rows']]
    lines += ['','Registered next-step flags: '+json.dumps(result['promising']), '', 'Detailed per-seed paired document-cluster intervals/timings: REPORT.json. No training-seed confidence claim.','', 'Flags: '+', '.join(FLAGS)+'.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files += [base/p for p in ('REPORT.json','REPORT.md','RESULTS.csv','inputs/complete.json','inputs/indices.npz','execution/source.json','execution/admission.json')]
    members={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'export-members.json',members);files.append(base/'export-members.json')
    archive=base/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if archive.stat().st_size>=25_000_000:raise ValueError('Small export limit')
    backup.mkdir(parents=True,exist_ok=False);copied={}
    for p in [archive,*sorted((base/'runs').glob('*/head/fullstate.pt'))]:
        dst=backup/p.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        if sha(p)!=sha(dst):raise ValueError('Backup mismatch')
        copied[str(dst)]=sha(dst)
    write_json(base/'backup.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=members,node_copies=copied,durability='Same-node only; original backbone checkpoints unchanged and previously backed up'))
    return result
