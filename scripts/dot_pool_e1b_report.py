"""Additive v1.5 readout from saved results only; no model execution."""
import argparse,csv,json,tarfile,time
from pathlib import Path
import numpy as np
from efficient_llm.dot_pool import SPEC_SHA
from efficient_llm.dot_pool.data import sha,write
from scripts.dot_pool_e1b_analysis import same_relation


def table(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(path):return json.loads(path.read_text())
def report(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=False);index=read(root/'run_index.json');budget=read(root/'budget.json')
    flags=list(index['flags']);rows=[];reentry=[];curves={};counts={};proof=[];artifacts=[]
    for c in index['runs']:
        run=Path(c['output']);warm=Path(c['first200_output']);name=c['name']
        if not (run/'train_complete.json').exists():flags.append(name+' incomplete');continue
        cfg=read(run/'config.json');done=read(run/'train_complete.json');final=read(run/'final_eval.json')
        if done['step']!=budget['epochs']*77 or cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Endpoint/protocol mismatch')
        if cfg['model']['stem']!=0 or cfg['model']['coda']!=0 or cfg['model']['middle']!=8:raise ValueError('Architecture mismatch')
        metrics=[]
        for stage in [warm,run]:
            for p in stage.glob('eval-*.json'):
                v=read(p);metrics.append((v['step']/77,v['metrics']['id_dev']['accuracy'],v['metrics']['ood_dev']['accuracy']))
        metrics.sort();curves[name]=metrics;lookup={int(e):o for e,i,o in metrics}
        n=budget['epochs'];cross=next((e for e,i,o in metrics if o>=.5),None)
        if n not in lookup or n-500 not in lookup:raise ValueError('Missing final500 dev evidence')
        f=final['metrics'];rows.append(dict(arm=c['arm'],seed=c['seed'],epochs=n,ID_test=f['id_test']['accuracy'],OOD_test=f['ood_test']['accuracy'],trained_fact_accuracy=f['train']['accuracy'],OOD_atomic_accuracy=f['ood_atoms']['accuracy'],epoch_OOD50=cross if cross is not None else 'never',OOD_dev_change_last500=lookup[n]-lookup[n-500],wall_seconds=done['elapsed']+read(warm/'train_complete.json')['elapsed']))
        counts[name]=cfg['counts']
        if c['arm'] in ['SO','DOT']:
            r=read(run/'reentry.json');same=read(run/'reentry_same_relation.json')
            q=np.load(Path(c['data'])/'ood_test.npy');b=np.load(Path(c['data'])/'ood_bridge.npy')
            with np.load(run/'routes_sample.npz') as routes:
                for split in ['id_dev','ood_dev','id_test','ood_test']:
                    if abs(float(routes[split+'_correct'].mean())-f[split]['accuracy'])>1e-12:raise ValueError('Route accuracy mismatch')
                replay=same_relation(routes['ood_test_ids'][:,4:8],routes['ood_bridge_ids'][:,:4],q[:,2],b[:,0],routes['ood_test_correct'])
            if replay['groups']!=same['groups']:raise ValueError('R4b replay differs')
            for group in ['all','correct','wrong']:
                rr=r['groups'][group];ss=same['groups'][group]
                reentry.append(dict(run=name,group=group,R4_excess=rr['excess'],R4_ci=json.dumps(rr['ci']),R4_label='re-entry observed' if rr['ci'] and rr['ci'][0]>0 else 're-entry not observed',R4b_excess=ss['excess_same'],R4b_ci=json.dumps(ss['ci']),R4b_label=ss['label'],R4b_eligible=ss['n'],R4b_excluded=ss['excluded_no_partner']))
            proof.append(dict(run=name,route_accuracy=True,R4b_replay=True))
        manifest=read(run/'artifact_manifest.json');artifacts+=manifest['artifacts']
        for item in manifest['artifacts']:
            if item['tier']==1 and sha(item['path'])!=item['sha256']:raise ValueError('Tier1 hash changed')
        (out/(name+'-config.json')).write_text((run/'config.json').read_text())
    table(out/'e1b_summary.csv',rows)
    if reentry:table(out/'e1b_reentry.csv',reentry)
    pilot=[]
    for p in sorted((root/'pilot_same_relation').glob('*.json')):
        v=read(p)
        for group,z in v['groups'].items():pilot.append(dict(run=p.stem,group=group,excess_same=z['excess_same'],ci=json.dumps(z['ci']),eligible=z['n'],excluded=z['excluded_no_partner'],label=z['label']))
    table(out/'e1_reentry_same_relation.csv',pilot)
    write(out/'learning_curves.json',curves);write(out/'parameter_counts.json',counts)
    # Counts use the same analytical definition as the pilot; FLOPs are estimates.
    flop_rows=[dict(run=name,nonembedding_total=v['nonembedding_total'],nonembedding_active=v['nonembedding_active'],forward_linear_FLOPs_per_token_estimate=2*v['nonembedding_active'],note='2*active nonembedding parameters; excludes attention quadratic, embedding/head, routing/Taylor overhead') for name,v in counts.items()]
    table(out/'parameter_flop_estimates.csv',flop_rows)
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for name,vals in curves.items():
            x=np.array(vals)
            for k,ax in enumerate(axes):ax.plot(x[:,0],x[:,k+1],label=name)
        for ax,title in zip(axes,['ID dev','OOD dev']):ax.set(xlabel='Epoch',ylabel='Accuracy',title=title,ylim=(0,1));ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'e1b_learning_curves.png',dpi=130);plt.close(fig)
    except ImportError:
        # No package installs from GPU jobs. Plot is deferred, numerical curves kept.
        flags.append('PNG plot pending: matplotlib absent; learning_curves.json retained')
    means={a:float(np.mean([r['OOD_test'] for r in rows if r['arm']==a])) for a in ['D','L','SO','DOT'] if sum(r['arm']==a for r in rows)==2}
    readings={}
    if len(means)==4:
        d_id=np.mean([r['ID_test'] for r in rows if r['arm']=='D'])
        loop_pass=means['L']>=means['D']+.2
        r0=d_id>=.9 and means['D']<.2 and loop_pass
        readings['R0']='setup passes' if r0 else 'inconclusive: composition did not emerge within the budget' if not loop_pass else 'dense setup control failed; loop improvement criterion passes'
        d=means['SO']-means['L'];readings['R1']=dict(delta=d,label='matches' if abs(d)<=.1 else 'better' if d>0 else 'worse')
        d=means['DOT']-means['SO'];readings['R2']=dict(delta=d,label='helps' if d>=.1 else 'hurts' if d<=-.1 else 'no clear difference')
        base=root.parent/'run-a03';sr=np.mean([read(base/f'E1-SR-s{s}/final_eval.json')['metrics']['ood_test']['accuracy'] for s in [0,1]])
        readings['R3']={a:dict(delta=means[a]-sr,label='learned routing matters' if means[a]-sr>=.1 else 'random better' if means[a]-sr<=-.1 else 'no clear difference',reference='pilot SR: unmatched layout/budget') for a in ['SO','DOT']}
        if not r0:flags.append(readings['R0'])
    else:flags.append('Missing arms/seeds; no complete mean-seed readings')
    for a in ['D','L','SO','DOT']:
        for s in [0,1]:
            if not any(r['arm']==a and r['seed']==s for r in rows):flags.append(f'E1b-{a}-s{s} missing')
    write(out/'readings.json',dict(mean_OOD=means,readings=readings,flags=flags))
    write(out/'replay_verification.json',dict(verified=True,runs=len(rows),routed_proofs=proof,Tier1_hashes_verified=True))
    # Preserve all first200 provenance and resume checkpoints in the inventory.
    for stage in sorted(root.glob('first200-*')):
        if stage.is_dir():
            for p in sorted(stage.rglob('*')):
                if p.is_file():artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),tier=1 if p.suffix in ['.json','.jsonl'] else 'local_training_state',locations=[str(p)],verified_utc=time.time(),off_node=False))
    write(out/'artifact_manifest.json',dict(artifacts=artifacts,flags=['weights/fullstates node-only; not off-node backup']))
    lines=['# DOT-pool v1.5 — E1b addendum',f'Spec SHA256: `{SPEC_SHA}`.',f'Completed endpoints: {len(rows)}/8. Common N={budget["epochs"]}; LR0.001; 8 steps; no stem/coda; fresh initialization; one GPU per run. First200epochs resume exact full optimizer/RNG/pending Taylor state.',
           'Kernel: FA4/grouped GEMM; eager. First200 measurement includes periodic dev evaluation. Stage transition and final evaluation overhead are separately included in wall time.',
           '## Flags',*['- '+str(x) for x in flags],'## Endpoint results','|Arm|Seed|ID-test|OOD-test|Trained|OOD atoms|OOD50|Last500 dev change|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["arm"]}|{r["seed"]}|{r["ID_test"]:.6f}|{r["OOD_test"]:.6f}|{r["trained_fact_accuracy"]:.6f}|{r["OOD_atomic_accuracy"]:.6f}|{r["epoch_OOD50"]}|{r["OOD_dev_change_last500"]:+.6f}|')
    lines+=['## Readings','```json',json.dumps(readings,indent=2),'```','R4 and R4b per seed and correct/wrong groups: e1b_reentry.csv. Pilot CPU-only same-relation update: e1_reentry_same_relation.csv. Bootstrap: 10000 draws, seed0, equal weight per eligible item; no-partner items excluded and counted.',
            '## Provenance and backups','budget.json, tests_summary.json and per-arm checks are retained at the run root. First200 outputs remain immutable. Full states include optimizer, RNG and pending Taylor labels; last two500epoch checkpoints plus final are retained. artifact_manifest.json lists SHA256/size/location/time. Tier1 bundles are published per completed run; off-node retrieval is verified separately. Native GPU handoff follows this report and requires its own supervisor proof. E2 is closed; no pilot model was retrained.',
            '## Original pilot report (unchanged; SR correction verified)']
    reference=Path.cwd()/'pilot_reference.md'
    if reference.exists():lines += [f'Original report SHA256 `{sha(reference)}`.',reference.read_text()]
    (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    for name in ['budget.json','tests_summary.json','run_index.json']:(out/name).write_text((root/name).read_text())
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}
    write(out/'manifest.json',dict(files=files))
    archive=root/'E1B_REPORT_SMALL.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(out.iterdir()):tar.add(p,arcname=p.name,recursive=False)
    write(root/'report_complete.json',dict(reported=True,runs=len(rows),all_eight_complete=len(rows)==8,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),ready_for_small_export=archive.stat().st_size<20*1024*1024,flags=flags))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();report(a.root)
