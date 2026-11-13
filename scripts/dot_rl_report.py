"""CPU report/replay of immutable DOT-RL evaluations, explicit science flags."""
import argparse,csv,json,tarfile,time
from pathlib import Path
import numpy as np
from efficient_llm.dot_pool.rl import write,sha,SPEC_SHA,ARMS

def dumpcsv(path,rows):
    with Path(path).open('w') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def reading(diff,small=False):
    return 'learned routing helps' if diff>=.1 else 'small' if small and diff>=.03 else 'no effect'
def comparison(diff):return 'beats' if diff>=.1 else 'worse' if diff<=-.1 else 'matches looping with composition rewards'
def readings(values):
    def mean(a):return float(np.mean(values[a]))
    def spread(a):return max(values[a])-min(values[a])
    out=[]
    def add(name,a,b,label,threshold):
        diff=mean(a)-mean(b);out.append(dict(reading=name,arm=a,reference=b,difference=diff,label=label,within_seed_noise=abs(diff)>=threshold and abs(diff)<2*max(spread(a),spread(b))))
    out.append(dict(reading='RL-R0',label='positive control works' if mean('L')>=.2 else 'inconclusive: composition did not emerge within the budget',mean=mean('L')))
    for a in ['J-RL','WS-RL']:add('RL-R1',a,'P-rand',reading(mean(a)-mean('P-rand'),True),.03)
    diff=mean('WS-RL')-mean('J-RL');add('RL-R2','WS-RL','J-RL','warm-up helps' if diff>=.1 else 'warm-up hurts' if diff<=-.1 else 'no clear difference',.1)
    best=max(['J-RL','WS-RL'],key=mean);add('RL-R3',best,'L',comparison(mean(best)-mean('L')),.1)
    add('RL-R4','FR-adaptive','P-rand',reading(mean('FR-adaptive')-mean('P-rand'),True),.03)
    add('RL-R5','FR-adaptive','FR-fixed','per-question routing matters' if mean('FR-adaptive')-mean('FR-fixed')>=.05 else 'one good path suffices',.05)
    diff=mean(best)-mean('FR-adaptive');add('RL-R6',best,'FR-adaptive','beats' if diff>=.1 else 'worse' if diff<=-.1 else 'matches selection-only',.1)
    return out

def report(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=False);index=json.loads((root/'run_index.json').read_text());flags=list(index['flags']);rows=[];paths=[];oracle=[];frcurves=[];values={};inventory=[];replay=[];curves={};configs=[];pathcomp=[]
    for cfg in index['runs']:
        run=Path(cfg['output']);arm=cfg['arm'];seed=cfg['seed'];fr=arm.startswith('FR');terminal=run/'complete.json'
        if not terminal.exists():flags.append(dict(run=run.name,flag='incomplete; excluded from readings'));continue
        done=json.loads(terminal.read_text());flags+=done.get('flags',[]);configs.append(dict(config=json.loads((run/'config.json').read_text()),completion=done,training=json.loads((run/'train_complete.json').read_text()) if (run/'train_complete.json').exists() else None))
        if fr and not done['T12']['passed']:flags.append(dict(run=run.name,flag='T12 FAILED'))
        curve=[]
        if (run/'curves.jsonl').exists():curve=[json.loads(l) for l in (run/'curves.jsonl').read_text().splitlines()]
        curves[run.name]=curve
        if fr:
            for line in (run/'metrics.jsonl').read_text().splitlines():
                r=json.loads(line);frcurves.append(dict(arm=arm,seed=seed,update=r['step'],rl_dev_accuracy=r['rl_dev']['accuracy'],rl_dev_reward=r['rl_dev']['reward']))
        for tag in (['selected','last'] if fr else ['final']):
            result=json.loads((run/(tag+'_eval.json')).read_text());arr=np.load(run/(tag+'_paths.npz'))
            for mode,metrics in result['modes'].items():
                for split in metrics:
                    key=mode+'_'+split+'_correct'
                    if key in arr:
                        assert abs(float(arr[key].mean())-metrics[split]['accuracy'])<1e-14
                        assert abs(float(arr[key].any(1).mean())-metrics[split]['oracle'])<1e-14
                        replay.append(dict(run=run.name,tag=tag,mode=mode,split=split,passed=True))
                keymode='random' if arm=='P-rand' else 'loop' if arm=='L' else 'greedy'
                cc=[(r['epoch'],r['modes'][mode]['ood_dev']['accuracy']) for r in curve if mode in r['modes']]
                first=next((ep for ep,v in cc if v>=.5),'never');change=''
                if cc:
                    end=cc[-1][0];past=[v for ep,v in cc if ep==end-500]
                    if past:change=cc[-1][1]-past[0]
                row=dict(arm=arm,seed=seed,checkpoint=tag,mode=mode,epochs=done.get('epochs',''),updates=done.get('selected_update',3000) if tag=='selected' else done.get('updates',''),id_test=metrics['id_test']['accuracy'],ood_test=metrics['ood_test']['accuracy'],rl_dev=metrics['rl_dev']['accuracy'],trained_facts=metrics['train']['accuracy'],ood_rl_atoms=metrics['ood_rl_atoms']['accuracy'],ood_test_atoms=metrics['ood_test_atoms']['accuracy'],first_ood_dev_50=first,ood_dev_last500_change=change)
                rows.append(row)
                if mode==keymode and tag in ['final','selected']:values.setdefault(arm,[]).append(row['ood_test'])
                if not fr and row['id_test']<.9:flags.append(dict(run=run.name,mode=mode,flag='did not fit ID'))
                for k in ['trained_facts','ood_rl_atoms','ood_test_atoms']:
                    if row[k]<.95:flags.append(dict(run=run.name,mode=mode,flag=k+' below95%; sanity not near100%'))
                if change!='' and change>0:flags.append(dict(run=run.name,mode=mode,flag='OOD-dev still rising over final500epochs',change=change))
                if arm=='P-rand' and mode=='random':
                    oracle.append(dict(seed=seed,random_mean=row['ood_test'],oracle=metrics['ood_test']['oracle'],headroom=metrics['ood_test']['oracle']-row['ood_test'],label='little headroom: hardly any path composes' if metrics['ood_test']['oracle']<.1 else 'headroom available'))
        for f in sorted(run.glob('paths-epoch*.json')):
            r=json.loads(f.read_text())
            for comparison_row in r.get('comparisons',[]):pathcomp.append(dict(arm=arm,seed=seed,epoch=str(r['epoch']),**comparison_row))
            for split,v in r['groups'].items():paths.append(dict(arm=arm,seed=seed,epoch=str(r['epoch']),split=split,items=v['items'],correct=v['correct'],loop_share=v['loop_share'],repeat_share=v['repeat_share'],top=json.dumps(v['top']),top_correct=json.dumps(v['top_correct']),top_wrong=json.dumps(v['top_wrong'])))
        manifest=run/'artifact_manifest.json'
        if manifest.exists():inventory+=json.loads(manifest.read_text())['artifacts']
    complete=all(len(values.get(a,[]))==2 for a in ARMS+['FR-adaptive','FR-fixed'])
    labels=readings(values) if complete else [dict(reading='all',label='INCOMPLETE; no final comparison')]
    write(out/'readings.json',dict(readings=labels,values=values));write(out/'replay.json',dict(passed=True,checks=replay,limits='accuracy aggregation replay only; no repeated model inference'))
    write(out/'configurations.json',dict(runs=configs));dumpcsv(out/'path_comparisons.csv',pathcomp)
    write(out/'learning_curves.json',dict(runs=curves));write(out/'artifact_manifest.json',dict(artifacts=inventory));write(out/'flags.json',dict(flags=flags))
    dumpcsv(out/'summary.csv',rows);dumpcsv(out/'paths.csv',paths);dumpcsv(out/'oracle.csv',oracle);dumpcsv(out/'fr_curves.csv',frcurves)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6))
    for name,cc in curves.items():
        for mode in sorted({mode for r in cc for mode in r['modes']}):
            points=[(r['epoch'],r['modes'][mode]['ood_dev']['accuracy']) for r in cc if mode in r['modes']]
            if points:ax.plot(*zip(*points),label=name+' '+mode)
    ax.set(xlabel='Epoch',ylabel='OOD-dev accuracy');ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/'curves.png');plt.close(fig)
    for name in ['budget.json','tests_summary.json','run_index.json']:
        (out/name).write_bytes((root/name).read_bytes())
    budget=json.loads((root/'budget.json').read_text());data=json.loads((root.parent/'data/manifest.json').read_text());write(out/'data_manifest.json',data)
    text=['# DOT-RL v1.2 report',f'Spec SHA256: `{SPEC_SHA}`.',f'Complete: {complete}. N={budget["epochs"]}, planned WS switch={budget["switch_epoch"]}. Speed reference: {budget.get("reference")}.', 'RL arms receive OOD-RL composition rewards; controls do not. This is not an equal-data comparison.','Environment, actual WS switch epochs and phase timings: configurations.json and run_index.json.','## Results','|Arm|Seed|Checkpoint|Mode|ID-test|OOD-test|RL-dev|','|---|---:|---|---|---:|---:|---:|']
    for r in rows:text.append('|'+ '|'.join(str(r[k]) for k in ['arm','seed','checkpoint','mode','id_test','ood_test','rl_dev'])+'|')
    text+=['## Readings',*[json.dumps(r) for r in labels],'## Flags',*[json.dumps(f) for f in flags], '## Reproduction and artifacts', 'The JSON/CSV files include split counts, timings, environments/configs, sanity accuracies, curves, selected/last FR, paths and oracle headroom. Artifact manifest records exact node locations and SHA256. Router weights are Tier1; model/fullstate remains node-only. No E1b/E2/pilot reruns.', 'Phase timings and failures: run_index.json. Numerical replay: replay.json.']
    (out/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    write(out/'manifest.json',dict(files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(out.iterdir()) if f.is_file()}))
    archive=root/'REPORT_SMALL.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for f in sorted(out.iterdir()):tar.add(f,arcname=f.name)
    write(root/'report_complete.json',dict(reported=complete,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),ready_for_small_export=archive.stat().st_size<20*1024**2))
    return complete

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();report(a.root)
