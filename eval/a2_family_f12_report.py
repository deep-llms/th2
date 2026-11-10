"""On-node raw trace/metric replay and paired width-pruning contrasts."""
import csv
import json
from pathlib import Path
import numpy as np
from eval.a2_family_f12 import read,write,descriptors,document_bootstrap,interval,lm_reading,budget_reading
from scripts.a2_family_f9_run import locked
from efficient_llm.diagnostic_data import sha256

def inventory_report(cfg,root):
    root=Path(root);inventory=locked(cfg['inventory']);prior=cfg['previous_report']
    if sha256(prior['path'])!=prior['sha256']:raise ValueError('F11 report changed')
    text=Path(prior['path']).read_text()+'\n\n# Family v1.16 — F12\n\n## F12a: length-pair inventory\n\n'
    text+=inventory['conclusion']+'\n\nNo weights downloaded or model runs for this inventory. Main revisions are in inventory.json; all branch metadata is retained in its referenced compressed metadata archive.\n\n'
    text+='| Repository | Main revision | Size B | Dataset | Tokens B | Intermediate checkpoints | Schedule evidence |\n|---|---|---:|---|---:|---|---|\n'
    for r in inventory['models']:text+='| '+' | '.join(str(r.get(k)) for k in ('repo','revision','size_B','dataset','tokens_B','intermediate_count','schedule'))+' |\n'
    text+='\nSources: '+', '.join('['+v+']('+v+')' for v in inventory['sources'])+'\n'
    write(root/'inventory.json',inventory);(root/'REPORT_FULL_v1.16.md').write_text(text);(root/'REPORT.md').write_text(text)
    return text

def width_readout(cfg,root):
    from efficient_llm.a2.train import audit_order
    from efficient_llm.a2.protocol import lr_for_update
    root=Path(root);records={};rows=[];columns=[];counts=None
    for item in descriptors():
        path=root/'heals'/item['run_id']/'result.json'
        if not path.exists():continue
        r=read(path);traces=[]
        if r['registered_run']!=item:raise ValueError('Unexpected width recipe')
        for v in r['raw_order_files']:
            if sha256(v['path'])!=v['sha256']:raise ValueError('Changed raw sample trace')
            traces.append([json.loads(line)['sample_ids'] for line in Path(v['path']).read_text().splitlines()])
        audit_order(traces,list(range(item['steps']*64)),8,4)
        if not np.allclose(r['lr_updates'],[lr_for_update(i,item['steps'],.001) for i in range(1,item['steps']+1)],rtol=1e-12,atol=0):raise ValueError('LR replay mismatch')
        values=[r['baseline'],r['pre']['final'],r['scores']['final']]
        for v in values:
            a=np.asarray(v['per_sequence_nll']);c=np.full(len(a),v['tokens']//len(a))
            if c.sum()!=v['tokens'] or len(a)!=v['sequences'] or not np.isfinite(a).all() or abs(a@c/c.sum()-v['loss'])>1e-10:raise ValueError('Per-sequence replay mismatch')
            if counts is not None and not np.array_equal(counts,c):raise ValueError('Unpaired evaluation sequences')
            counts=c;columns.append(a)
        records[item['run_id']]=len(records)
        rows.append(dict(run=item['run_id'],arm=item['arm'],budget_B=float(item['budget'][:-1]),parent=values[0]['loss'],pre=values[1]['loss'],post=values[2]['loss'],delta_pre=values[1]['loss']-values[0]['loss'],delta_post=values[2]['loss']-values[0]['loss'],minus_13B=values[2]['loss']-cfg['reference_13']['final_nll'],flags=r['flags']))
    contrasts=[];draws={};points={}
    if columns:
        p,d=document_bootstrap(np.asarray(columns).T,counts)
        for b,depth in [(.2,.0211),(.8,.0208),(3.2,.0217)]:
            names=[f'W-dclm-{arm}-FFN5440-{b:.1f}B' for arm in ('P300','P1T')]
            if not all(n in records for n in names):continue
            a,z=[3*records[n] for n in names];ew=np.zeros(len(p));ew[[a,z,a+2,z+2]]=[1,-1,-1,1];prew=np.zeros(len(p));prew[[a,z,a+1,z+1]]=[1,-1,-1,1]
            E=float(p@ew);ed=d@ew;G=p[a]-p[z];gd=d[:,a]-d[:,z];draws[b]=ed;points[b]=E
            contrasts.append(dict(budget_B=b,E=interval(E,ed),E_pre=interval(p@prew,d@prew),G=interval(G,gd),E_over_G=interval(E/G,ed/gd) if abs(G)>1e-15 and np.all(np.abs(gd)>1e-15) else None,depth_E=depth,FE1=lm_reading(E) if b==3.2 else None))
    trend=None;reading=None
    if len(points)==3:
        x=np.log([.2,.8,3.2]);w=(x-x.mean())/sum((x-x.mean())**2)
        trend=interval(sum(c*points[b] for c,b in zip(w,[.2,.8,3.2])),sum(c*draws[b] for c,b in zip(w,[.2,.8,3.2])))
        reading=budget_reading(points[.2],points[3.2])
    return dict(rows=rows,contrasts=contrasts,slope_per_log_budget=trend,FE2=reading,complete=len(rows)==6,missing=[v['run_id'] for v in descriptors() if v['run_id'] not in records])

def report(cfg,root):
    root=Path(root);width=width_readout(cfg,root);down=downstream(cfg,root)
    write(root/'width_readout.json',width);write(root/'downstream.json',down)
    text=inventory_report(cfg,root)+'\n\n## F12b: FFN width pruning\n\nLR 1e-3 was selected for depth pruning/KD and was not retuned for width. Each budget starts fresh from its own parent. Paired bootstrap: 10,000 draws, seed 0, using the original packed-sequence units and token weighting. These are not independent document units; this inherited qualification remains. Intervals cover evaluation noise only.\n\n'
    text+='| Budget B | Width E (CI) | Depth E | E_pre (CI) | G (CI) | E/G (CI) | FE1 |\n|---:|---|---:|---|---|---|---|\n'
    for r in width['contrasts']:text+='| '+' | '.join(str(r[k]) for k in ('budget_B','E','depth_E','E_pre','G','E_over_G','FE1'))+' |\n'
    text+='\nFE2: '+str(width['FE2'])+'; slope per log budget: '+str(width['slope_per_log_budget'])+'\n\n'
    text+='| Run | Parent | Δ_pre | Δ_post | Student NLL | NLL minus F0 1.3B | Flags |\n|---|---:|---:|---:|---:|---:|---|\n'
    for r in width['rows']:text+='| '+' | '.join(str(r[k]) for k in ('run','parent','delta_pre','delta_post','post','minus_13B','flags'))+' |\n'
    text+='\nDownstream accuracy and continuous metrics, with joint item bootstrap and budget slopes:\n\n```json\n'+json.dumps(down,indent=2)+'\n```\n'
    masks={a:read(root/'masks'/(a+'.json')) for a in ('P300','P1T') if (root/'masks'/(a+'.json')).exists()}
    mask_flags={a:v['flags'] for a,v in masks.items()};text+='\nFT20 flags: '+json.dumps(mask_flags)+'. Prior flags are retained; missing runs: '+json.dumps(width['missing'])+'\n'
    (root/'REPORT_FULL_v1.16.md').write_text(text);(root/'REPORT.md').write_text(text)
    with (root/'F12_MODELS.csv').open('w') as f:
        fields=['spec_sha256','run','arm','budget_B','parent','pre','post','delta_pre','delta_post','minus_13B','flags'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        from eval.a2_family_f12 import SPEC_SHA
        w.writerows(dict(r,spec_sha256=SPEC_SHA) for r in width['rows'])
    write(root/'report_complete.json',dict(complete=width['complete'] and down['complete'] and len(masks)==2,width_replayed=len(width['rows']),downstream_replayed=down['complete'],mask_flags=mask_flags))

from eval.a2_family_f10_report import contrast_readout,replay_new
def downstream(cfg,root):
    parents=[e for e in cfg['entries'] if e['role']=='parent' and e['dataset']=='dclm' and e['arm'] in ('P300','P1T')]
    students=read(root/'width_entries.json')['entries'];entries=parents+students;records={};continuous={};scores={}
    f10=Path(cfg['f10']['output']);old=Path(cfg['f10']['f9_data_root']);prepared=locked(cfg['f10']['prepared'])
    from eval.a2_family_f9_v110 import export_items
    oldmanifest=locked(cfg['f10']['f9_manifest'])['files']
    for e in entries:
        m=e['model_id'];parent=e['role']=='parent'
        p=old/'per_item'/(m+'.json') if parent else root/'nine'/m/'per_item.json'
        if not p.is_file():continue
        if parent:
            if sha256(p)!=oldmanifest['per_item/'+m+'.json']['sha256']:raise ValueError('Parent F9 changed')
            exported=read(p)
        else:
            folder=root/'nine'/m;exported=read(p)
            replayed,_=export_items(read(folder/'replay.json'),locked(cfg['f10']['benchmarks']['gpt_neox']['prepared']),read(folder/'scores.json'))
            if any(replayed[k]!=exported[k] for k in replayed):raise ValueError('Nine-task export replay mismatch')
        records[m]={t:dict(identity=exported['continuous_records'][t]['identity'],values=[r['correctness'] for r in rows]) for t,rows in exported['tasks'].items()}
        continuous[m]=exported['continuous_records']
        folder=(f10 if parent else root)/'new'/m
        if not (folder/'generation_items.json').is_file():continue
        if parent:
            manifest=read(f10/'backup/verified.json')['files']
            for name in ('likelihood_items.json','generation_items.json','scores.json'):
                rel=str((folder/name).relative_to(f10))
                if sha256(folder/name)!=manifest[rel]['sha256']:raise ValueError('F10 parent artifact changed: '+rel)
        lik=read(folder/'likelihood_items.json');gen=read(folder/'generation_items.json')
        replay_new(prepared,read(folder/'scores.json'),lik,gen)
        records[m]['mmlu']=dict(identity=lik['task_identity']['mmlu'],values=[r['correctness'] for r in lik['tasks']['mmlu']])
        records[m]['triviaqa']=dict(identity=gen['task_identity'],values=[r['exact_match'] for r in gen['items']])
        continuous[m]['mmlu']=dict(identity=lik['task_identity']['mmlu'],values=[r['p_correct'] for r in lik['tasks']['mmlu']])
        continuous[m]['triviaqa']=dict(identity=gen['task_identity'],values=[r['conf'] for r in lik['tasks']['triviaqa']])
        scores[m]=dict(TriviaQA_EM=gen['exact_match'],TriviaQA_F1=gen['token_f1'],MMLU_acc_norm=float(np.mean(records[m]['mmlu']['values'])))
    accuracy,_=contrast_readout(entries,records,scale=100.);secondary,_=contrast_readout(entries,continuous)
    return dict(accuracy=accuracy,continuous=secondary,model_scores=scores,
                complete=len(students)==6 and len(scores)==8 and all(len(r)==11 for r in records.values()) and len(records)==8)
