"""Node-side F11 replay and paired readout; only compact deliverables leave node."""
from pathlib import Path
import csv
import json
import numpy as np
from eval.a2_family_f11 import read,write,SPEC_SHA,choose_temperature,document_bootstrap,interval,lm_reading,calibration_reading
from eval.a2_family_f10_report import pairs,contrast_readout,replay_new
from scripts.a2_family_f9_run import locked
from efficient_llm.diagnostic_data import sha256

def verify_calibration(folder):
    r=read(folder/'result.json');tune=read(folder/'tuning.json');final=read(folder/'final.json')
    for v in (tune,final):
        c=np.asarray(v['token_counts']);n=len(c)
        if n!=v['sequences'] or c.sum()!=v['tokens'] or np.any(c<=0):raise ValueError('Calibration coverage mismatch')
        for t,arr in v['per_sequence_nll'].items():
            a=np.asarray(arr)
            if a.shape!=c.shape or not np.isfinite(a).all() or abs(a@c/c.sum()-v['nll'][t])>1e-10:raise ValueError('Calibration NLL replay mismatch')
        ent=np.asarray(v['per_sequence_entropy'])
        if ent.shape!=c.shape or not np.isfinite(ent).all() or abs(ent@c/c.sum()-v['mean_entropy'])>1e-10:raise ValueError('Entropy replay mismatch')
    if choose_temperature(tune['nll'])!=r['selection']:raise ValueError('Temperature selection replay mismatch')
    actual=final['nll']['100']-r['FT17']['expected']
    if abs(actual-r['FT17']['difference'])>1e-10 or (abs(actual)<=1e-4)!=r['FT17']['passed']:raise ValueError('FT17 replay mismatch')
    return r,final

def calibrations(cfg,root):
    entries=cfg['entries'];records={};missing=[];model_rows=[]
    for e in entries:
        folder=root/'calibration'/e['model_id']
        if not (folder/'result.json').is_file():missing.append(e['model_id']);continue
        r,v=verify_calibration(folder);records[e['model_id']]=(r,v)
        model_rows.append(dict(model=e['model_id'],dataset=e['dataset'],arm=e['arm'],role=e['role'],budget=e.get('budget'),
            T_star=r['selection']['T_star'],NLL=v['nll']['100'],NLL_cal=v['nll'][r['selection']['T_key']],
            entropy=v['mean_entropy'],FT17=r['FT17']['passed'],flags=r['flags'],extension=r['selection']['extension']))
    ids=list(records);counts=None;matrix=[]
    for name in ids:
        r,v=records[name];c=v['token_counts']
        if counts is not None and c!=counts:raise ValueError('Calibration paired sequence counts mismatch')
        counts=c;matrix.extend([v['per_sequence_nll']['100'],v['per_sequence_nll'][r['selection']['T_key']]])
    rows=[]
    if matrix:
        point,draw=document_bootstrap(np.asarray(matrix).T,counts)
        for row in pairs(entries):
            names=row['models']
            if not set(names)<=set(ids):continue
            weights=np.asarray([1,-1,-1,1]*(len(names)//4))*4/len(names)
            col=[2*ids.index(m) for m in names];E=point[col]@weights;Ec=point[np.asarray(col)+1]@weights
            ratio,reading=calibration_reading(E,Ec)
            registered=row['pair']==['P300','P1T'] and ((row['dataset']=='pooled' and row['budget_B'] in (.2,.8,3.2)) or (row['dataset']=='dclm' and row['budget_B']==12.8))
            rows.append(dict(row,E=interval(E,draw[:,col]@weights),E_cal=interval(Ec,draw[:,np.asarray(col)+1]@weights),r=ratio,
                reading=reading if registered else None,registered_reading=registered,
                flags=sorted({f for m in names for f in records[m][0]['flags']})))
    byid={r['model']:r for r in model_rows}
    for e in entries:
        if e['model_id'] not in byid:continue
        r=byid[e['model_id']];p=byid.get(e.get('parent_id'))
        r['delta_post']=r['NLL']-p['NLL'] if p else None
        r['delta_post_cal']=r['NLL_cal']-p['NLL_cal'] if p else None
    return dict(models=model_rows,contrasts=rows,missing=missing,replayed=len(records),bootstrap='joint paired documents, token weighted, 10000 resamples seed0; own tuning T for every parent/student')

def lm_result(root):
    results={}
    for p in (root/'heals').glob('*/result.json'):
        r=read(p);results[r['arm']]=r
        for path in r['raw_order_files']:
            if sha256(path['path'])!=path['sha256']:raise ValueError('FT18 trace changed')
        traces=[]
        for path in r['raw_order_files']:
            traces.append([json.loads(line)['sample_ids'] for line in Path(path['path']).read_text().splitlines()])
        from efficient_llm.a2.train import audit_order
        audit_order(traces,list(range(3051*64)),8,4)
        for s in [r['baseline'],r['pre']['final'],r['scores']['final']]:
            a=np.asarray(s['per_sequence_nll'])
            if len(a)!=s['sequences'] or abs(a.mean()-s['loss'])>1e-10:raise ValueError('LM per-sequence replay mismatch')
    rows=[dict(arm=a,parent=r['baseline']['loss'],pre=r['pre']['final']['loss'],post=r['scores']['final']['loss'],
               delta_pre=r['pre']['final']['loss']-r['baseline']['loss'],delta_post=r['delta_post'],flags=r['flags']) for a,r in results.items()]
    if set(results)!= {'P300','P1T'}:return dict(rows=rows,complete=False,missing=sorted({'P300','P1T'}-set(results)))
    a,b=results['P300'],results['P1T']
    mat=np.asarray([a['baseline']['per_sequence_nll'],b['baseline']['per_sequence_nll'],a['scores']['final']['per_sequence_nll'],b['scores']['final']['per_sequence_nll']]).T
    n=len(mat);counts=[a['scores']['final']['tokens']//n]*n;p,d=document_bootstrap(mat,counts);w=np.array([1,-1,-1,1]);E=float(p@w)
    return dict(rows=rows,complete=True,E_LM=interval(E,d@w),E_KD_08=.020786423,reading=lm_reading(E),
                learning_rate_qualification='1e-3 chosen for KD, not retuned for LM',teacher=None)

def downstream(cfg,root):
    parents=[e for e in cfg['entries'] if e['role']=='parent' and e['dataset']=='dclm' and e['arm'] in ('P300','P1T')]
    students=read(root/'lm_entries.json')['entries'];entries=parents+students;records={};continuous={};scores={}
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
                complete=len(students)==2 and len(scores)==4 and all(len(r)==11 for r in records.values()) and len(records)==4)

def report(cfg,root):
    root=Path(root);cal=calibrations(cfg,root);lm=lm_result(root);down=downstream(cfg,root)
    write(root/'calibration.json',cal);write(root/'LM_comparison.json',lm);write(root/'downstream.json',down)
    previous=cfg['previous_report']
    if sha256(previous['path'])!=previous['sha256']:raise ValueError('Prior report changed')
    text=Path(previous['path']).read_text()+'\n\n# Family v1.13 — F11\n\n'
    text+=f'Spec SHA256: {SPEC_SHA}. F10 retained unchanged.\n\n'
    text+='| Pair | E_LM (95% CI) | E_KD at 0.8B | Reading | Downstream E (pp, 95% CI) |\n|---|---|---|---|---|\n'
    contrasts=down['accuracy']['contrasts'];edown=contrasts[0]['E'] if contrasts else {}
    text+='| DCLM P300→P1T | '+json.dumps(lm.get('E_LM'))+' | 0.020786423 | '+str(lm.get('reading','unavailable'))+' | '+json.dumps(edown)+' |\n'
    text+='\nLM uses next-token CE, no teacher, original Stage1 data/seed/optimizer/LR schedule. LR 1e-3 was selected for KD and was not retuned for LM.\n\n'
    text+='| Corpus | Pair | Budget B (0=one-shot) | E T=1 (CI) | E_cal (CI) | r | Reading |\n|---|---|---:|---|---|---:|---|\n'
    for r in cal['contrasts']:text+='| '+' | '.join(map(str,[r['dataset'],'→'.join(r['pair']),r['budget_B'],r['E'],r['E_cal'],r['r'],r['reading']]))+' |\n'
    text+='\n| Model | T* | NLL T=1 | NLL T* | Δpost T=1 | Δpost T* | Entropy T=1 | FT17 | Flags |\n|---|---:|---:|---:|---:|---:|---:|---|---|\n'
    for r in cal['models']:text+='| '+' | '.join(map(str,[r[k] for k in ('model','T_star','NLL','NLL_cal','delta_post','delta_post_cal','entropy','FT17','flags')]))+' |\n'
    text+='\nThe 31-value grid is primary. Five extra reductions on either side share the same logits; only the triggered side is admitted after a base-grid edge optimum. No second decoder forward. An optimum still on the extended edge is flagged. Temperatures are selected on tuning only; parents and students each use their own T*. Paired token-weighted document bootstrap, 10,000 draws seed0; evaluation noise only.\n\n'
    text+='Full LM rows and downstream accuracy/continuous comparisons:\n\n```json\n'+json.dumps(dict(LM=lm,downstream=down),indent=2)+'\n```\n'
    complete=cal['replayed']==42 and lm['complete'] and down['complete']
    text+='\nCoverage: '+json.dumps(dict(calibration=cal['replayed'],missing=cal['missing'],LM=lm['complete'],downstream=down['complete']))+'\n'
    text+='\nScientific flags remain on their rows; prior F0–F10 flags retained above. No token-group split (removed in v1.13).\n'
    (root/'REPORT_FULL_v1.13.md').write_text(text)
    with (root/'F11_MODELS.csv').open('w') as f:
        fields=['spec_sha256','model','dataset','arm','role','budget','T_star','NLL','NLL_cal','delta_post','delta_post_cal','entropy','FT17','extension','flags']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(dict(r,spec_sha256=SPEC_SHA) for r in cal['models'])
    write(root/'report_complete.json',dict(complete=complete,calibration_replayed=cal['replayed'],LM_replayed=lm['complete'],downstream_replayed=down['complete'],previous_report=previous))
