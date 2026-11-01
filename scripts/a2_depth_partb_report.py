"""Depth short-heal quantities calculated from saved final sequence arrays."""
import json
from pathlib import Path
import numpy as np
from efficient_llm.a2_depth.common import read,write,SPEC_SHA
from efficient_llm.a2_stage1.protocol import decision as classify


def compute(cfg,root):
    root=Path(root);table={};flags=[];runs=[];tests=[]
    for ds in ('dclm','fwe'):
        table[ds]={}
        for block in ('W1','W9'):
            values={}
            for arm in ('P300','P1T','X72'):
                name=f'D-{ds}-{arm}-{block}-0.2B-lr1e-3'
                try:
                    r=read(root/'heals'/name/'result.json')
                    final=float(np.mean(r['scores']['final']['per_sequence_nll']))
                    parent=float(np.mean(r['baseline']['per_sequence_nll']))
                    values[arm]=dict(val=final,parent_val=parent,delta=final-parent,checkpoint=r['checkpoint'])
                    runs.append(dict(run_id=name,dataset=ds,block=block,arm=arm,**values[arm],spec_sha256=SPEC_SHA))
                    tests.append(dict(test='saved_mean:'+name,passed=abs(final-r['scores']['final']['loss'])<=1e-12 and abs(final-parent-r['delta_post'])<=1e-12,spec_sha256=SPEC_SHA))
                except Exception as error:
                    flags.append(dict(run=name,error=repr(error)))
            if set(values)=={'P300','P1T','X72'}:
                den=values['P300']['delta'];E=values['P1T']['delta']-den;A=values['X72']['val']-values['P300']['val']
                table[ds][block]=dict(parents=values,E=E,A_init=A,E_relative=E/den if den else None,
                    A_init_relative=A/den if den else None,E_label=classify(E,'length'))
        table[ds]['S1']=cfg['short_S1_reference'][ds]
    clauses={}
    for ds in ('dclm','fwe'):
        t=table[ds];ref=t['S1']
        clauses[ds]={}
        for name,block,key in [('B1','W1','E_relative'),('B2','W9','A_init_relative')]:
            v=t.get(block,{}).get(key);clauses[ds][name]=bool(v>ref[key]) if v is not None else None
    predictions={name:all(clauses[ds][name] for ds in table) if all(clauses[ds][name] is not None for ds in table) else None for name in ('B1','B2')}
    for path in sorted((root/'heals').glob('*/tests-rank*.json')):
        tests.extend(dict(t,run=path.parent.name,rank_file=path.name) for t in read(path)['checks'])
    flags.extend(t for t in tests if not t['passed'])
    dt4=root/'heals/ST1-dclm-P300-S1-0.2B-lr1e-3/result.json'
    dt4_result=read(dt4) if dt4.exists() else None
    return dict(part='B',coverage_complete=len(runs)==12 and dt4_result is not None,
        datasets=table,predictions=predictions,clauses=clauses,DT4=dt4_result,flags=flags,
        deviations=cfg.get('deviations',[]),remaining=['publish A+B report','DT6','Part C','C downstream','final replay/report/native handoff']),tests,runs


def finalize(cfg):
    root=Path(cfg['results_root']);decision,tests,runs=compute(cfg,root)
    write(root/'decision_depth_partb.json',decision);write(root/'tests.json',dict(checks=tests))
    (root/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in runs))
    lines=['# A2 Depth — Parts A and B', '',f'Spec SHA-256: `{SPEC_SHA}`.', '',
        'Part A report/decision: '+str(cfg.get('part_a_report','pending consolidation'))+'.', '',
        '| Dataset | Block | E | E / Δ(P300) | A_init | A_init / Δ(P300) | E label |',
        '|---|---|---:|---:|---:|---:|---|']
    for ds,t in decision['datasets'].items():
        for block,r in t.items():
            lines.append('| '+ ' | '.join([ds,block,*[str(r.get(k)) for k in ('E','E_relative','A_init','A_init_relative')],str(r.get('E_label'))])+' |')
    lines+=['','Predictions: `'+json.dumps(decision['predictions'])+'`. Every clause and saved-array calculation is in decision_depth_partb.json.', '',
        'DT4 and scientific flags remain recorded; they do not cancel Part C. Ledger: ledger.json (report-only).', '',
        'Remaining: publish this A+B result, DT6, three 12.8B fits, downstream, independent replay and final native handoff.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return decision
