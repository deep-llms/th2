"""Depth short-heal quantities calculated from saved final sequence arrays."""
import json
from pathlib import Path
import numpy as np
from efficient_llm.a2_depth.common import read,write,SPEC_SHA
from efficient_llm.a2_stage1.protocol import decision as classify
from efficient_llm.a2_depth.readout_audit import array_audit, collect_checks


def compute(cfg,root):
    root=Path(root);table={};flags=[];runs=[];tests=collect_checks(root);arrays_complete=True
    for ds in ('dclm','fwe'):
        table[ds]={}
        for block in ('W1','W9'):
            values={}
            for arm in ('P300','P1T','X72'):
                name=f'D-{ds}-{arm}-{block}-0.2B-lr1e-3'
                try:
                    r=read(root/'heals'/name/'result.json')
                    flags.extend(dict(run=name,detail=flag) for flag in r.get('flags',[]))
                    arrays=[]
                    for key,row in [('final',r['scores']['final']),('parent',r['baseline'])]:
                        arr,usable,full,audits=array_audit(dict(row,spec_sha256=r.get('spec_sha256')),cfg,name+'/'+key)
                        tests.extend(audits);arrays_complete=arrays_complete and full
                        if not usable:raise ValueError('Nonfinite or malformed '+key+' array')
                        arrays.append(arr)
                    if len(arrays[0])!=len(arrays[1]):raise ValueError('Unpaired parent/post arrays')
                    final,parent=map(lambda arr:float(arr.mean()),arrays)
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
    dt4_coverage=False
    if dt4_result is not None:
        try:
            _,_,dt4_coverage,audits=array_audit(dict(dt4_result['scores']['final'],spec_sha256=dt4_result.get('spec_sha256')),cfg,'DT4/final')
            tests.extend(audits);flags.extend(t for t in audits if not t['passed'])
        except Exception as error:flags.append(dict(run='DT4',error=repr(error)))
    return dict(part='B',coverage_complete=arrays_complete and len(runs)==12 and dt4_coverage,
        datasets=table,predictions=predictions,clauses=clauses,DT4=dt4_result,flags=flags,
        deviations=cfg.get('deviations',[]),remaining=['publish A+B report','DT6','Part C','C downstream','final replay/report/native handoff']),tests,runs


def finalize(cfg):
    root=Path(cfg['results_root']);decision,tests,runs=compute(cfg,root)
    part_a_text=None
    try:
        part_a_text=Path(cfg['part_a_report_local']).read_text()
    except Exception as error:
        decision['flags'].append(dict(test='part_a_report_consolidation',passed=False,error=repr(error)))
    decision['part_a_report_included']=part_a_text is not None
    write(root/'decision_depth_partb.json',decision);write(root/'tests.json',dict(checks=tests))
    (root/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in runs))
    lines=['# A2 Depth — Parts A and B', '',f'Spec SHA-256: `{SPEC_SHA}`.', '',
        'Part A source report: '+str(cfg.get('part_a_report','unavailable; flagged'))+'.', '',
        '| Dataset | Block | E | E / Δ(P300) | A_init | A_init / Δ(P300) | E label |',
        '|---|---|---:|---:|---:|---:|---|']
    for ds,t in decision['datasets'].items():
        for block,r in t.items():
            lines.append('| '+ ' | '.join([ds,block,*[str(r.get(k)) for k in ('E','E_relative','A_init','A_init_relative')],str(r.get('E_label'))])+' |')
    lines+=['','Predictions: `'+json.dumps(decision['predictions'])+'`. Every clause: `'+json.dumps(decision['clauses'])+'`.', '',
        'Scientific coverage complete: '+str(decision['coverage_complete'])+'. Flags: '+str(len(decision['flags']))+'.', '',
        'DT4 and scientific flags remain recorded; they do not cancel Part C. Ledger: ledger.json (report-only).', '',
        'Remaining: publish this A+B result, DT6, three 12.8B fits, downstream, independent replay and final native handoff.']
    if part_a_text is not None:
        lines+=['','## Part A: completed depth table and A1–A4','',
            'The Part A numerical results below are retained from its earlier report; its then-pending-stage footer is omitted here.','',
            '\n'.join(line for line in part_a_text.splitlines() if not line.startswith(('# A2 Depth','Remaining:')))]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return decision
