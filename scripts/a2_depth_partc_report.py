"""CPU readout and per-run notes from Part C saved sequence arrays."""
import json
from pathlib import Path
from efficient_llm.a2_depth.common import read,write,SPEC_SHA
from efficient_llm.a2_depth.readout_audit import array_audit,collect_checks
from efficient_llm.a2_stage1.protocol import decision as classify


def compute(cfg,root):
    root=Path(root);tests=collect_checks(root);runs=[];values={};flags=[];complete=True
    if (root/'runtime_checks.json').exists():tests.extend(read(root/'runtime_checks.json')['checks'])
    for arm in ('P300','P1T','X72'):
        name=f'ST1-dclm-{arm}-S1-12.8B-lr1e-3'
        try:
            row=read(root/'heals'/name/'result.json');arrays=[]
            for label,v in [('final',row['scores']['final']),('parent',row['baseline'])]:
                arr,usable,full,audit=array_audit(dict(v,spec_sha256=row.get('spec_sha256')),cfg,name+'/'+label)
                tests.extend(audit);complete=complete and full
                if not usable:raise ValueError('Unusable '+label+' array')
                arrays.append(arr)
            if len(arrays[0])!=len(arrays[1]):raise ValueError('Unpaired parent/final arrays')
            val,parent=(float(a.mean()) for a in arrays)
            values[arm]=dict(val=val,parent_val=parent,delta_post=val-parent,checkpoint=row['checkpoint'])
            runs.append(dict(run_id=name,arm=arm,**values[arm]))
            flags.extend(dict(run_id=name,detail=f) for f in row.get('flags',[]))
            tests.append(dict(test='saved_delta:'+name,passed=abs(val-parent-row['delta_post'])<=1e-12))
        except Exception as error:
            complete=False;flags.append(dict(run_id=name,error=repr(error)))
    E=values['P1T']['delta_post']-values['P300']['delta_post'] if {'P300','P1T'}<=values.keys() else None
    A=values['X72']['val']-values['P300']['val'] if {'P300','X72'}<=values.keys() else None
    history=cfg['stage1_dclm_reference'];ref=history['3.2B']
    def ratio(v,key):return v/ref[key] if v is not None and ref[key] else None
    for path in sorted((root/'heals').glob('*/tests-rank*.json')):
        tests.extend(dict(t,run_id=path.parent.name,rank_file=path.name) for t in read(path)['checks'])
    dt6=root/'DT6.json'
    if dt6.exists():tests.extend(read(dt6)['checks'])
    else:tests.append(dict(test='DT6_evidence',passed=False,detail='Not available'))
    flags.extend(t for t in tests if not t['passed'])
    return dict(part='C',coverage_complete=complete and len(runs)==3,parents=values,
        E=E,A_init=A,E_ratio_3_2B=ratio(E,'E'),A_init_ratio_3_2B=ratio(A,'A_init'),
        E_label=classify(E,'length'),A_init_label=classify(A,'state'),stage1_budgets=history,
        predictions=dict(C1=E>=.008 if E is not None else None,C2=A<.01 if A is not None else None),
        flags=flags,deviations=cfg.get('deviations',[])),tests,runs


def finalize(cfg, *, note_arm=None):
    root=Path(cfg['results_root']);decision,tests,runs=compute(cfg,root)
    write(root/'decision_depth_partc.json',decision);write(root/'tests_partc.json',dict(checks=tests))
    (root/'runs_partc.jsonl').write_text(''.join(json.dumps(dict(r,spec_sha256=SPEC_SHA))+'\n' for r in runs))
    lines=['# A2 Depth — Part C saved-array readout','',f'Spec SHA-256: `{SPEC_SHA}`.','',
           '| Arm | Final NLL | Δ_post |','|---|---:|---:|']
    for arm,r in decision['parents'].items():lines.append(f"| {arm} | {r['val']} | {r['delta_post']} |")
    lines+=['','| Budget | E | A_init |','|---|---:|---:|']
    for budget,r in decision['stage1_budgets'].items():lines.append(f"| {budget} | {r['E']} | {r['A_init']} |")
    lines.append(f"| 12.8B | {decision['E']} | {decision['A_init']} |")
    lines+=['',json.dumps({k:decision[k] for k in ('E_ratio_3_2B','A_init_ratio_3_2B','E_label','A_init_label','predictions')},ensure_ascii=False),
            '',f"Coverage: {decision['coverage_complete']}; flags: {len(decision['flags'])}. Missing results remain unavailable.",
            '', 'Remaining: finish registered arms, downstream/paired intervals, independent replay and consolidated final report/native handoff.']
    (root/'REPORT_PART_C.md').write_text('\n'.join(lines)+'\n')
    if note_arm:
        note=dict(arm=note_arm,result=decision['parents'].get(note_arm),E=decision['E'],A_init=decision['A_init'],flags=decision['flags'])
        write(root/'notes'/f'{note_arm}.json',note)
        print('DEPTH_PART_C_RUN_NOTE',json.dumps(dict(note,spec_sha256=SPEC_SHA)),flush=True)
    return decision
