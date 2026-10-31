"""CPU readout directly from saved per-document/per-sequence results."""
import argparse
import json
from pathlib import Path
import numpy as np
from efficient_llm.a2_breadth.common import read,write,weighted,relative_bootstrap,spearman,SPEC_SHA

def compute(cfg,root):
    curves={};pair={};rho={};end={};runs=[];missing=[];tests=[]
    entries=cfg['downloads']['checkpoints']
    for entry in entries:
        name=entry['revision'];files={b:root/'perseq'/name/(b+'.json') for b in ('parent','B_early','B_S1','B_end')}
        if not all(p.exists() for p in files.values()):missing.append(name);continue
        row={b:read(p) for b,p in files.items()};cnt=np.array(row['parent']['per_sequence_tokens']);par=np.array(row['parent']['per_sequence_nll'])
        curve=dict(tokens=entry['tokens'],step=entry['step'],commit=entry['commit'],parent=weighted(par,cnt),damage={})
        for b in ('B_early','B_S1','B_end'):
            vals=np.array(row[b]['per_sequence_nll']);curve['damage'][b]=weighted(vals-par,cnt)
            tests.append(dict(test='replay:'+name+'/'+b,passed=abs(weighted(vals,cnt)-row[b]['loss'])<=1e-12,spec_sha256=SPEC_SHA))
        curves[name]=curve;runs.append(dict(kind='one_shot',revision=name,**curve))
    stages=[e['revision'] for e in entries[:9]]
    if all(k in curves for k in stages):
        for b in ('B_early','B_S1','B_end'):
            first=curves[stages[0]]['damage'][b]
            for key in stages:curves[key].setdefault('relative_to_first',{})[b]=curves[key]['damage'][b]/first-1 if abs(first)>1e-15 else None
            rho[b]=spearman(np.log([curves[k]['tokens'] for k in stages]),[curves[k]['damage'][b] for k in stages])
    early,late=stages[2],stages[4]
    if early in curves and late in curves:
        pe=read(root/'perseq'/early/'parent.json');pl=read(root/'perseq'/late/'parent.json');cnt=pe['per_sequence_tokens']
        tests.append(dict(test='paired_document_token_counts',passed=cnt==pl['per_sequence_tokens'],spec_sha256=SPEC_SHA))
        for b in ('B_early','B_S1','B_end'):
            a=np.array(read(root/'perseq'/early/(b+'.json'))['per_sequence_nll'])-pe['per_sequence_nll']
            z=np.array(read(root/'perseq'/late/(b+'.json'))['per_sequence_nll'])-pl['per_sequence_nll']
            pair[b]=relative_bootstrap(a,z,cnt)
    if all(e['revision'] in curves for e in entries[-2:]):
        a,z=[curves[e['revision']] for e in entries[-2:]]
        for b in ('B_early','B_S1','B_end'):
            d=a['damage'][b];end[b]=dict(difference=z['damage'][b]-d,relative=(z['damage'][b]-d)/d if abs(d)>1e-15 else None,confounded_data_mix=True)
    stable={}
    for ds in ('dclm','fwe'):
        hs={}
        for arm in ('S72','S188'):
            name=f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3';p=root/'heals'/name/'result.json'
            if not p.exists():missing.append(name);continue
            result=read(p);runs.append(dict(kind='heal',run_id=name,delta_post=result['delta_post'],training_seconds=result['training_seconds']))
            if result['scores']['final']:
                delta=float(np.mean(result['scores']['final']['per_sequence_nll']))-float(np.mean(result['baseline']['per_sequence_nll']))
                tests.append(dict(test='replay:'+name,passed=abs(delta-result['delta_post'])<=1e-12,spec_sha256=SPEC_SHA));hs[arm]=delta
        if len(hs)==2:
            e=hs['S188']-hs['S72'];ref=cfg['annealed_reference'][ds];stable[ds]=dict(delta_post=hs,E_stab=e,relative=e/hs['S72'] if abs(hs['S72'])>1e-15 else None,E_ann=ref['E_ann'],E_ann_relative=ref['E_ann']/ref['P300_delta'],annealed_vs_stable=ref['P300_delta']-hs['S72'])
    if entries[0]['revision'] in curves and entries[-1]['revision'] in curves:
        tests.append(dict(test='ST1_final_parent_below_first',passed=curves[entries[-1]['revision']]['parent']<curves[entries[0]['revision']]['parent'],spec_sha256=SPEC_SHA))
    def sm1():
        p=pair['B_S1'];return p['value']>0 and p['ci95'][0]>0
    def sm3():
        r=curves[stages[-1]]['relative_to_first'];return r['B_early']>r['B_S1']
    clauses={}
    for name,fn in [('SM1',sm1),('SM2',lambda:rho['B_S1']>=.7),('SM3',sm3),('SP1',lambda:len(stable)==2 and all(r['E_stab']>=.008 for r in stable.values()))]:
        try:clauses[name]=None if name=='SP1' and len(stable)!=2 else bool(fn())
        except (KeyError,TypeError):clauses[name]=None
    for path in sorted(root.glob('**/tests*.json')):
        if path==root/'tests.json':continue
        tests.extend(read(path).get('checks',[]))
    for path in (root/'document_checks.json',root/'preparation_checks.json',root/'coordinator_checks.json'):
        if path.exists():tests.extend(read(path).get('checks',[]))
    decision=dict(curves=curves,matched_pair=pair,spearman=rho,descriptive_end_pair=end,stable_healing=stable,clauses=clauses,missing=missing,coverage_complete=not missing,flags=[t for t in tests if not t['passed']],deviations=cfg['deviations'])
    return decision,tests,runs

def finalize(cfg):
    root=Path(cfg['results_root']);d,t,r=compute(cfg,root)
    write(root/'decision_breadth.json',d);write(root/'tests.json',dict(checks=t))
    (root/'runs.jsonl').write_text(''.join(json.dumps(dict(x,spec_sha256=SPEC_SHA))+'\n' for x in r))
    lines=['# A2 Breadth / stable-state healing',f'Spec SHA256: `{SPEC_SHA}`','',f'Coverage complete: {d["coverage_complete"]}. Scientific test flags: {len(d["flags"])}.','']
    lines+=['## SmolLM3 pruning damage','','| Revision | Tokens | Parent NLL | B_early D | B_S1 D | B_end D |','|---|---:|---:|---:|---:|---:|']
    for name,v in d['curves'].items():lines.append('| '+name+' | '+str(v['tokens'])+' | '+' | '.join(f'{x:.9f}' for x in [v['parent'],*v['damage'].values()])+' |')
    lines+=['','## Matched 283B → 755B pair','','| Block | Relative change | Paired 95% CI | Spearman, nine stage1 points |','|---|---:|---|---:|']
    for b,p in d['matched_pair'].items():lines.append(f'| {b} | {p["value"]} | {p["ci95"]} | {d["spearman"].get(b)} |')
    lines+=['','First-to-last relative changes and descriptive stage2→stage3 pair (confounded data mix):','', '```json',json.dumps(dict(relative_to_first={k:v.get('relative_to_first') for k,v in d['curves'].items()},end_pair=d['descriptive_end_pair']),indent=2),'```','','## Stable-state heals','','| Dataset | S72 Δpost | S188 Δpost | E_stab | E_stab / S72 | E_ann | E_ann / P300 | P300 − S72 Δpost |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds,v in d['stable_healing'].items():lines.append('| '+ds+' | '+' | '.join(f'{x:.9f}' if x is not None else 'undefined' for x in [v['delta_post']['S72'],v['delta_post']['S188'],v['E_stab'],v['relative'],v['E_ann'],v['E_ann_relative'],v['annealed_vs_stable']])+' |')
    lines+=['','## Preregistered clauses','',json.dumps(d['clauses']), '', '## Deviations and limitations','',*['- '+x for x in d['deviations']],'- Separate-document scoring uses token-weighted NLL and document-paired resampling (10,000, seed0); no BOS/template, explicit EOS, truncate4096.','- The stage3 end contrast changes data mix and is descriptive only.','- Scientific checks do not stop registered work; failed checks remain in tests.json and decision flags.','',f'Missing results: {d["missing"]}.','', '## Compute ledger','']
    if (root/'ledger.json').exists():lines+=['```json',json.dumps(read(root/'ledger.json'),indent=2),'```']
    lines+=['',f'Expected 2 node-hours plus download (estimate only). Actual execution, scientific coverage and independent native handoff are separate receipts.','',f'Open: {d["missing"]}; Depth C unavailable because its spec is absent. Native terminal/handoff must be verified independently.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Recompute deterministic readout from saved arrays, including fresh paired RNG.
    repeated,_,_=compute(cfg,root)
    write(root/'independent_replay.json',dict(readout_exact=d==repeated,coverage_complete=d['coverage_complete'],method='second CPU recomputation from persisted per-sequence arrays',external_verification_pending=True))
    return d
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();finalize(read(a.config))
