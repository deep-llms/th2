"""Read saved sequence arrays only; exact Q paired counts and explicit clauses."""
from pathlib import Path
import json
import numpy as np
from efficient_llm.a2.quant_stats import nll_report, CONTRASTS, REFERENCES
from efficient_llm.a2_breadth.common import spearman
from efficient_llm.a2_depth.common import read, write, SPEC_SHA, DATASETS, PARENTS, SETS


def clauses(tables, field):
    per = {}
    for ds in DATASETS:
        t = tables.get(ds, {})
        if not all(w in t for w in ('W1', 'W8', 'W9')):
            per[ds] = {k: None for k in ('A1_W9_gt_W8', 'A1_W8_gt_W1', 'A2_L_ann', 'A2_L_stab', 'A3_negative', 'A4')}
            continue
        def c(w, name):
            return t[w]['contrasts'][name][field]
        def compare(w1, n1, w2, n2, op):
            a, b = c(w1, n1), c(w2, n2)
            return bool(op(a, b)) if a is not None and b is not None else None
        per[ds] = dict(
            A1_W9_gt_W8=compare('W9', 'R_short', 'W8', 'R_short', lambda a,b:a>b),
            A1_W8_gt_W1=compare('W8', 'R_short', 'W1', 'R_short', lambda a,b:a>b),
            A2_L_ann=compare('W1', 'L_ann', 'W8', 'L_ann', lambda a,b:a>b),
            A2_L_stab=compare('W1', 'L_stab', 'W8', 'L_stab', lambda a,b:a>b),
            A3_negative=c('W1','R_short') < 0 if c('W1','R_short') is not None else None,
            A4=compare('W9', 'R_short', 'W9', 'L_stab', lambda a,b:a>=b))
    def all_known(names):
        values = [per[ds][name] for ds in DATASETS for name in names]
        return all(values) if all(v is not None for v in values) else None
    signs = [per[ds]['A3_negative'] for ds in DATASETS]
    return dict(scale=field, per_dataset=per, predictions=dict(
        A1=all_known(['A1_W9_gt_W8', 'A1_W8_gt_W1']),
        A2=all_known(['A2_L_ann','A2_L_stab']),
        A3=sum(signs)>=2 if all(v is not None for v in signs) else None,
        A4=all_known(['A4'])))


def compute(cfg, root):
    root = Path(root)
    values, tests, runs, failures = {}, [], [], []
    def check(name, passed, detail=None):
        tests.append(dict(test=name, passed=bool(passed), detail=detail, spec_sha256=SPEC_SHA))
    for ds in DATASETS:
        values[ds] = {name:{} for name in ('parent',)+SETS}
        for parent in PARENTS:
            directory = root/'part_a'/ds/parent
            if (directory/'tests.json').exists():
                tests.extend(dict(t, dataset=ds, parent=parent) for t in read(directory/'tests.json')['checks'])
            for name in ('parent',)+SETS:
                path = directory/(name+'.json')
                try:
                    row = read(path)
                    arr = np.asarray(row['per_sequence_nll'], dtype=np.float64)
                    good = arr.ndim == 1 and len(arr)>0 and np.isfinite(arr).all()
                    check('saved_array:'+ds+'/'+parent+'/'+name, good and row['spec_sha256']==SPEC_SHA)
                    if not good:
                        continue
                    check('saved_mean:'+ds+'/'+parent+'/'+name, abs(float(arr.mean())-row['loss'])<=1e-12)
                    values[ds][name][parent] = arr
                    runs.append(dict(run_id=f'D-{ds}-{parent}-{name}-noheal', dataset=ds,
                        parent=parent, layer_set=name, val=float(arr.mean()), sequences=len(arr),
                        per_sequence_file=str(path), spec_sha256=SPEC_SHA))
                except Exception as error:
                    check('saved_array:'+ds+'/'+parent+'/'+name, False, repr(error))
    tables, correlations = {}, {}
    for ds in DATASETS:
        # Missing/malformed one set must not hide the remaining valid sets.
        v = values[ds]
        base = v['parent']
        if set(base) != set(PARENTS):
            failures.append(dict(dataset=ds, reason='incomplete parent references'))
            continue
        n = len(base[PARENTS[0]])
        if any(len(base[p])!=n for p in PARENTS):
            failures.append(dict(dataset=ds, reason='unpaired parent arrays'))
            continue
        valid = {'parent':base}
        for name in SETS:
            good = set(v[name]) == set(PARENTS) and all(len(x)==n for x in v[name].values())
            check('paired_coverage:'+ds+'/'+name, good)
            if good:
                valid[name] = v[name]
        report = nll_report({ds:valid}, resamples=cfg.get('resamples',10000), seed=0)
        tables[ds] = report['datasets'][ds]
        counts = np.random.default_rng(0).multinomial(n, np.full(n,1/n), size=cfg.get('resamples',10000))
        base_matrix = np.stack([base[p] for p in PARENTS], axis=1)
        for name in tables[ds]:
            damage = np.stack([valid[name][p] for p in PARENTS], axis=1)-base_matrix
            draws = counts@damage/n
            for contrast, ref in REFERENCES.items():
                numerator = draws@CONTRASTS[contrast]
                denominator = draws[:,PARENTS.index(ref)]
                mask = np.isfinite(denominator) & (np.abs(denominator)>1e-15)
                ratios = numerator[mask]/denominator[mask]
                tables[ds][name]['contrasts'][contrast].update(
                    relative_ci95=np.quantile(ratios,[.025,.975]).tolist() if len(ratios) else None,
                    relative_undefined_draws=int((~mask).sum()),
                    reference_damage_ci95=np.quantile(denominator,[.025,.975]).tolist())
        correlations[ds] = {}
        for field in ('estimate', 'relative_to_reference_damage'):
            correlations[ds][field] = {}
            for contrast in ('R_short','L_ann','L_stab'):
                series = [tables[ds].get(f'W{k}',{}).get('contrasts',{}).get(contrast,{}).get(field) for k in range(1,10)]
                correlations[ds][field][contrast] = spearman(range(1,10),series) if all(x is not None for x in series) else None
    complete = len(runs)==156 and all(set(tables.get(ds,{}))==set(SETS) for ds in DATASETS)
    decision = dict(part='A', coverage_complete=complete, datasets=tables, correlations=correlations,
        predictions_nats=clauses(tables,'estimate'), predictions_relative=clauses(tables,'relative_to_reference_damage'),
        prediction_scale_note='Spec defines contrasts in nats and additionally requests relative values. Both clause evaluations are reported explicitly, without choosing a scale after observing results.',
        resamples=cfg.get('resamples',10000), seed=0,
        uncertainty='paired final-set sequences; not training seeds', readout_failures=failures,
        flags=[t for t in tests if not t['passed']], deviations=cfg.get('deviations',[]),
        remaining=['DT4', 'Part B: 12 heals and report', 'DT6', 'Part C: three 12.8B heals', 'Part C downstream', 'independent final replay and native handoff'])
    return decision, tests, runs


def finalize(cfg):
    root = Path(cfg['results_root'])
    decision, tests, runs = compute(cfg, root)
    write(root/'decision_depth.json', decision)
    write(root/'tests.json', dict(checks=tests))
    (root/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in runs))
    lines = ['# A2 Depth — Part A', '', f'Spec SHA-256: `{SPEC_SHA}`.', '',
        f'Coverage complete: **{decision["coverage_complete"]}**. Saved arrays: {len(runs)}/156 (144 pruned + 12 parents).', '',
        'FP32 parameters rounded to BF16 saved precision, native BF16 AMP/SDPA; final CulturaX set, unchanged A2 scorer.', '',
        'Paired sequence bootstrap: 10,000 draws, seed 0. Relative intervals recompute the reference damage in every paired draw.', '',
        '| Dataset | Set | L_ann / D(P300) | L_stab / D(S72) | R_short / D(S72) | R_long / D(S188) |',
        '|---|---|---:|---:|---:|---:|']
    for ds, table in decision['datasets'].items():
        for name, row in table.items():
            cells = []
            for c in ('L_ann','L_stab','R_short','R_long'):
                x = row['contrasts'][c]
                cells.append(f'{x["relative_to_reference_damage"]:.6f}' if x['relative_to_reference_damage'] is not None else 'undefined')
            lines.append('| '+' | '.join([ds,name]+cells)+' |')
    lines += ['', 'Full absolute/relative estimates, intervals, damages, correlations and every prediction clause are in `decision_depth.json`.', '',
              decision['prediction_scale_note'], '',
              'Predictions (nats): `'+json.dumps(decision['predictions_nats']['predictions'])+'`.', '',
              'Predictions (relative): `'+json.dumps(decision['predictions_relative']['predictions'])+'`.', '',
              'Scientific flags: '+str(len(decision['flags']))+'. Failed checks retain their original values; remaining registered work continues.', '',
              'Deviations: '+json.dumps(decision['deviations'])+'.', '',
              'Ledger: see `ledger.json`; estimates and elapsed time are report-only.', '',
              'Remaining: DT4, Part B and its report, DT6, Part C, downstream, independent replay and native handoff.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('DEPTH_PART_A_NOTE', json.dumps(dict(coverage_complete=decision['coverage_complete'],
          nats=decision['predictions_nats'], relative=decision['predictions_relative'])), flush=True)
    return decision
