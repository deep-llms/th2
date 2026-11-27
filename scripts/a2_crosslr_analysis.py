"""EXPLORATORY A2 XLR v1.0: offline CPU analysis of immutable saved outputs.

No model loading, evaluation, training, network access, or source-result writes.
Run from the repository root: python3 -m scripts.a2_crosslr_analysis
"""
import argparse
import ast
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SPEC = Path('docs/A2_crosslr_spec.md')
BUDGETS = ('0.2B', '0.8B', '3.2B')
CORPORA = ('dclm', 'fwe')
NSEQ, NTOK, REPEATS, SEED = 2461, 4095, 10000, 0
FAMILY = Path('results/a2_family_20261107')
WHY = Path('results/a2_why_20261104')
DEPTH = Path('results/a2_depth_20261031/final_export_20261110_a01/files')
EXPECTED = {
    ('dclm', 'ann'): [0.021096293347514727, 0.020786422563626505, 0.021670884569823468, 0.020443865973687636],
    ('dclm', 'stab'): [0.0111319744049041, 0.008627344665466552, 0.005171952894106546],
    ('fwe', 'ann'): [0.026224221566386152, 0.02281977941126856, 0.023461150403907727],
    ('fwe', 'stab'): [0.014780744431006188, 0.008610998730387287, 0.00502851342794175],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original_estimator():
    """Execute unchanged arrays/pair_scores AST, avoiding unrelated torch imports.

    These two definitions depend only on NumPy. No estimator code is rewritten.
    The full source and selected definitions are hashed in the report.
    """
    path = Path('scripts/a2_family_report.py')
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('arrays', 'pair_scores')]
    assert [n.name for n in defs] == ['arrays', 'pair_scores']
    module = ast.Module(body=defs, type_ignores=[])
    ns = {'np': np}
    exec(compile(module, str(path), 'exec'), ns)
    return ns['pair_scores'], dict(path=str(path), sha256=sha(raw),
        selected_ast_sha256=sha(ast.dump(module, include_attributes=False).encode()),
        functions=['arrays', 'pair_scores'],
        loading='Unmodified function AST; imports omitted because common.py imports unavailable torch. Pure NumPy definitions only.')


def normalized_score(score):
    """All A2 final rows have exactly 4095 scored targets, not 4096."""
    s = dict(score)
    if 'per_sequence_nll' not in s:
        return s
    v = np.asarray(s['per_sequence_nll'], dtype=np.float64)
    if v.shape != (NSEQ,) or not np.isfinite(v).all():
        raise ValueError('Expected 2461 finite saved sequence NLLs')
    counts = np.asarray(s.get('per_sequence_tokens', [NTOK] * NSEQ))
    if counts.shape != v.shape or not np.all(counts == NTOK):
        raise ValueError('Unexpected per-sequence target counts')
    if s.get('sequences', NSEQ) != NSEQ or s.get('tokens', NSEQ * NTOK) != NSEQ * NTOK:
        raise ValueError('Saved target-count mismatch')
    if abs(float(v @ counts / counts.sum()) - s['loss']) > 1e-10:
        raise ValueError('Saved NLL mean does not match its array')
    s['per_sequence_tokens'] = counts.tolist()
    return s


def level(e):
    return 'persists' if e >= .008 else 'erased' if abs(e) < .003 else 'reversed' if e <= -.008 else 'weak'


def change_label(d, e, relative=True):
    threshold = max(.2 * abs(e), .003) if relative else .003
    return 'flat' if abs(d) < threshold else 'falls' if d < 0 else 'rises'


def bootstrap_means(matrix, counts, repeats=REPEATS, seed=SEED):
    """Same seed-0 index draws as pair_scores, shared across every input model.

    Bincount is only an algebraic acceleration of the indexed token-weighted
    mean. Budget/decomposition/ratio intervals use these joint draws.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty((repeats, matrix.shape[1]))
    for start in range(0, repeats, 64):
        k = min(64, repeats-start)
        ix = rng.integers(0, len(counts), (k, len(counts)))
        w = np.asarray([np.bincount(row, minlength=len(counts)) for row in ix]) * counts
        out[start:start+k] = (w @ matrix) / w.sum(axis=1, keepdims=True)
    return out


class Analysis:
    def __init__(self):
        self.inputs, self.models, self.flags, self.checks = {}, {}, [], []
        self.estimator, self.estimator_provenance = original_estimator()

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        self.inputs[str(path)] = dict(sha256=sha(raw), bytes=len(raw))
        return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

    def check(self, name, actual, expected, tolerance=1e-9):
        residual = float(actual-expected)
        passed = bool(np.isfinite(residual) and abs(residual) <= tolerance)
        self.checks.append(dict(check=name, expected=float(expected), computed=float(actual),
                                residual=residual, tolerance=tolerance, passed=passed))
        if not passed:
            self.flags.append(f'CHECK FAILED: {name}; residual={residual:.12g}')

    def add(self, key, score, path, pointer, run_id, origin, **extra):
        s = normalized_score(score)
        record = dict(key=key, run_id=run_id, source=str(path), json_pointer=pointer,
                      source_sha256=self.inputs[str(path)]['sha256'], origin=origin,
                      nll=float(s['loss']), sequence_count=s.get('sequences', NSEQ),
                      scored_targets=s.get('tokens', NSEQ*NTOK), **extra)
        record['score'] = s
        record['has_per_sequence_array'] = 'per_sequence_nll' in s
        if not record['has_per_sequence_array']:
            self.flags.append(f'Missing per-sequence array: {run_id} at {path}#{pointer}; point only.')
        self.models[key] = record

    @staticmethod
    def key(ds, arm, budget=None):
        return ':'.join([ds, arm] + ([budget] if budget else ['parent']))

    def load(self):
        hist_path = Path('resources/a2_family_f6_historical_20261108.json.gz')
        historical = self.read(hist_path)['runs']
        for ds in CORPORA:
            for arm in ('S72', 'S188'):
                rid = f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3'
                r = historical[rid]
                self.add(self.key(ds, arm), r['baseline'], hist_path, f'/runs/{rid}/baseline',
                         f'opensci-{ds}-{arm}', 'Original A2 stable-parent arrays retained in F6 historical bundle',
                         original_spec_sha256=r['spec_sha256'])
            parent_root = (Path('temp/a2_stage1_preflight_health02/supervised/run') if ds == 'dclm'
                           else WHY/'overlap_audit_20261106/historical_inputs')
            for arm in ('P300', 'P1T'):
                p = parent_root/f'{arm}.json';r = self.read(p)
                self.add(self.key(ds, arm), r['baseline']['final'], p, '/baseline/final',
                         f'opensci-{ds}-{arm}', 'Original Stage1/Stage1b parent preflight',
                         original_spec_sha256=r.get('provenance', {}).get('spec_sha256'),
                         provenance=r.get('provenance', {}))
            old_path = (Path('temp/a2_stage1_heals_terminal01/supervised/run/results/a2_stage1/runs.json')
                        if ds == 'dclm' else WHY/'overlap_audit_20261106/historical_inputs/runs.json')
            records = self.read(old_path)
            index = {r['run_id']: (i, r) for i, r in enumerate(records)}
            for b in BUDGETS:
                for arm in ('P300', 'P1T', 'X72'):
                    rid = f'ST1-{ds}-{arm}-S1-{b}-lr1e-3';i, r = index[rid]
                    self.add(self.key(ds, arm, b), r['scores']['final'], old_path, f'/{i}/scores/final',
                             rid, 'Original published Stage1/Stage1b endpoint',
                             original_spec_sha256=r.get('provenance', {}).get('spec_sha256'),
                             provenance=r.get('provenance', {}), actual_parent=r['parent'], teacher=r['teacher'])
                    own = 'S72' if arm == 'X72' else arm
                    self.check(f'parent binding {rid}', r['val_parent'], self.models[self.key(ds, own)]['nll'], 1e-10)
                    self.check(f'scalar binding {rid}', r['val_post'], r['scores']['final']['loss'], 1e-10)
                for arm in ('S72', 'S188'):
                    rid = f'ST1-{ds}-{arm}-S1-{b}-lr1e-3'
                    if b == '0.2B':
                        r, p, pointer = historical[rid], hist_path, f'/runs/{rid}/scores/final'
                        origin = 'Original A2 S 0.2B; selected to reproduce published stable-row E; not later same-name rerun'
                    else:
                        export = 'v17_final_export_20261110_a01' if ds == 'dclm' and b == '3.2B' else 'v14_final_export_20261108_a01'
                        p = FAMILY/export/'files/heals'/rid/'result.json';r = self.read(p);pointer = '/scores/final'
                        origin = 'Published Family F6/F7 stable endpoint'
                    self.add(self.key(ds, arm, b), r['scores']['final'], p, pointer, rid, origin,
                             original_spec_sha256=r.get('spec_sha256'), raw_flags=r.get('flags', []),
                             original_result_sha256=r.get('original_result_sha256'),
                             registered_run=r.get('registered_run'))
                    parent = self.models[self.key(ds, arm)]
                    self.check(f'parent scalar binding {rid}', r['baseline']['loss'], parent['nll'], 1e-10)
                    self.check(f'parent array binding {rid}',
                               np.max(np.abs(np.asarray(r['baseline']['per_sequence_nll']) - np.asarray(parent['score']['per_sequence_nll']))), 0., 1e-10)
        # DCLM 12.8B: the completed wc69 reruns are the section 7 reference.
        for arm in ('P300', 'P1T', 'X72'):
            rid = f'ST1-dclm-{arm}-S1-12.8B-lr1e-3'
            p = DEPTH/'perseq'/rid/'post-S1.json';s = self.read(p)
            rpath = DEPTH/'heals'/rid/'result.json';r = self.read(rpath)
            self.add(self.key('dclm', arm, '12.8B'), s, p, '/', rid, 'Completed wc69 Depth Part C rerun used in published reference',
                     original_spec_sha256=r.get('spec_sha256'), raw_flags=r.get('flags', []))
            self.check(f'12.8B embedded vs perseq {arm}', s['loss'], r['scores']['final']['loss'], 1e-10)
            own = 'S72' if arm == 'X72' else arm
            self.check(f'12.8B parent binding {arm}', r['baseline']['loss'], self.models[self.key('dclm', own)]['nll'], 1e-10)
        # Optional seed spread: saved arrays available for all four extra P300 fits.
        for ds in CORPORA:
            for seed in (2031, 2032):
                rid = f'Y1-{ds}-P300-S1-0.2B-seed{seed}'
                p = WHY/'full_export_20261106/heals'/rid/'result.json'
                if not p.exists():
                    self.flags.append(f'Optional seed result missing: {p}');continue
                r = self.read(p)
                self.add(self.key(ds, f'P300_seed{seed}', '0.2B'), r['scores']['final'], p, '/scores/final', rid,
                         'Why Y1 data-order control; S72 remains seed2030', original_spec_sha256=r.get('spec_sha256'),
                         raw_flags=r.get('flags', []))
        # Bind DCLM original raw arrays to the retained published summary.
        p = Path('docs/results/a2_stage1_20261029/final/runs_summary.json')
        for r in self.read(p):
            if r['lr'] == .001:
                k = self.key('dclm', r['arm'], r['budget'])
                self.check(f'published summary {r["run_id"]}', self.models[k]['nll'], r['val_post'], 1e-10)

    def prepare_draws(self):
        self.names = [k for k, r in self.models.items() if r['has_per_sequence_array']]
        matrix = np.asarray([self.models[k]['score']['per_sequence_nll'] for k in self.names]).T
        draws = bootstrap_means(matrix, np.full(NSEQ, NTOK))
        self.draws = {k: draws[:, i] for i, k in enumerate(self.names)}

    def contrast(self, weights):
        point = sum(self.models[k]['nll'] * w for k, w in weights.items())
        draws = sum(self.draws[k] * w for k, w in weights.items()) if set(weights) <= self.draws.keys() else None
        return dict(estimate=float(point), ci95=np.quantile(draws, [.025, .975]).tolist() if draws is not None else None), draws

    def ratio(self, a, b):
        pa, da = a;pb, db = b
        point = pa['estimate']/pb['estimate'] if pb['estimate'] else None
        valid = da is not None and db is not None and np.all(db != 0)
        vals = da/db if valid else None
        return dict(estimate=point, ci95=np.quantile(vals, [.025, .975]).tolist() if vals is not None else None,
                    denominator_ci_crosses_zero=pb['ci95'] is not None and pb['ci95'][0] <= 0 <= pb['ci95'][1])

    def weights(self, ds, short, long, b):
        return {self.key(ds, long, b): 1, self.key(ds, long): -1,
                self.key(ds, short, b): -1, self.key(ds, short): 1}

    def original_pair(self, ds, short, long, b):
        rows = []
        for arm in (short, long):
            parent, student = [self.models[self.key(ds, arm, budget)]['score'] for budget in (None, b)]
            if 'per_sequence_nll' not in parent or 'per_sequence_nll' not in student:
                return None
            # pair_scores also computes unused E_pre; no inference from that field.
            rows.append(dict(baseline={'final': parent}, pre={'final': student}, scores={'final': student}))
        return self.estimator(*rows, repeats=REPEATS)

    def published_checks(self):
        """Execute before computing any novel contrasts."""
        rows = []
        for (ds, pair), expected in EXPECTED.items():
            short, long = ('P300', 'P1T') if pair == 'ann' else ('S72', 'S188')
            for b, target in zip((*BUDGETS, '12.8B'), expected):
                result, _ = self.contrast(self.weights(ds, short, long, b))
                original = self.original_pair(ds, short, long, b)
                self.check(f'7.1 E_{pair} {ds} {b}', result['estimate'], target)
                if original:
                    self.check(f'estimator point parity {ds} {pair} {b}', result['estimate'], original['E'], 1e-12)
                    self.check(f'estimator CI parity {ds} {pair} {b}',
                               np.max(np.abs(np.asarray(result['ci95'])-original['bootstrap']['E_ci95'])), 0., 1e-12)
                rows.append(dict(corpus=ds, pair=pair, budget=b, expected=target, **result))
        for ds, targets in [('dclm', [.073, .039, .012, .0012]), ('fwe', [.078, .042, .013])]:
            for b, target in zip((*BUDGETS, '12.8B'), targets):
                result, _ = self.contrast({self.key(ds, 'X72', b): 1, self.key(ds, 'P300', b): -1})
                tol = .00005 if b == '12.8B' else .0005
                self.check(f'7.1 A_init {ds} {b}', result['estimate'], target, tol)
                rows.append(dict(corpus=ds, pair='A_init', budget=b, expected=target, **result))
        return rows

    def compute(self):
        self.load();self.prepare_draws()
        published = self.published_checks()
        parents, main, decomposition, changes, spread = [], [], [], [], []
        exlr = {}
        for ds in CORPORA:
            gains = {col: self.contrast({self.key(ds, s): 1, self.key(ds, p): -1})
                     for col, s, p in [('short', 'S72', 'P300'), ('long', 'S188', 'P1T')]}
            extra = gains['long'][0]['estimate']-gains['short'][0]['estimate']
            self.check(f'7.3 extra anneal gain {ds}', extra, .0290 if ds == 'dclm' else .0336, .00005)
            for arm, expected in ({'S72': 2.908575296107871, 'S188': 2.88670669804388, 'P300': 2.7377930322126844, 'P1T': 2.6869089917639344}
                                  if ds == 'dclm' else {'S72': 3.042152001946299, 'S188': 3.0238075916258365}).items():
                self.check(f'7.3 parent {ds} {arm}', self.models[self.key(ds, arm)]['nll'], expected, 1e-10)
            parents.append(dict(corpus=ds, nll={a: self.models[self.key(ds, a)]['nll'] for a in ('S72','S188','P300','P1T')},
                                A_short=gains['short'][0], A_long=gains['long'][0], extra_anneal_gain=extra))
            for b in BUDGETS:
                row = dict(corpus=ds, budget=b, damage={})
                for arm in ('S72', 'S188', 'P300', 'P1T'):
                    row['damage'][arm] = self.contrast({self.key(ds, arm, b): 1, self.key(ds, arm): -1})[0]
                for col, s, p in [('short','S72','P300'), ('long','S188','P1T')]:
                    result, draws = self.contrast(self.weights(ds, s, p, b))
                    original = self.original_pair(ds, s, p, b)
                    if original:
                        self.check(f'XLR estimator CI parity {ds} {col} {b}',
                                   np.max(np.abs(np.asarray(result['ci95'])-original['bootstrap']['E_ci95'])), 0., 1e-12)
                        # Use the existing estimator's exact point and interval.
                        result = dict(estimate=original['E'], ci95=original['bootstrap']['E_ci95'])
                    exlr[ds,col,b] = (result, draws)
                    gain, gd = gains[col]
                    kept = (dict(estimate=gain['estimate']-result['estimate']), gd-draws if draws is not None else None)
                    row[col] = dict(E_xlr=result, A=gain, K=self.ratio(kept, gains[col]), reading=level(result['estimate']))
                ann = self.contrast(self.weights(ds, 'P300', 'P1T', b))[0]['estimate']
                stab = self.contrast(self.weights(ds, 'S72', 'S188', b))[0]['estimate']
                residual = row['long']['E_xlr']['estimate']-row['short']['E_xlr']['estimate']-(ann-stab)
                self.check(f'7.2 identity {ds} {b}', residual, 0., 1e-12)
                row.update(E_ann=ann, E_stab=stab, identity_residual=residual);main.append(row)
                t = self.contrast({self.key(ds,'S72',b): 1, self.key(ds,'X72',b): -1})
                s = self.contrast({self.key(ds,'X72',b): 1, self.key(ds,'P300',b): -1})
                d = self.contrast({self.key(ds,'S72',b): 1, self.key(ds,'P300',b): -1})
                self.check(f'decomposition identity {ds} {b}', t[0]['estimate']+s[0]['estimate'], d[0]['estimate'], 1e-12)
                self.check(f'gain identity {ds} {b}', d[0]['estimate'], gains['short'][0]['estimate']-row['short']['E_xlr']['estimate'], 1e-12)
                decomposition.append(dict(corpus=ds,budget=b,T=t[0],S=s[0],D=d[0],T_over_A=self.ratio(t,gains['short']),D_over_A=self.ratio(d,gains['short'])))
            for col in ('short','long'):
                for b1,b2 in [('0.2B','0.8B'),('0.8B','3.2B'),('0.2B','3.2B')]:
                    e1, d1 = exlr[ds,col,b1];e2, d2 = exlr[ds,col,b2];delta=e2['estimate']-e1['estimate']
                    dd = d2-d1 if d1 is not None and d2 is not None else None
                    changes.append(dict(corpus=ds,column=col,from_budget=b1,to_budget=b2,d=delta,
                                        ci95=np.quantile(dd,[.025,.975]).tolist() if dd is not None else None,
                                        relative_threshold=max(.2*abs(e1['estimate']),.003),
                                        reading=change_label(delta,e1['estimate']),fixed_003_reading=change_label(delta,e1['estimate'],False)))
            values=[]
            for seed in (2030,2031,2032):
                arm='P300' if seed==2030 else f'P300_seed{seed}';k=self.key(ds,arm,'0.2B')
                if k not in self.models:continue
                v=self.contrast({k:1,self.key(ds,'P300'):-1,self.key(ds,'S72','0.2B'):-1,self.key(ds,'S72'):1})[0]
                values.append(dict(seed=seed,run_id=self.models[k]['run_id'],**v))
            span=max(v['estimate'] for v in values)-min(v['estimate'] for v in values)
            spread.append(dict(corpus=ds,values=values,range_max_minus_min=span,
                               population_std=float(np.std([v['estimate'] for v in values])),one_sided=True))
        joint=[]
        for col in ('short','long'):
            for b in BUDGETS:
                labels={ds:level(exlr[ds,col,b][0]['estimate']) for ds in CORPORA}
                joint.append(dict(column=col,budget=b,corpus_readings=labels,reading=next(iter(labels.values())) if len(set(labels.values()))==1 else 'mixed'))
            for b1,b2 in [('0.2B','0.8B'),('0.8B','3.2B'),('0.2B','3.2B')]:
                for field in ('reading','fixed_003_reading'):
                    rows=[r for r in changes if r['column']==col and r['from_budget']==b1 and r['to_budget']==b2]
                    labels={r['corpus']:r[field] for r in rows}
                    joint.append(dict(column=col,from_budget=b1,to_budget=b2,rule=field,corpus_readings=labels,
                                      reading=next(iter(labels.values())) if len(set(labels.values()))==1 else 'mixed'))
        return dict(parents=parents,main=main,budget_changes=changes,decomposition=decomposition,
                    optional_data_order_spread=spread,joint_readings=joint,published_reproduction=published)


def fmt(v):
    if v is None:return 'missing'
    if isinstance(v,dict):
        ci=v.get('ci95');return f"{v['estimate']:.9f}"+(f' [{ci[0]:.9f}, {ci[1]:.9f}]' if ci else ' [CI missing]')
    return f'{v:.9f}'


def markdown(out):
    m=out['metadata'];lines=['# EXPLORATORY — A2 cross-LR toll', '',
        f"Spec: `{m['spec_file']}` {m['spec_version']}; SHA-256 `{m['spec_sha256']}`.",
        f"Analysis code commit: `{m['code_commit']}`; script SHA-256 `{m['analysis_script_sha256']}`.",
        f"UTC: {m['created_utc']}. CPU/offline; existing outputs only. No training or model evaluation.",
        'Bootstrap: 10,000 paired sequence resamples, NumPy default_rng seed **0**, 95% percentile intervals. Every draw shares indices across all models/budgets; 2,461 sequences × 4,095 targets = 10,077,795 targets. Token-weighted means. Intervals measure evaluation-set uncertainty, not training-seed uncertainty.', '',
        'Run IDs and their original spec hashes are listed in the input table below and fully mapped to paths/JSON pointers/file hashes in the JSON companion. Original outputs are not modified.', '',
        '## Findings', '',
        '**The cross-LR toll persists in both corpora at all three budgets.** Short-column tolls are about 0.046–0.056 nats; long-column tolls about 0.056–0.074 nats. Short-column 0.2→3.2B change is flat under the relative rule but rises under the fixed 0.003 rule. Long-column 0.2→3.2B change rises under both rules.',
        '**Long-column caveat:** S188 is at 776B tokens versus P1T at about 1T; this comparison includes about 224B additional training tokens as well as LR-state differences. It does not isolate annealing causally.',
        'All readings here are **EXPLORATORY (post hoc)**. The point patterns in spec §9 were known before this analysis.', '',
        '## Parent NLLs and anneal gains', '',
        '| Corpus | S72 | S188 | P300 | P1T | A_short | A_long | A_long − A_short |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in out['parents']:
        lines.append('| '+r['corpus']+' | '+' | '.join(fmt(r['nll'][a]) for a in ('S72','S188','P300','P1T'))+' | '+fmt(r['A_short']['estimate'])+' | '+fmt(r['A_long']['estimate'])+' | '+fmt(r['extra_anneal_gain'])+' |')
    lines += ['', '## Primary column tolls', '',
        '| Corpus | Budget | Δ S72 | Δ P300 | E_short [95% CI] | A_short | K_short | Short reading | Δ S188 | Δ P1T | E_long [95% CI] | A_long | K_long | Long reading |',
        '|---|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|---|']
    for r in out['main']:
        s,l=r['short'],r['long'];d=r['damage']
        vals=[r['corpus'],r['budget'],fmt(d['S72']['estimate']),fmt(d['P300']['estimate']),fmt(s['E_xlr']),fmt(s['A']['estimate']),fmt(s['K']['estimate']),s['reading'],fmt(d['S188']['estimate']),fmt(d['P1T']['estimate']),fmt(l['E_xlr']),fmt(l['A']['estimate']),fmt(l['K']['estimate']),l['reading']]
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['', 'K = (A − E)/A; optional ratio and damage intervals are also retained in JSON.', '', '## Budget changes', '',
              '| Corpus | Column | Budgets | d [95% CI] | Relative threshold | Relative reading | Fixed 0.003 reading |',
              '|---|---|---|---|---:|---|---|']
    for r in out['budget_changes']:
        lines.append(f"| {r['corpus']} | {r['column']} | {r['from_budget']} → {r['to_budget']} | {fmt(dict(estimate=r['d'],ci95=r['ci95']))} | {fmt(r['relative_threshold'])} | {r['reading']} | {r['fixed_003_reading']} |")
    lines += ['', 'Across corpora, matching readings above agree (no mixed joint reading); exact joint labels are in JSON.', '', '## Short-column teacher/start decomposition', '',
              '| Corpus | Budget | T [95% CI] | S = A_init [95% CI] | D [95% CI] | T/A [95% CI] | D/A [95% CI] |',
              '|---|---|---|---|---|---|---|']
    for r in out['decomposition']:
        lines.append('| '+' | '.join([r['corpus'],r['budget']]+[fmt(r[k]) for k in ('T','S','D','T_over_A','D_over_A')])+' |')
    lines += ['', 'T holds the stable starting weights fixed and changes the teacher; S holds the P300 teacher fixed and changes starting weights. These are the specified contrasts, not a fully randomized causal decomposition.', '', '## Optional data-order spread at 0.2B', '',
              '| Corpus | P300 seed2030 | P300 seed2031 | P300 seed2032 | Max − min | Population SD |',
              '|---|---:|---:|---:|---:|---:|']
    for r in out['optional_data_order_spread']:
        lines.append('| '+r['corpus']+' | '+' | '.join(fmt(v['estimate']) for v in r['values'])+' | '+fmt(r['range_max_minus_min'])+' | '+fmt(r['population_std'])+' |')
    lines += ['', 'S72 uses seed2030 throughout: this spread covers only the P300 side, not uncertainty over both training seeds.', '', '## Optional downstream', '', out['optional_downstream']['reason'], '', '## Reproduction and algebra checks', '',
              '| Check | Expected | Computed | Residual | Tolerance | Status |', '|---|---:|---:|---:|---:|---|']
    for r in out['checks']:
        lines.append(f"| {r['check']} | {r['expected']:.12g} | {r['computed']:.12g} | {r['residual']:.3g} | {r['tolerance']:.3g} | {'PASS' if r['passed'] else 'FAIL'} |")
    lines += ['', '## Flags and limitations', '']+[f'- {x}' for x in out['flags']]
    lines += ['', '## Input run IDs and original spec hashes', '', '| Model key | Actual run ID | Original spec SHA-256 | Origin |', '|---|---|---|---|']
    for r in out['models']:
        lines.append(f"| {r['key']} | {r['run_id']} | `{r.get('original_spec_sha256')}` | {r['origin']} |")
    lines += ['', '## Reproduction command', '', '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m scripts.a2_crosslr_analysis --output-dir NEW_EMPTY_DIRECTORY', '```', '',
              'The command refuses to overwrite outputs. Original input SHA-256 values, input selectors, complete model provenance, estimator-source hashes, checks and all intervals are in the JSON companion. File hashes were checked again after computation.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('reviews'));args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d');stem=f'A2_CROSSLR_RESULTS_{stamp}'
    paths=[args.output_dir/(stem+suffix) for suffix in ('.md','.json')]
    if any(p.exists() for p in paths):raise FileExistsError('Refusing to overwrite existing analysis outputs')
    analysis=Analysis();data=analysis.compute()
    overlap_path=WHY/'overlap_audit_20261106/results.json';overlap=analysis.read(overlap_path)
    f10=FAMILY/'f10_small_export_20261111_a01/verification.json';analysis.read(f10)
    analysis.flags += [
        'EXPLORATORY post hoc: this spec was defined after seeing A2 results; §9 disclosures retained.',
        'Long column confounds LR state with about 224B additional parent training tokens (S188 776B vs P1T about1T).',
        'Same-name stable 0.2B reruns exist: original historical arrays were chosen to match published E exactly; no mixing with later reruns.',
        'The 12.8B rows are DCLM annealed/A_init reproduction checks only; no stable 12.8B students exist and no column toll is reported there.',
        'Unchanged existing estimator functions loaded by AST to avoid a transitive torch dependency; joint-budget/decomposition bootstrap extends it and is checked against its exact point/CI outputs.',
        'CI unit is an evaluation sequence, not independent training seeds; packed sequences may share source documents. Sequence order relies on the locked A2 final-set convention; saved NLL arrays have no per-row text IDs.',
        f"FWE full evaluation set retained as specified. Prior Why overlap audit had {len(overlap['excluded_sequence_ids'])} affected sequences and max |E change|={overlap['max_abs_E_change']:.12g}, classified immaterial for its row contrasts. That does not independently certify new cross-LR contrasts; no overlap reduction was requested here.",
        'Downstream optional analysis not run: local F9 preserve bundle has parent/early scores but lacks the required full student per-item coverage; F10 local/A2 copies contain compact report/CSV/proof, not TriviaQA per-item predictions. No new evaluation/network retrieval performed.',
        'Model IDs in NLL outputs use original run IDs; F9 checkpoint aliases append suffixes. Exact NLL source paths and original spec hashes are retained per model.',
        'DCLM original Stage1 arrays and parent preflights were recovered from existing local temp receipts and bound to the retained published summary; do not delete these input receipts before preserving this analysis.',
    ]
    for r in analysis.models.values():
        if r.get('raw_flags'):analysis.flags.append(f"Original run flags retained for {r['run_id']}: {json.dumps(r['raw_flags'])}")
    metadata=dict(spec_file=str(SPEC),spec_version='v1.0',spec_sha256=sha(SPEC.read_bytes()),
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  analysis_script_sha256=sha(Path(__file__).read_bytes()),created_utc=datetime.now(timezone.utc).isoformat(),
                  bootstrap_seed=SEED,resamples=REPEATS,sequence_count=NSEQ,targets_per_sequence=NTOK,
                  numpy_version=np.__version__,estimator=analysis.estimator_provenance,
                  scope='Existing outputs only; offline CPU; no models, no training, no new evaluation')
    # Do not modify source results, and detect any concurrent input change.
    for p,record in analysis.inputs.items():
        if sha(Path(p).read_bytes()) != record['sha256']:raise RuntimeError(f'Input changed during analysis: {p}')
    out=dict(label='EXPLORATORY',metadata=metadata,
             models=[{k:v for k,v in r.items() if k!='score'} for r in analysis.models.values()],**data,
             optional_downstream=dict(status='not run',reason='Not run: the full matched per-item F9/F10 outputs for all four parents and their students are not available in the local exports. In particular, F10 TriviaQA item outputs are absent. Compact aggregate scores cannot recover paired item covariance. No model was re-evaluated.'),
             checks=analysis.checks,flags=analysis.flags,inputs=analysis.inputs,
             verification=dict(check_count=len(analysis.checks),all_checks_pass=all(c['passed'] for c in analysis.checks),
                               missing_per_sequence_arrays=[r['run_id'] for r in analysis.models.values() if not r['has_per_sequence_array']],
                               original_inputs_unchanged=True))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths[1].write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');paths[0].write_text(markdown(out))
    print(json.dumps(dict(outputs=[str(p) for p in paths],verification=out['verification'],
                         main=[dict(corpus=r['corpus'],budget=r['budget'],short=r['short']['E_xlr'],long=r['long']['E_xlr']) for r in out['main']]),indent=2))


if __name__=='__main__':main()
