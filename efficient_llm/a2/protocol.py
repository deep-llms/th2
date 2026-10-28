"""Preregistered constants and pure selection/decision functions. No GPU imports."""
from dataclasses import asdict, dataclass
from decimal import Decimal
import math
import statistics

DATASETS = ('dclm', 'fwe', 'nem')
PARENTS = ('P300', 'P1T', 'S72', 'S188')
ARMS = (*PARENTS, 'X72')
TOKENS_PER_STEP = 262144
LRS = (3e-5, 1e-4, 3e-4)
FIXED_SETS = {'S1': (8, 9, 10), 'S3': (3, 6, 9), 'S4': (1, 2, 6)}
CAPS = {'0A': (15., 18.), '0B': (14.5, 17.)}
REPOS = {
    'dclm': 'dclm', 'fwe': 'fineweb-edu-1.4t', 'nem': 'nemotron-hq',
}

# Only this reviewed test-scope amendment may reuse the original packed data.
# Neither the sampling recipe nor the bytes/order of prepared data changed.
ORIGINAL_DATA_SPEC = '9854e8338cecab6cd6fccd1700e7196c62a329a3b44fa76689d31d11ecbc9afd'
PREFLIGHT_SCOPE_SPEC = 'f91f401f066a3054f3c866a51e6cc4d68aee8bb4e54948998d6c926f7dff5e64'
DEPENDENCY_SPEC = 'b5507004791c958ec433005d97be8482561a6e3b010ec17a9be420f533204cca'
REPRESENTATIVE = 'dclm/P300'


def compatible_data_spec(prepared, current):
    return (prepared == current or (prepared, current) == (ORIGINAL_DATA_SPEC, PREFLIGHT_SCOPE_SPEC)
        or current == DEPENDENCY_SPEC and prepared in (ORIGINAL_DATA_SPEC, PREFLIGHT_SCOPE_SPEC))


def shared_check_sets(all_sets, assets, *, scope='all'):
    """One representative probe, covering each actual removal pattern once."""
    early = scope in ('dclm_P300_S1_grid', 'dclm_S1_pair')
    if scope not in ('all', 'dclm_P300_S1_grid', 'dclm_S1_pair'):
        raise ValueError('Unknown preflight scope')
    datasets = ('dclm',) if early else DATASETS
    parents = ('P300','P1T') if scope == 'dclm_S1_pair' else ('P300',) if early else PARENTS
    if set(all_sets) != set(datasets):
        raise ValueError('Freeze S2 for every dataset before shared checks')
    reference = assets[REPRESENTATIVE]['code_files']
    required_code = {'modeling_opensci.py', 'configuration_opensci.py'}
    if not required_code <= reference.keys() or any(
        assets[f'{ds}/{p}']['code_files'] != reference for ds in datasets for p in parents
    ):
        raise ValueError('Shared checks require identical pinned native source')
    patterns = {}
    for ds in datasets:
        sets = all_sets[ds]
        if set(sets) != ({'S1'} if early else {'S1', 'S2', 'S3', 'S4'}):
            raise ValueError('All four frozen sets required per dataset')
        for name, groups in sets.items():
            if len(groups) != 3:
                raise ValueError('Production pruning requires three groups')
            if name in FIXED_SETS and sorted(groups) != sorted(FIXED_SETS[name]):
                raise ValueError('Fixed removal set changed')
            layers = tuple(layers_for(groups))
            patterns.setdefault(layers, (f'{ds}/{name}', list(groups)))
    return dict(patterns.values())


def validate_shared_check(receipt, all_sets, assets, provenance, sets_sha256, *, scope='all'):
    expected = {tuple(layers_for(g)) for g in shared_check_sets(all_sets, assets,scope=scope).values()}
    expected.update(((0,), (11,), (23,)))
    if (receipt.get('passed') is not True or receipt.get('scope') != 'shared_mechanism'
        or receipt.get('representative') != REPRESENTATIVE
        or receipt.get('provenance') != provenance or receipt.get('sets_sha256') != sets_sha256
        or {tuple(p) for p in receipt.get('removal_patterns', [])} != expected):
        raise ValueError('Missing, stale or incomplete shared numerical evidence')


def validate_run_scope(run, gate, scope, rerun_of=None):
    if gate.get('launch_scope', 'all') != scope:
        raise ValueError('Preflight launch scope mismatch')
    if scope == 'all':
        return
    if scope == 'dclm_S1_pair':
        if (run.dataset != 'dclm' or run.arm not in ('P300','P1T') or run.set != 'S1'
            or run.budget not in ('0.03B','0.05B','0.2B')
            or run.budget == '0.03B' and run.arm != 'P1T'):
            raise ValueError('Run not covered by bounded pair')
        expected = LRS if run.budget == '0.03B' else (gate.get('chosen_lrs',{}).get(run.arm),)
        if not rerun_of:
            if run.lr not in expected or run.suffix:
                raise ValueError('Pair requires registered grid or tuning-selected LR')
        else:
            original=Run(run.dataset,run.arm,run.set,run.budget,run.lr*2)
            if run.lr*2 not in expected or run.suffix!='-rerun' or original.run_id!=rerun_of:
                raise ValueError('Only one registered half-LR retry allowed')
        return
    if scope != 'dclm_P300_S1_grid' or (run.dataset,run.arm,run.set,run.budget) != ('dclm','P300','S1','0.03B'):
        raise ValueError('Run not covered by bounded preflight')
    if not rerun_of:
        if run.lr not in LRS or run.suffix:
            raise ValueError('Only registered grid points allowed')
    else:
        original=Run('dclm','P300','S1','0.03B',run.lr*2)
        if run.lr*2 not in LRS or run.suffix!='-rerun' or original.run_id!=rerun_of:
            raise ValueError('Only one registered half-LR retry allowed')
# Order: P300, P1T, S72, S188. Branch weights always use P1T main code.
COMMITS = {
 'dclm': ('9d02563017b25fd0a1e1849b7cd2bfd02b30d4e0', '90481ca923a6cba5c8e107e2249e09fcd32ec617',
          '2ee702ba4c05cd1494c530d81f4e338bcf70536b', '0340fcb7b7717498ad8bcb229a653a31a7af0877'),
 'fwe': ('de8d9259fcac769157070eb63cd67fa89d4c2db1', 'ca1795337a73d08235c904609c4b6daa89ecd6ca',
         '2acd776f32a524e413b02f9c2c5f1f3a49c8d4e5', '85e25569105f1c84b0d653f6f8e69df1e30bd3de'),
 'nem': ('5cc94475d50d221ac201909c36aa61855fb2c50f', 'c9ccbca7125d7072cead55a1f0a07e53a3fb6de1',
         'a69f22d194ba460ce35e9d29a8d6cc4af919aca1', '10f3e6ae5e55164ce14709839c230a107d809aef'),
}


def parent_ref(dataset, parent):
    index = PARENTS.index(parent)
    return dict(repo=f'open-sci/open-sci-ref-v0.01-1.7b-{REPOS[dataset]}-'
                     f'{"300B" if parent == "P300" else "1T"}-4096',
                revision=COMMITS[dataset][index],
                code_revision=COMMITS[dataset][0 if parent == 'P300' else 1])


def steps_for(budget):
    if budget not in ('0.03B', '0.05B', '0.2B', '0.8B'):
        raise ValueError('Unregistered healing budget')
    return int(Decimal(budget[:-1]) * 10**9 // TOKENS_PER_STEP)


def warmup_steps(steps):
    return max(10, math.ceil(.02 * steps))


def lr_for_update(update, steps, peak):
    """LR actually used by update j, not the value after scheduler.step()."""
    w = warmup_steps(steps)
    if steps <= w or not 1 <= update <= steps or peak <= 0:
        raise ValueError('Invalid update/schedule')
    return peak * (update / w if update <= w else (steps-update)/(steps-w))


def layers_for(groups):
    if len(set(groups)) != len(groups) or any(g not in range(12) for g in groups):
        raise ValueError('Invalid original group indices')
    return sorted(layer for g in groups for layer in (2*g, 2*g+1))


def choose_s2(group_bi, usable):
    """T1/T3 eligibility only. Inputs must all come from the frozen SMALL set."""
    if not {'P300', 'P1T'} <= set(usable) or not set(usable) <= set(PARENTS):
        raise ValueError('S2 needs both annealed parents and only valid branch parents')
    for p in usable:
        if len(group_bi[p]) != 12 or not all(map(math.isfinite, group_bi[p])):
            raise ValueError('Invalid group BI')
    mean = [statistics.mean(group_bi[p][g] for p in usable) for g in range(12)]
    order = sorted(range(1, 11), key=lambda g: (mean[g], g))
    raw = sorted(order[:3]); selected = set(raw)
    excluded = {frozenset(v) for v in FIXED_SETS.values()}
    # Swap in the best group outside the CURRENT set, not the removed member.
    while frozenset(selected) in excluded:
        add = next(g for g in order if g not in selected)
        remove = max(selected, key=lambda g: (mean[g], g))
        selected.remove(remove); selected.add(add)
    return dict(raw=raw, groups=sorted(selected), mean_bi=mean, usable=list(usable),
                per_parent={p: sorted(sorted(range(1,11), key=lambda g:(group_bi[p][g],g))[:3])
                            for p in usable})


def choose_lr(rows):
    candidates = {r['lr']: r['val_tune_post'] for r in rows
                  if not r.get('diverged', False) and not r.get('rerun_of')
                  and r.get('status') == 'complete' and r['lr'] in LRS
                  and math.isfinite(r['val_tune_post'])}
    # Caller supplies exactly one row for EACH original grid point, including failures.
    if len(rows) != 3 or {r['lr'] for r in rows} != set(LRS) or any(r.get('rerun_of') for r in rows):
        raise ValueError('Require all three original grid records, no retries')
    if not candidates:
        return None
    best = min(candidates.values())
    return min(lr for lr, value in candidates.items() if value <= best + .001 + 1e-12)


def _finite(x):
    return x is not None and math.isfinite(x)


def _sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def length_decision(main, pre, short):
    """main/pre: dataset -> two contrasts; short: twelve contrasts, None for failures."""
    if set(main) != set(DATASETS) or set(pre) != set(DATASETS) or len(short) != 12:
        raise ValueError('Expected three dataset pairs and twelve short contrasts')
    def means(data):
        if any(len(v) != 2 for v in data.values()): raise ValueError('Expected S1/S2')
        return {d: statistics.mean(v) if all(map(_finite,v)) else None for d,v in data.items()}
    ed, epd = means(main), means(pre)
    incomplete = not all(map(_finite, ed.values()))
    e = None if incomplete else statistics.mean(ed.values())
    ep = statistics.mean(epd.values()) if all(map(_finite, epd.values())) else None
    same = e is not None and e != 0 and all(_sign(v) == _sign(e) for v in ed.values())
    samepre = ep is not None and ep != 0 and all(_sign(v) == _sign(ep) for v in epd.values())
    count = sum(_finite(v) and e is not None and e != 0 and _sign(v) == _sign(e) for v in short)
    # Decimal conversion avoids label changes at the preregistered decimal boundaries.
    D = lambda x: Decimal(str(x))
    small = e is not None and abs(D(e)) < Decimal('.003')
    disagree = (not incomplete and any(D(v) >= Decimal('.003') for v in ed.values())
                and any(D(v) <= Decimal('-.003') for v in ed.values()))
    clauses = dict(INCOMPLETE=incomplete,
        GO=e is not None and abs(D(e)) >= Decimal('.008') and same and count >= 10,
        REFRAME=small and ep is not None and abs(D(ep)) >= Decimal('.05') and samepre,
        STOP=(small and ep is not None and (abs(D(ep)) < Decimal('.03') or not samepre)) or disagree)
    label = next((name for name, value in clauses.items() if value), 'GREY')
    return dict(label=label, clauses=clauses, E=e, E_d=ed, E_pre=ep, E_pre_d=epd,
                sign_count=count, stop_reason='dataset_sign_disagreement' if disagree else 'small_effect' if clauses['STOP'] else None)


def lr_state_decision(values, x_lrs=()):
    if len(values) != 6: raise ValueError('Expected six A_init contrasts')
    good = [v for v in values if _finite(v)]
    mean = statistics.mean(good) if good else None
    negative, positive = sum(v < 0 for v in good), sum(v > 0 for v in good)
    clauses = {'INCOMPLETE': len(good) < 5,
        'prune before you anneal': mean is not None and mean <= -.01 and negative >= 5,
        'anneal first': mean is not None and mean >= .01 and positive >= 5}
    return dict(label=next((k for k,v in clauses.items() if v), 'no consistent effect meeting the threshold'),
                clauses=clauses, mean=mean, defined=len(good), negative=negative, positive=positive,
                grid_edge=any(lr == max(LRS) for lr in x_lrs), values=list(values))


@dataclass(frozen=True)
class Run:
    dataset: str
    arm: str
    set: str
    budget: str
    lr: float | None
    suffix: str = ''

    def __post_init__(self):
        if self.dataset not in DATASETS or self.arm not in ARMS or self.set not in ('S1','S2','S3','S4'):
            raise ValueError('Unregistered dataset/arm/layer set')
        steps_for(self.budget)
        suffixes=('', '-rerun', '-t14a', '-t14b', '-t14a-rerun', '-t14b-rerun')
        if self.suffix not in suffixes: raise ValueError('Unregistered repeat')
        if self.lr is not None and self.lr not in (tuple(x/2 for x in LRS) if self.suffix.endswith('-rerun') else LRS):
            raise ValueError('Unregistered learning rate')
        if (self.budget=='0.03B' and self.set!='S1'
            or self.budget=='0.05B' and self.arm=='X72'
            or self.budget=='0.2B' and self.set not in ('S1','S2')
            or self.budget=='0.8B' and (self.dataset!='nem' or self.arm not in ('P300','P1T') or self.set!='S1')
            or 't14' in self.suffix and (self.dataset,self.arm,self.set,self.budget)!=('nem','P300','S1','0.05B')):
            raise ValueError('Run is outside the preregistered matrix')

    @property
    def parent(self): return 'S72' if self.arm == 'X72' else self.arm
    @property
    def teacher(self): return 'P300' if self.arm == 'X72' else self.arm
    @property
    def part(self): return '0A' if self.arm in ('P300', 'P1T') else '0B'
    @property
    def steps(self): return steps_for(self.budget)
    @property
    def run_id(self):
        if self.lr is None: lr = 'chosen'
        else:
            mantissa, exponent = format(self.lr, '.8e').split('e')
            lr = mantissa.rstrip('0').rstrip('.') + 'e' + str(int(exponent))
        return f'P0-{self.dataset}-{self.arm}-{self.set}-{self.budget}-lr{lr}{self.suffix}'
    def record(self):
        return dict(asdict(self), run_id=self.run_id, parent=self.parent, teacher=self.teacher,
                    part=self.part, steps=self.steps, heal_tokens=self.steps*TOKENS_PER_STEP)


def matrix(chosen=None):
    """All 127 preregistered runs; chosen keyed by (dataset, ARM), never parent."""
    chosen = {} if chosen is None else chosen
    runs = []
    for arms in (('P300','P1T'), ('S72','X72','S188')):
        for ds in DATASETS:
            for arm in arms:
                runs.extend(Run(ds,arm,'S1','0.03B',lr) for lr in LRS)
                lr = chosen.get((ds,arm))
                if arm != 'X72':
                    runs.extend(Run(ds,arm,s,'0.05B',lr) for s in ('S1','S2','S3','S4'))
                runs.extend(Run(ds,arm,s,'0.2B',lr) for s in ('S1','S2'))
                if ds == 'nem' and arm in ('P300','P1T'):
                    runs.append(Run(ds,arm,'S1','0.8B',lr))
                if ds == 'nem' and arm == 'P300':
                    runs.extend(Run(ds,arm,'S1','0.05B',lr,suffix) for suffix in ('-t14a','-t14b'))
    return runs
