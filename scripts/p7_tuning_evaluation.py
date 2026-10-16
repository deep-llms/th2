"""Freeze completed tuning checkpoints from split queues for the existing eval suite."""
from pathlib import Path

from scripts.p7_tuning_queue import ALL_CASES, check_case_config
from scripts.proxy_confirmation_queue import evaluation_reference, validate_training, checkpoint_path
from scripts.train_then_burn import digest
from scripts.p7_completed_evaluation import read


def selected_cases(sources):
    """Exclude interrupted models, even if they have an intermediate checkpoint."""
    selected = {}
    for source in sources:
        root = Path(source)
        state = read(root / 'run.json')
        if state['status'] == 'running':
            raise ValueError('Stop training before freezing evaluation scope')
        passed = {j['name'] for j in state['jobs'] if j['status'] == 'ok'}
        for case in ALL_CASES:
            label = case[0]
            if not (root / label / 'validated.json').exists():
                continue
            if not {label+'-train', label+'-validate'} <= passed:
                raise ValueError('Validator has no successful queue receipt: '+label)
            if label in selected:
                raise ValueError('Duplicate completed tuning case: '+label)
            selected[label] = str(root)
    if not selected:
        raise ValueError('No completed tuning checkpoints')
    return [(selected[c[0]], c) for c in ALL_CASES if c[0] in selected]


def identities(scope):
    known = {case[0]: case for case in ALL_CASES}
    selected = scope['completed']
    labels = [item['label'] for item in selected]
    if (not labels or len(set(labels)) != len(labels)
            or labels != [c[0] for c in ALL_CASES if c[0] in labels]):
        raise ValueError('Changed tuning selection or order')
    items = []
    for item in selected:
        case = known[item['label']]
        if item['training_root'] not in scope['training_roots']:
            raise ValueError('Unexpected training source')
        items.append((str(checkpoint_path(item['training_root'], case)), *case))
    for label, arm in (('reference-A', 'A'), ('reference-lm', 'P7-simple-lm')):
        ck = Path(scope['reference_screen']) / 'supervised/run/seed-42' / arm / 'checkpoint-2500'
        items.append((str(ck), label, arm, .1, 42))
    return items


def validate_scope(scope):
    items = identities(scope)
    if (scope['checkpoints'] != [r[0] for r in items]
            or set(scope['specs']) != set(scope['checkpoints'])
            or len(set(scope['checkpoints'])) != len(items)
            or scope['completed_labels'] != [r['label'] for r in scope['completed']]
            or scope['evaluation_seed'] != 42):
        raise ValueError('Changed tuning checkpoint paths or evaluation seed')
    for ck, label, arm, weight, seed in items:
        spec = scope['specs'][ck]
        if (spec['label'], spec['arm'], spec['step'], spec['lambda_max'], spec['seed']) != (label, arm, 2500, weight, seed):
            raise ValueError('Changed tuning checkpoint identity')


def build_scope(sources, reference):
    from deep_kv.report import report
    previous = evaluation_reference(reference)
    sources = [str(Path(p)) for p in sources]
    selected = selected_cases(sources)
    rows = []
    for root in sources:
        cases = tuple(c for source, c in selected if source == root)
        if cases:
            rows.extend(validate_training(root, cases=cases))
    scope = dict(kind='p7-tuning-completed', training_roots=sources,
        completed=[dict(training_root=r, label=c[0]) for r, c in selected],
        completed_labels=[c[0] for _, c in selected],
        reference_screen=previous['source_screen'], reference=str(reference),
        protocols=previous['protocols'], data_root=previous['data_root'],
        manifest=previous['manifest'], manifest_sha256=previous['manifest_sha256'],
        versions=previous['versions'], evaluation_seed=42, finetuning=False, paired_loss=False)
    specs = {}; fingerprints = {}; tokenizer = None; eval_fingerprint = None
    source_by_label = {case[0]: Path(root) for root, case in selected}
    for ck, label, arm, weight, seed in identities(scope):
        folder = Path(ck).parent
        c = read(folder/'train_config.json'); r = read(folder/'result.json')
        report(folder.parent, [arm], expected_step=2500, expected_seed=seed, expected_attention_backend='fa4')
        if (r['status'] != 'complete' or r['global_step'] != 2500 or r['arm'] != arm
                or c['training']['max_steps'] != 2500 or c['training']['warmup_steps'] != 125
                or c['training']['seed'] != seed or c['training']['data_seed'] != seed
                or c['model_config']['num_hidden_layers'] != 12 or not c['training']['bf16']
                or c['pilot']['arm'] != arm or c['pilot']['proxy_lambda_max'] != weight
                or c['pilot']['attention_backend'] != 'fa4'
                or (arm != 'A' and c['pilot']['proxy_module_seed'] != seed+1)
                or read(Path(ck)/'trainer_state.json')['global_step'] != 2500):
            raise ValueError('Invalid completed checkpoint: '+label)
        if label in source_by_label:
            jobs = read(source_by_label[label]/'jobs.snapshot.json')['jobs']
            summary = next(j for j in jobs if j['name']=='summary')
            argv = summary['argv']; config_root = Path(argv[argv.index('--config-root')+1])
            check_case_config(c, read(config_root/(label+'.json')))
        pair = (c['train_fingerprint'], c['eval_fingerprint'])
        fingerprints.setdefault(seed, pair)
        if tokenizer is None:
            tokenizer, eval_fingerprint = c['model']['tokenizer_name'], c['eval_fingerprint']
        if (fingerprints[seed] != pair or tokenizer != c['model']['tokenizer_name']
                or eval_fingerprint != c['eval_fingerprint']):
            raise ValueError('Checkpoint data/tokenizer mismatch')
        specs[ck] = dict(label=label, arm=arm, step=2500, lambda_max=weight, seed=seed,
                         checkpoint_sha256=digest(Path(ck)/'model.safetensors'))
    scope.update(checkpoints=list(specs), specs=specs)
    validate_scope(scope)
    return scope, rows
