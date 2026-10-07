"""Offline acceptance and completeness checks for checkpoint evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    print(json.dumps(data), flush=True)


def check_data(args):
    from eval.benchmarks import local_dataset_paths, task_configs
    from lm_eval.tasks import TaskManager, get_task_dict
    from eval.models import file_hash
    from eval.eval_checkpoint import json_default
    paths = local_dataset_paths(args.dataset_root, args.dataset_manifest)
    tasks = get_task_dict(task_configs(list(paths), paths), task_manager=TaskManager())
    counts = {}
    for name, task in tasks.items():
        docs = task.eval_docs
        require(len(docs) > 0, f'Empty task: {name}')
        digest = hashlib.sha256()
        for doc in docs:
            choices = task.doc_to_choice(doc)
            gold = int(task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc))
            require(0 <= gold < len(choices), f'Invalid label in {name}')
            digest.update(json.dumps(doc, sort_keys=True, ensure_ascii=False, default=json_default).encode())
            digest.update(b'\n')
        counts[name] = dict(rows=len(docs), documents_sha256=digest.hexdigest())
    save(args.output, dict(status='passed', tasks=counts,
                          manifest_sha256=file_hash(args.dataset_manifest)))


def check_numerics(args):
    import torch
    from deep_kv.model import Context
    from eval.models import load_checkpoint
    torch.manual_seed(42)
    model, _, meta = load_checkpoint(args.checkpoint, 'cuda', attention_backend='fa4')
    require(meta['step'] == 2500, 'Wrong checkpoint step')
    wrapped = model.wrapped
    buffers = {k: v.clone() for k, v in wrapped.named_buffers()}
    calls = []
    wrapped._fa4_observer = lambda q, k, v, mask: calls.append(tuple(q.shape))
    # Full EMS chunks for the training-forward reference; include two documents.
    ids = torch.randint(10, 10000, (2, 128), device='cuda')
    segments = torch.zeros_like(ids); segments[:, 64:] = 1
    positions = torch.arange(64, device='cuda').repeat(2).expand_as(ids)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(ids, labels=ids, segments=segments)
        ctx = Context(ids, torch.ones_like(ids, dtype=torch.bool), positions, segments)
        ref = wrapped(ctx, compute_auxiliary_losses=False)
        reference_loss = ref['lm_sum'] / ref['lm_count']
    torch.testing.assert_close(actual.loss, reference_loss, rtol=1e-5, atol=1e-5)
    require(len(calls) >= wrapped.backbone.config.num_hidden_layers, 'FA4 did not execute')
    changed = ids.clone(); changed[:, :64] = 10001
    independent = model(changed, segments=segments).logits
    torch.testing.assert_close(actual.logits[:, 64:], independent[:, 64:], rtol=0, atol=0)
    # Real harness calls have variable lengths and causal right padding.
    short = ids[:1, :37]
    alone = model(short).logits
    padded = torch.nn.functional.pad(short, (0, 91), value=0)
    pad_mask = torch.nn.functional.pad(torch.ones_like(short), (0, 91), value=0)
    together = model(padded, attention_mask=pad_mask).logits[:, :37]
    pad_relative = float((alone.float()-together.float()).norm()/alone.float().norm().clamp_min(1e-8))
    require(pad_relative < .01, 'Padding changes real logits beyond BF16 tolerance')
    proxy_delta = None
    if wrapped.family:
        with wrapped.without_proxy():
            disabled = model(short).logits
        proxy_delta = float((alone.float()-disabled.float()).abs().max())
        require(proxy_delta > 0, 'Proxy consumer is inactive')
    # Same weights through dense SDPA: reference for the variable-length adapter.
    wrapped.attention_backend = 'sdpa'
    dense = model(short).logits
    backend_relative = float((alone.float()-dense.float()).norm()/dense.float().norm().clamp_min(1e-8))
    require(backend_relative < .02, 'Cross-backend logits exceed BF16 tolerance')
    for key, value in wrapped.named_buffers():
        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
    require(all(torch.isfinite(x).all() for x in (actual.logits, alone, dense)), 'Nonfinite logits')
    save(args.output, dict(status='passed', arm=meta['arm'], checkpoint=meta['checkpoint'],
        checkpoint_sha256=meta['checkpoint_sha256'], step=meta['step'], fa4_calls=len(calls),
        lm_loss=float(actual.loss), training_forward_loss=float(reference_loss),
        padding_relative_l2=pad_relative, sdpa_relative_l2=backend_relative,
        proxy_disabled_max_difference=proxy_delta, buffers_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def validate(args):
    root = Path(args.directory)
    expected = json.loads(Path(args.data_check).read_text())
    mapping = json.loads((root / 'checkpoints.json').read_text())
    require(len(mapping) == args.count, 'Wrong checkpoint count')
    results, signatures = {}, {}
    for checkpoint, destination in mapping.items():
        dest = Path(destination)
        meta = json.loads((dest / 'eval_metadata.json').read_text())
        full = json.loads((dest / 'eval_benchmarks_full.json').read_text())
        metrics = json.loads((dest / 'eval_benchmarks.json').read_text())
        require(meta['status'] == 'completed' and meta['step'] == 2500, 'Incomplete checkpoint evaluation')
        require(meta['checkpoint'] == checkpoint and meta['attention_backend'] == 'fa4', 'Wrong model/backend')
        require(meta['dataset_manifest_sha256'] == expected['manifest_sha256'], 'Data manifest changed')
        require(meta['arguments']['limit'] == args.limit, 'Diagnostic/full evaluation mismatch')
        require(meta['arguments']['seed'] == 42 and meta['arguments']['num_fewshot'] == 0, 'Evaluation recipe changed')
        require(set(metrics) == set(expected['tasks']), 'Missing or unexpected tasks')
        samples = {t: {} for t in metrics}
        for line in (dest / 'eval_samples.jsonl').read_text().splitlines():
            sample = json.loads(line); task = sample['task']; doc_id = sample['doc_id']
            require(doc_id not in samples[task], 'Duplicate scored example')
            samples[task][doc_id] = {key: sample.get(key) for key in ('doc_hash','prompt_hash','target_hash')}
            require(all(samples[task][doc_id].values()), 'Missing sample provenance')
        for name, count in expected['tasks'].items():
            rows = min(args.limit, count['rows']) if args.limit else count['rows']
            require(len(samples[name]) == rows, f'Wrong sample count: {name}')
            require(full['n-samples'][name]['effective'] == rows, f'Wrong effective count: {name}')
            for key, value in metrics[name].items():
                if key.startswith(('acc,', 'acc_norm,')):
                    require(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1, 'Invalid accuracy')
        arm = meta['arm']; require(arm not in results, 'Duplicate arm')
        results[arm] = metrics
        if signatures:
            require(samples == signatures, 'Models were scored on different documents/prompts/targets')
        else:
            signatures = samples
    save(args.output, dict(status='passed', arms=results, limit=args.limit,
                          task_counts={k: len(v) for k, v in signatures.items()},
                          manifest_sha256=expected['manifest_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    data = commands.add_parser('data')
    data.add_argument('--dataset-root', required=True)
    data.add_argument('--dataset-manifest', required=True)
    data.set_defaults(action=check_data)
    numerics = commands.add_parser('numerics')
    numerics.add_argument('--checkpoint', required=True)
    numerics.set_defaults(action=check_numerics)
    check = commands.add_parser('validate')
    check.add_argument('--directory', required=True)
    check.add_argument('--data-check', required=True)
    check.add_argument('--count', type=int, required=True)
    check.add_argument('--limit', type=int)
    check.set_defaults(action=validate)
    for subparser in (data, numerics, check):
        subparser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
