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
    tasks = get_task_dict(task_configs(args.tasks or list(paths), paths), task_manager=TaskManager())
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


def request_digest(rows):
    digest = hashlib.sha256()
    for doc_id, arguments in sorted(rows.items()):
        digest.update(json.dumps([doc_id, arguments], ensure_ascii=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def check_fewshot(args):
    """Build exact harness prompts on CPU before inference; audit split and length."""
    from collections import defaultdict
    from transformers import AutoTokenizer
    from eval.benchmarks import local_dataset_paths, task_configs
    from eval.models import file_hash
    from lm_eval.tasks import TaskManager, get_task_dict
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    # Qwen has no automatically inserted BOS/EOS. Match HFLM's causal input
    # length: tokenize context+continuation, then drop the final target token.
    require(tokenizer.encode('', add_special_tokens=True) == [], 'Unexpected automatic special tokens')
    paths = local_dataset_paths(args.dataset_root, args.dataset_manifest)
    configs = [dict(c, num_fewshot=args.num_fewshot) for c in task_configs(args.tasks, paths)]
    tasks = get_task_dict(configs, task_manager=TaskManager())
    reports = {}
    for name, task in tasks.items():
        split = task.fewshot_cfg.split or task.config.training_split
        require(split is not None and split == task.config.training_split,
                f'{name}: demonstrations must come from the training split')
        require(split not in (task.config.test_split, task.config.validation_split), 'Demonstration/evaluation split overlap')
        task.set_fewshot_seed(args.seed)
        task.build_all_requests()
        rows = defaultdict(list)
        for request in task.instances:
            require(request.request_type == 'loglikelihood', 'Expected likelihood requests')
            rows[request.doc_id].append(request.args)
        maximum, truncated_requests, truncated_docs, requests = 0, 0, 0, 0
        for arguments in rows.values():
            truncated = False
            for context, continuation in arguments:
                require(bool(context), 'Few-shot prompt must be nonempty')
                total = len(tokenizer.encode(context + continuation, add_special_tokens=False))
                maximum = max(maximum, total-1)
                cropped = total > args.max_length+1
                truncated_requests += int(cropped); truncated |= cropped; requests += 1
            truncated_docs += int(truncated)
        reports[name] = dict(rows=len(rows), requests=requests, demonstration_split=split,
            demonstration_pool_rows=len(task.sampler.df), max_input_tokens=maximum,
            truncated_requests=truncated_requests, truncated_documents=truncated_docs,
            requests_sha256=request_digest(rows),
            smoke_requests_sha256=request_digest({k:v for k,v in rows.items() if k < args.smoke_limit}))
    save(args.output, dict(status='passed', tasks=reports, num_fewshot=args.num_fewshot,
        seed=args.seed, max_length=args.max_length, smoke_limit=args.smoke_limit,
        manifest_sha256=file_hash(args.dataset_manifest),
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        truncation_policy='Unchanged HFLM left truncation to trained context; requested shots may be partially removed'))


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
    audit = json.loads(Path(args.fewshot_audit).read_text()) if args.fewshot_audit else None
    if audit:
        require(audit['status']=='passed' and audit['seed']==args.seed and
                audit['num_fewshot']==args.num_fewshot and
                audit['manifest_sha256']==expected['manifest_sha256'], 'Wrong prompt audit')
        require(set(audit['tasks'])==set(expected['tasks']), 'Audit task mismatch')
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
        require(meta['arguments']['seed'] == args.seed and meta['arguments']['num_fewshot'] == args.num_fewshot, 'Evaluation recipe changed')
        require(set(metrics) == set(expected['tasks']), 'Missing or unexpected tasks')
        samples = {t: {} for t in metrics}
        requests = {t: {} for t in metrics}
        for line in (dest / 'eval_samples.jsonl').read_text().splitlines():
            sample = json.loads(line); task = sample['task']; doc_id = sample['doc_id']
            require(doc_id not in samples[task], 'Duplicate scored example')
            samples[task][doc_id] = {key: sample.get(key) for key in ('doc_hash','prompt_hash','target_hash')}
            if audit:
                requests[task][doc_id] = sample['arguments']
            require(all(samples[task][doc_id].values()), 'Missing sample provenance')
        for name, count in expected['tasks'].items():
            rows = min(args.limit, count['rows']) if args.limit else count['rows']
            require(len(samples[name]) == rows, f'Wrong sample count: {name}')
            require(full['n-samples'][name]['effective'] == rows, f'Wrong effective count: {name}')
            require(full['n-shot'][name] == args.num_fewshot, f'Wrong actual shot count: {name}')
            if audit:
                require(meta['context_length']==audit['max_length'], 'Context limit differs from audit')
                require(args.limit is None or args.limit==audit['smoke_limit'], 'Wrong smoke audit limit')
                key = 'smoke_requests_sha256' if args.limit else 'requests_sha256'
                require(request_digest(requests[name])==audit['tasks'][name][key], f'Prompt audit mismatch: {name}')
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
                          manifest_sha256=expected['manifest_sha256'],
                          num_fewshot=args.num_fewshot, seed=args.seed, context_audit=audit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    data = commands.add_parser('data')
    data.add_argument('--dataset-root', required=True)
    data.add_argument('--dataset-manifest', required=True)
    data.add_argument('--tasks', nargs='+')
    data.set_defaults(action=check_data)
    fewshot = commands.add_parser('fewshot')
    fewshot.add_argument('--dataset-root', required=True)
    fewshot.add_argument('--dataset-manifest', required=True)
    fewshot.add_argument('--tokenizer', required=True)
    fewshot.add_argument('--tasks', nargs='+', required=True)
    fewshot.add_argument('--num-fewshot', type=int, choices=[1,5,10,25], required=True)
    fewshot.add_argument('--seed', type=int, default=42)
    fewshot.add_argument('--max-length', type=int, default=2048)
    fewshot.add_argument('--smoke-limit', type=int, default=8)
    fewshot.set_defaults(action=check_fewshot)
    numerics = commands.add_parser('numerics')
    numerics.add_argument('--checkpoint', required=True)
    numerics.set_defaults(action=check_numerics)
    check = commands.add_parser('validate')
    check.add_argument('--directory', required=True)
    check.add_argument('--data-check', required=True)
    check.add_argument('--count', type=int, required=True)
    check.add_argument('--limit', type=int)
    check.add_argument('--num-fewshot', type=int, default=0)
    check.add_argument('--seed', type=int, default=42)
    check.add_argument('--fewshot-audit')
    check.set_defaults(action=validate)
    for subparser in (data, fewshot, numerics, check):
        subparser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
