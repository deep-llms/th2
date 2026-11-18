"""Dense-initialized local-order fine-tuning, exact recipe/RNG guarded resume."""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from .model import Config, Model
from .data import Rows, sha
from .train import amp, lr_at, write_json, source_hashes as pilot_sources
from .local_policy import (LOCAL_ORDERS, LocalRouter, policy_probabilities,
                           sample_actions, policy_gradient_loss,
                           full_information_loss, routed_nll)


def sources():
    root = Path(__file__).resolve().parents[2]
    return {**pilot_sources(), **{p: sha(root/p) for p in (
        'efficient_llm/warm_order/local_policy.py',
        'efficient_llm/warm_order/local_train.py')}}


def update(model, router, optimizer, head_optimizer, x, arm, exploration, generator):
    """All labels/policy/sample/body CE see the same pre-update parameters."""
    optimizer.zero_grad(set_to_none=True)
    if head_optimizer is not None:
        head_optimizer.zero_grad(set_to_none=True)
    logits = None
    extra_forwards = 0
    prefix_tokens = 0
    if arm == 'C':
        choices = torch.zeros(len(x), dtype=torch.long, device=x.device)
    elif arm == 'U':
        # Match the actual initial CF/PG distribution, including epsilon.
        probs = x.new_tensor([.85, .05, .05, .05], dtype=torch.float32)
        probs = ((1-exploration)*probs + exploration/4).expand(len(x), -1)
        choices = sample_actions(probs, generator)
    else:
        logits = router(model.features(x))
        probs = policy_probabilities(logits, exploration)
        choices = sample_actions(probs.detach(), generator)
        prefix_tokens = len(x) * model.c.prefix
    with torch.no_grad():
        if arm == 'CF':
            costs = torch.stack([model.nll(x, path) for path in LOCAL_ORDERS], -1)
            extra_forwards = 4
        elif arm == 'PG':
            baseline = model.nll(x, LOCAL_ORDERS[0])
            extra_forwards = 1
    values = routed_nll(model, x, choices)
    body_loss = values.mean()
    if arm == 'CF':
        head_loss = full_information_loss(logits, costs)
    elif arm == 'PG':
        head_loss = policy_gradient_loss(logits, choices, values, baseline, exploration)
    else:
        head_loss = body_loss.new_zeros(())
    if not torch.isfinite(body_loss) or not torch.isfinite(head_loss):
        raise ValueError('Nonfinite loss')
    body_loss.backward()
    if logits is not None:
        head_loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if logits is not None:
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if head_optimizer is not None:
        head_optimizer.step()
    return dict(loss=float(body_loss.detach()), router_loss=float(head_loss.detach()),
                grad_norm=float(norm), counts=torch.bincount(choices, minlength=4).tolist(),
                ce_block_tokens=6*x.numel(), extra_forward_block_tokens=6*x.numel()*extra_forwards,
                prefix_forward_block_tokens=prefix_tokens)


def train(config, out, device='cuda', stop_after=None, resume=False):
    c = dict(config); out = Path(out)
    if c['arm'] not in ('C', 'U', 'CF', 'PG') or not 0 < c['exploration'] < 1:
        raise ValueError('Invalid arm/exploration')
    if str(device).startswith('cuda') and torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU required')
    torch.set_num_threads(c.get('threads', 4))
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed'])
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError('Nonempty fresh output')
    if resume and (out/'complete.json').exists():
        raise ValueError('Already complete')
    rows = Rows(c['data'], c['model']['length'])
    ids = np.load(c['indices'], allow_pickle=False)
    batch, total = c['batch'], c['steps']; needed = batch*total
    if (len(ids) != needed or len(np.unique(ids)) != needed or ids.dtype.kind not in 'iu'
            or ids.min() < 0 or ids.max() >= len(rows.ids)):
        raise ValueError('Invalid unique training indices')
    inputs = {p: sha(p) for p in [c['data']+'.bin', c['data']+'.starts', c['indices'], c['initial_weights']]}
    if inputs != c['expected_inputs']:
        raise ValueError('Input identity mismatch')
    recipe = dict(config=c, inputs=inputs, source=sources(), torch=torch.__version__,
                  numpy=np.__version__, precision='bf16' if device.startswith('cuda') else 'fp32')
    initial = torch.load(c['initial_weights'], map_location='cpu', weights_only=False)
    if initial['config'] != c['model']:
        raise ValueError('Initial architecture mismatch')
    model = Model(Config(**c['model'])).to(device)
    model.load_state_dict(initial['model']); del initial
    router = LocalRouter(model.c.d).to(device) if c['arm'] in ('CF', 'PG') else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], betas=(.9, .95), weight_decay=.1)
    head_optimizer = torch.optim.AdamW(router.parameters(), lr=c['router_lr'], weight_decay=.01) if router is not None else None
    generator = torch.Generator(device=device).manual_seed(c['seed']+9271)
    step = 0; elapsed = 0.
    counts = [0]*4
    compute = dict(ce_block_tokens=0, extra_forward_block_tokens=0, prefix_forward_block_tokens=0)
    if resume:
        receipt = json.loads((out/'checkpoint.json').read_text())
        if receipt['sha256'] != sha(out/'fullstate.pt'):
            raise ValueError('Checkpoint hash mismatch')
        state = torch.load(out/'fullstate.pt', map_location='cpu', weights_only=False)
        if state['recipe'] != recipe:
            raise ValueError('Resume recipe mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if router is not None:
            router.load_state_dict(state['router']); head_optimizer.load_state_dict(state['head_optimizer'])
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['sample_rng'])
        step, elapsed, counts, compute = state['step'], state['elapsed'], state['counts'], state['compute']
        if receipt['step'] != step:
            raise ValueError('Checkpoint step mismatch')
        trace = [json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()] if (out/'trace.jsonl').exists() else []
        if trace and trace[-1]['step'] > step:
            raise ValueError('Trace ahead of checkpoint')
    else:
        out.mkdir(parents=True, exist_ok=True); write_json(out/'recipe.json', recipe)
    end = min(total, stop_after) if stop_after is not None else total
    if end <= step:
        raise ValueError('Stop must advance checkpoint')
    model.train()
    if router is not None:
        router.train()
    start = time.monotonic()
    for s in range(step, end):
        ix = ids[s*batch:(s+1)*batch]; x = rows.batch(ix, device)
        lr = lr_at(s, total, c['lr'], c['warmup'])
        for g in optimizer.param_groups:
            g['lr'] = lr
        with amp(device):
            record = update(model, router, optimizer, head_optimizer, x, c['arm'], c['exploration'], generator)
        step = s+1
        counts = [a+b for a,b in zip(counts, record['counts'])]
        for k in compute:
            compute[k] += record[k]
        if step % c.get('log_every', 50) == 0 or step == end:
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            record.update(step=step, lr=lr, first_row=int(ix[0]), elapsed_seconds=elapsed+time.monotonic()-start)
            with (out/'trace.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
        if step % c.get('save_every', 1000) == 0 or step == end:
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         router=router.state_dict() if router is not None else None,
                         head_optimizer=head_optimizer.state_dict() if head_optimizer is not None else None,
                         recipe=recipe, step=step, elapsed=elapsed+time.monotonic()-start, counts=counts,
                         compute=compute, sample_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                         numpy_rng=np.random.get_state(), python_rng=random.getstate())
            torch.save(state, out/'fullstate.tmp'); os.replace(out/'fullstate.tmp', out/'fullstate.pt')
            write_json(out/'checkpoint.json', dict(step=step, sha256=sha(out/'fullstate.pt'), bytes=(out/'fullstate.pt').stat().st_size))
    if step == total:
        torch.save(dict(config=c['model'], model=model.state_dict(), router=router.state_dict() if router is not None else None), out/'weights.pt')
        write_json(out/'complete.json', dict(step=step, arm=c['arm'], seed=c['seed'], input_tokens=needed*model.c.length,
                   scored_tokens=needed*(model.c.length-model.c.prefix), counts=counts, compute=compute,
                   elapsed_seconds=elapsed+time.monotonic()-start, weights_sha256=sha(out/'weights.pt'),
                   fullstate_sha256=sha(out/'fullstate.pt'), initial_weights_sha256=inputs[c['initial_weights']]))
    return model, router, optimizer, head_optimizer


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--stop-after', type=int); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); train(json.loads(Path(a.config).read_text()), a.out, a.device, a.stop_after, a.resume)


if __name__ == '__main__':
    main()
