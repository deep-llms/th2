"""Loop screen rounds (8 jobs, one GPU each) from scratch on verified DCLM rows.

Round 1: docs/LOOP_SCREEN_20261123.md (seed 0, eight arms).
Round 2: docs/LOOP_SCREEN2_20261123.md (seed-1 confirmations plus combinations).
Reuses the verified long-study row arrays (seed s uses rows/s{s}.npy), the
registered per-seed measurement and calibration rows, and the owned native-burn
handoff supervisor from the completed anchor studies.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime, training_runtime
from efficient_llm.warm_order.data import Rows, sha
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.loop_screen import (ALL_ARMS, ARMS, LOOP_ARMS, SPECS, STEPS, SWAP4, LoopScreen,
                                                  block_applications, lr_at, native_path, token_pass_targets,
                                                  train_passes)
from efficient_llm.warm_order.model import Config
from scripts.anchor_long_study import manifest, read, save_state, score
from scripts.anchor_long_queue import stage, supervise

OUTPUTS = Path('/mnt/local/_outputs/deep-llms_th2')
OLD = OUTPUTS/'anchor_long_20261122_a01'
ROW_PINS = ('70fc5792ed95b9ad3931a9604f29670d6769905407c06eb24a62cf8df0fe2853',  # verified long-study rows
            '46f6f3bd65b08297396799b458c91ac5c034293a364244d4e9fb02acf122e0b9')
ROUNDS = {
    1: [(0, a) for a in ARMS],
    2: [(1, 'memory'), (1, 'dense8'), (0, 'memory_var'), (0, 'loop_attnres'), (0, 'loop_attnres_var'),
        (0, 'attnres8'), (0, 'attnres_2x'), (0, 'memory_2x')],
    4: [(0, 'loop_attnres_tokvar'), (0, 'gated_tokvar'), (1, 'loop_attnres'), (1, 'attnres'), (1, 'dense4'),
        (1, 'attnres8'), (0, 'loop_attnres_2x'), (0, 'dense8_2x')],
}
BATCH = 32
CALIBRATION_EVERY = 512
DEVICE = 'cuda'  # tests switch this to 'cpu' with tiny fixtures


def inputs(seed):
    meta = manifest(seed)
    bind = meta['data_binding']
    rp = bind['prior_recipe_pin']
    if sha(rp['path']) != rp['sha256']:
        raise ValueError('Historical recipe mismatch')
    recipe = read(Path(rp['path']))['config']
    cal_file = recipe['calibration_indices']
    if sha(cal_file) != bind['input_pins'][cal_file]:
        raise ValueError('Calibration rows changed')
    row_path = OLD/'rows'/f's{seed}.npy'
    if sha(row_path) != ROW_PINS[seed]:
        raise ValueError('Training rows changed')
    ids = np.load(row_path, mmap_mode='r')
    measurement = np.array(meta['row_ids'], dtype=np.int64)
    cal = np.load(cal_file)
    return meta, Config(**meta['checkpoint_proofs']['model_config']), Rows(recipe['data']), ids, cal, measurement


def extended_ids(seed):
    """Own verified long-study row array followed by the other seed's (disjoint, same stem, both
    exclude every registered calibration/measurement row). Used only when a fit needs more rows."""
    arrays = []
    for s in (seed, 1 - seed):
        path = OLD/'rows'/f's{s}.npy'
        if sha(path) != ROW_PINS[s]:
            raise ValueError('Training rows changed')
        arrays.append(np.load(path, mmap_mode='r'))
    if np.intersect1d(np.asarray(arrays[0][:4096]), np.asarray(arrays[1][:4096])).size:
        raise ValueError('Row arrays unexpectedly overlap')
    return np.concatenate(arrays)


def rows_for(seed, ids, need):
    return ids if need <= len(ids) else extended_ids(seed)


@torch.no_grad()
def moe_trace_stats(model, rows, ids):
    """Per MoE block and application (loop pass): expert usage and the fraction of experts chosen at one
    application that are chosen again at the next for the same token (chance = top / experts)."""
    model.eval()
    model.moe_trace = {}
    chunks = 0
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        for i in range(0, len(ids), 4):
            model.token_losses(rows.batch(ids[i:i+4], DEVICE), native_path(model.arm))
            chunks += 1
    trace, model.moe_trace = model.moe_trace, None
    k, top, out = model.moe_k, model.moe_top, {}
    for identity, apps in trace.items():
        n = len(apps) // chunks
        per = [torch.cat([apps[c * n + a] for c in range(chunks)]) for a in range(n)]
        out[str(identity)] = dict(
            usage=[(torch.bincount(p.reshape(-1), minlength=k).float() / p.numel()).tolist() for p in per],
            consecutive_overlap=[float((per[a].unsqueeze(-1) == per[a + 1].unsqueeze(-2)).any(-1).float().mean())
                                 for a in range(n - 1)],
            chance=top / k)
    return out


def make_prepare(jobs):
    def prepare(base):
        record = {}
        for seed in sorted({s for s, _ in jobs}):
            meta, config, rows, ids, cal, measurement = inputs(seed)
            need = max(STEPS[a] for s, a in jobs if s == seed) * BATCH
            ids = rows_for(seed, ids, need)
            if len(ids) < need:
                raise ValueError(f'Need {need} training rows, have {len(ids)}')
            if np.intersect1d(np.asarray(ids[:need]), np.concatenate([cal, measurement])).size:
                raise ValueError('Training rows overlap evaluation rows')
            if len(measurement) != 512 or len(cal) != 256:
                raise ValueError('Unexpected evaluation row counts')
            record[str(seed)] = dict(row_sha256=ROW_PINS[seed], train_rows=need)
        write_json(base/'prepared.json', dict(seeds=record, measurement_rows=512, calibration_rows=256,
                                              config=config.__dict__, jobs=jobs))
    return prepare


def build(config, arm, device, seed):
    torch.manual_seed(seed)  # identical base initialization in every arm of a seed
    from efficient_llm.warm_order.loop_screen import make_model
    model = make_model(config, arm).to(device)
    if SPECS[arm].get('fast_attn'):  # f10: torch's fused SDPA kernels (documents still isolated when segments are given)
        import dataclasses
        from efficient_llm.dot_pool.model import Attention
        for module in model.modules():
            if isinstance(module, Attention):
                module.c = dataclasses.replace(module.c, backend='fused')
    if hasattr(model, 'generator'):  # sampled (rl) routing during training
        model.generator = torch.Generator(device=device).manual_seed(20261123 + seed)
    return model


SWEEP_DEPTHS, HALT_DEPTHS, HALT_TAUS, HALT_ROWS = (2, 3, 4, 5, 6, 8), (2, 3, 4, 5, 6, 8), (.3, .5, .7, .9), 128


@torch.no_grad()
def halting_stats(model, rows, ids, path, depths=HALT_DEPTHS, taus=HALT_TAUS):
    """Per-token halting by confidence for a DOT sandwich: each scored token reads out at the first depth (ascending)
    whose next-token max-probability reaches tau, else at the deepest depth. The state after d steps does not depend
    on later steps, so this equals reading out at step d of one deep pass. Reports NLL, mean depth and mean block
    applications (2 d + 1) per tau, next to the NLL of every fixed depth on the same rows."""
    from torch.nn import functional as F
    model.eval()
    losses, confs = {}, {}
    p = model.c.prefix
    try:
        with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
            for d in depths:
                model.test_depth = d
                L, C = [], []
                for start in range(0, len(ids), 2):
                    x = rows.batch(ids[start:start + 2], DEVICE)
                    h = model.readout_hidden(model.hidden(x, path))[:, p - 1:-1]
                    logits = F.linear(h, model.embed.weight).float()
                    L.append(F.cross_entropy(logits.transpose(1, 2), x[:, p:], reduction='none').double().cpu())
                    C.append(logits.softmax(-1).max(-1).values.cpu())
                losses[d], confs[d] = torch.cat(L), torch.cat(C)
    finally:
        model.test_depth = None
    loss = torch.stack([losses[d] for d in depths])
    conf = torch.stack([confs[d] for d in depths])
    dvec = torch.tensor(depths, dtype=torch.double)
    out = dict(rows=len(ids), fixed={str(d): float(losses[d].mean()) for d in depths})
    for tau in taus:
        reach = conf >= tau
        first = torch.where(reach.any(0), reach.long().argmax(0), torch.full(conf.shape[1:], len(depths) - 1))
        dep = dvec[first]
        out[str(tau)] = dict(nll=float(loss.gather(0, first.unsqueeze(0)).mean()), mean_depth=float(dep.mean()),
                             apps=float(2 * dep.mean() + 1),
                             depth_share={str(d): float((first == i).double().mean()) for i, d in enumerate(depths)})
    return out


@torch.no_grad()
def dot_route_stats(model, rows, ids):
    """Per step block usage, number of distinct per-token block orders and share of the most common order."""
    model.eval()
    model.trace = []
    chunks = []
    with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
        for i in range(0, len(ids), 4):
            model.trace = []
            model.token_losses(rows.batch(ids[i:i+4], DEVICE), native_path(model.arm))
            chunks.append(torch.stack(model.trace, -1).reshape(-1, len(model.trace)).cpu())
    model.trace = None
    orders = torch.cat(chunks)
    unique, counts = torch.unique(orders, dim=0, return_counts=True)
    stats = dict(usage=[torch.bincount(orders[:, s], minlength=4).div(len(orders)).tolist() for s in range(orders.shape[1])],
                 distinct_orders=int(len(unique)), modal_order=unique[counts.argmax()].tolist(),
                 modal_share=float(counts.max() / len(orders)))
    if hasattr(model, 'stop_index'):  # middles used before STOP (stopped steps are recorded as stop_index)
        depth = (orders != model.stop_index).long().cumprod(1).sum(1).float()
        stats.update(mean_depth=float(depth.mean()), depth_hist=torch.bincount(depth.long(), minlength=orders.shape[1] + 1).tolist())
    return stats


def check_parity(got, ref):
    """CPU-vs-device token-loss parity. Round-2 a01 showed FP32 rounding of up to 1.6e-4 relative on
    1-2 of 960 tokens for 17-25-sublayer AttnRes stacks; real kernel/mask bugs differ by orders more.
    Elementwise tolerance 1e-3 plus a tight 1e-4 bound on the mean absolute difference."""
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
    mean_abs = float((got.double() - ref.double()).abs().mean())
    if mean_abs > 1e-4:
        raise AssertionError(f'Mean CPU/device token-loss difference {mean_abs:.3g} exceeds 1e-4')


def qualify(base, seed, arm, config, rows, cal):
    """Actual-GPU parity against CPU at init and with perturbed new parameters; one finite update."""
    out = base/f's{seed}'/f'qualification-{arm}'
    out.mkdir(parents=True, exist_ok=False)
    gpu, cpu = build(config, arm, DEVICE, seed), build(config, arm, 'cpu', seed)
    x = rows.batch(cal[:2], 'cpu')
    path = native_path(arm)
    for perturb in (False, True):
        if perturb:
            with torch.no_grad():
                g = torch.Generator().manual_seed(7)
                for name, p in cpu.named_parameters():
                    if name == 'attnres_dyn_a':
                        continue  # keep its random init; the zero-init B carries the perturbation
                    if name in ('recipient', 't_recipient'):
                        # exercise the recipient-id path at a realistic trained gain (zero at init)
                        p.copy_(torch.randn(p.shape, generator=g) * .1)
                        continue
                    if name.startswith(('t_film', 't_lora_b', 'sh_o_cols', 'sh_w', 'sh_gate')):  # zero init: exercise them
                        p.copy_(torch.randn(p.shape, generator=g) * .02)
                        continue
                    if name.startswith(('moe_router', 'dot_router')):
                        p.copy_(torch.randn(p.shape, generator=g) * .3)  # sharp: no CPU/GPU top-k tie flips
                        continue
                    if name.startswith(('memory_', 'message_gate', 'attnres_')):
                        # AttnRes queries: trained norms were 0.1-0.6 (per-element ~0.004-0.02); std .3
                        # (norm ~8) made softmax far sharper than any trained model (round-4 a01 failure).
                        # Dynamic delta B: std .003 gives per-element query deltas ~0.017, the trained
                        # AttnRes scale; std .02 on A and B failed parity (moe/dyn a01, mean 1.9e-4).
                        scale = .003 if name == 'attnres_dyn_b' else .02 if name.startswith('attnres_') else .3
                        p.copy_(torch.randn(p.shape, generator=g) * scale)
            gpu.load_state_dict(cpu.state_dict())
        for passes in ((1, 3, 5) if arm in LOOP_ARMS else (3,)):
            cpu.passes = gpu.passes = passes
            with torch.inference_mode(), measurement_runtime('cpu'), anchor_runtime('cpu'):
                ref = cpu.token_losses(x, path)
            if getattr(cpu, 'decisions', None) is not None:  # f19 MoR: discrete routing decided once (CPU), shared
                gpu.forced_decisions = [d.to(DEVICE) for d in cpu.decisions]
            with torch.inference_mode(), measurement_runtime(DEVICE), anchor_runtime(DEVICE):
                got = gpu.token_losses(x.to(DEVICE), path).cpu()
            if hasattr(gpu, 'forced_decisions'):
                gpu.forced_decisions = None
            check_parity(got, ref)
        if SPECS[arm]['var'] == 'token':  # copy-through training path on the device
            targets = token_pass_targets(x.shape, 0, seed, 'cpu')
            cpu.token_passes, gpu.token_passes = targets, targets.to(DEVICE)
            with torch.inference_mode(), measurement_runtime('cpu'), anchor_runtime('cpu'):
                ref = cpu.token_losses(x, path)
            with torch.inference_mode(), measurement_runtime(DEVICE), anchor_runtime(DEVICE):
                got = gpu.token_losses(x.to(DEVICE), path).cpu()
            check_parity(got, ref)
            cpu.token_passes = gpu.token_passes = None
    model = build(config, arm, DEVICE, seed).train()
    opt = make_optimizer(model, arm)
    with training_runtime(DEVICE, fast_attention=bool(SPECS[arm].get('fast_attn'))), anchor_runtime(DEVICE):
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            batch = rows.batch(cal[:BATCH], DEVICE)
            loss = (model.train_loss(batch, path, 0, STEPS[arm]) if hasattr(model, 'train_loss')
                    else model.nll(batch, path).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite qualification loss')
    write_json(out/'passed.json', dict(seed=seed, arm=arm, actual_gpu=True, parity=True, finite_update=True))


def make_optimizer(model, arm):
    """AdamW without weight decay (every loop-screen arm), or a full method's own optimizer (f19) at the scaled peak."""
    if hasattr(model, 'optimizer'):
        return model.optimizer(lr_at(STEPS[arm] // 2, STEPS[arm]) * SPECS[arm].get('lr_scale', 1.))
    return torch.optim.AdamW(model.parameters(), lr=lr_at(0, STEPS[arm]), betas=(.9, .95), eps=1e-8, weight_decay=0)


def step_lr(model, arm, step, total):
    scale = SPECS[arm].get('lr_scale', 1.)  # per-arm peak-lr scale (f07)
    if hasattr(model, 'lr'):  # f19: the method's own schedule shape at peak = our peak * scale
        return model.lr(step, total, lr_at(total // 2, total) * scale)
    return lr_at(step, total) * scale


def train_objective(model, x, path, step, total):
    if hasattr(model, 'train_loss'):  # f19 full methods: the method's own objective (incl. auxiliary terms)
        return model.train_loss(x, path, step, total)
    token_losses = model.token_losses(x, path)
    loss = token_losses.mean(1).mean()  # = model.nll(x, path).mean()
    if getattr(model, 'extra_loss', None) is not None:  # DOT router signals needing per-token losses
        extra = model.extra_loss(x, token_losses.detach())
        if extra is not None:
            loss = loss + extra
    if model.aux is not None:  # balance / entropy terms
        loss = loss + model.aux
    return loss


def fit(base, seed, arm, meta, config, rows, ids, cal, measurement):
    out = base/f's{seed}'/arm
    out.mkdir(parents=True, exist_ok=False)
    ids = rows_for(seed, ids, STEPS[arm] * BATCH)
    model = build(config, arm, DEVICE, seed)
    opt = make_optimizer(model, arm)
    total = STEPS[arm]
    path = native_path(arm)
    binding = dict(rows_sha256=ROW_PINS[seed], seed=seed, arm=arm, steps=total)

    def evaluate(data, order=path, passes=None):
        model.passes = passes or SPECS[arm].get('passes', 3)
        return score(model, rows, data, {'v': dict(path=order, skip=None)}, DEVICE)['v']

    def calibration(step):
        with (out/'calibration.jsonl').open('a') as f:
            print(json.dumps(dict(step=step, nll=float(evaluate(cal).mean()),
                                  block_applications=step*block_applications(arm))), file=f)

    calibration(0)
    started = time.time()
    with training_runtime(DEVICE, fast_attention=bool(SPECS[arm].get('fast_attn'))), anchor_runtime(DEVICE):
        for step in range(total):
            for group in opt.param_groups:
                group['lr'] = step_lr(model, arm, step, total)
            model.passes = train_passes(arm, step) or SPECS[arm].get('passes', 3)
            model.train_progress = step / total  # exploration schedules (DOT stop sandwich)
            model.train()
            x = rows.batch(ids[step*BATCH:(step+1)*BATCH], DEVICE)
            if SPECS[arm]['var'] == 'token':
                model.token_passes = token_pass_targets(x.shape, step, seed, DEVICE)
            opt.zero_grad(set_to_none=True)
            rand = SPECS[arm].get('rand_depth') or (SPECS[arm].get('sandwich') or {}).get('rand_depth')
            if rand is not None:  # DOT sandwich trained at a random depth per step (docs/DOT_SYNTH_20261124.md, s10)
                model.test_depth = int(np.random.default_rng((seed, step)).integers(rand[0], rand[1] + 1))
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = train_objective(model, x, path, step, total)
            model.token_passes = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if rand is not None:
                model.test_depth = None  # every evaluation runs at the native depth unless it says otherwise
            done = step + 1
            if done % 128 == 0:
                elapsed = time.time() - started
                write_json(out/'progress.json', dict(step=done, total=total, last_batch=float(loss.detach()),
                                                     elapsed=elapsed, tokens_per_s=done*BATCH*512/elapsed))
            if done % CALIBRATION_EVERY == 0:
                calibration(done)
    train_seconds = time.time() - started
    save_state(out/'terminal.pt', model, opt, seed, arm, total, binding)
    values = {'native': evaluate(measurement)}
    halting = None
    if (SPECS[arm].get('sandwich') or {}).get('mode') == 'stop' or SPECS[arm].get('schedule') not in (None, 'dense'):
        deeper = (10, 12, 14) if (SPECS[arm].get('sandwich') or {}).get('rand_depth') else ()  # L9 depth extension
        for d in SWEEP_DEPTHS + deeper:  # NLL with more / fewer steps than trained (block applications = 2 d + 1)
            model.test_depth = d
            try:
                values[f'depth{d}'] = evaluate(measurement)
            finally:
                model.test_depth = None
        halting = halting_stats(model, rows, np.asarray(measurement[:HALT_ROWS]), path)
    if (SPECS[arm].get('sandwich') or {}).get('random_order') or SPECS[arm].get('order_eval'):
        depth = SPECS[arm]['sandwich']['depth']
        k = len(model.middles)
        for i in range(3):  # f18: the same three random middle orders for every arm (order robustness)
            model.eval_order = [int(v) for v in np.random.default_rng(1000 + i).integers(0, k, depth)]
            try:
                values[f'order{i}'] = evaluate(measurement)
            finally:
                model.eval_order = None
    if hasattr(model, 'eval_variants'):  # f19: e.g. Huginn r sweep, Ouro per-step / Q-exit, MoR top-k routing
        for name, mode in model.eval_variants():
            model.eval_mode = mode
            try:
                values[name] = evaluate(measurement)
            finally:
                model.eval_mode = {}
    if SPECS[arm]['blocks'] == 4 and not (SPECS[arm].get('dot') or SPECS[arm].get('sandwich') or SPECS[arm].get('schedule')
                                          or SPECS[arm].get('full') or SPECS[arm].get('dotrec')):
        values['swap'] = evaluate(measurement, SWAP4)
    if arm in LOOP_ARMS:
        for passes in range(1, 7):
            values[f'native_p{passes}'] = evaluate(measurement, passes=passes)
    if SPECS[arm]['loop'] == 'memory':
        model.memory_enabled = False
        values['memory_off'] = evaluate(measurement)
        model.memory_enabled = True
    extra = {}
    if SPECS[arm]['loop'] == 'gated':
        g = model.message_gate.detach().float()
        extra['message_gate'] = dict(mean=float(g.mean()), abs_mean=float(g.abs().mean()),
                                     min=float(g.min()), max=float(g.max()))
    if SPECS[arm]['mix'] == 'attnres':
        extra['attnres_query_norm'] = model.attnres_query.detach().float().norm(dim=-1).tolist()
    if SPECS[arm].get('dyn'):
        extra['attnres_dyn_b_norm'] = model.attnres_dyn_b.detach().float().norm(dim=(1, 2)).tolist()
    if SPECS[arm].get('moe'):
        extra['moe_usage'] = {str(i): getattr(model, f'moe_usage_{i}').tolist() for i in SPECS[arm]['moe']}
        extra['moe_trace'] = moe_trace_stats(model, rows, measurement[:64])
    if SPECS[arm].get('dot') or SPECS[arm].get('sandwich'):
        extra['dot_routes'] = dot_route_stats(model, rows, measurement[:64])
    if halting is not None:
        extra['halting'] = halting
    if SPECS[arm].get('full'):
        from efficient_llm.warm_order.full_methods import full_stats
        with measurement_runtime(DEVICE), anchor_runtime(DEVICE):
            extra['full_stats'] = full_stats(model, rows, measurement[:64], DEVICE)
    np.savez_compressed(out/'scores.npz', row_ids=measurement, **values)
    write_json(out/'summary.json', dict(
        binding=binding, step=total, rows=len(measurement), score_sha256=sha(out/'scores.npz'),
        nll={k: float(v.mean()) for k, v in values.items()}, train_seconds=train_seconds,
        tokens_per_s=total*BATCH*512/train_seconds, block_applications_per_token=block_applications(arm),
        parameters=sum(p.numel() for p in model.parameters()),
        non_embedding_parameters=sum(p.numel() for n, p in model.named_parameters()
                                     if not n.startswith(('embed', 'lm_head'))),  # Ouro's untied head is embedding
        **extra))


def worker(base, seed, arm, action):
    meta, config, rows, ids, cal, measurement = inputs(seed)
    if action == 'qualify':
        qualify(base, seed, arm, config, rows, cal)
    else:
        fit(base, seed, arm, meta, config, rows, ids, cal, measurement)


def make_experiment(jobs):
    def experiment(base):
        module = 'scripts.loop_screen_study'
        stage(base, 'qualify', jobs, module)
        for seed, arm in jobs:
            if read(base/f's{seed}'/f'qualification-{arm}'/'passed.json')['arm'] != arm:
                raise ValueError('Missing qualification')
        stage(base, 'fit', jobs, module)
        report, backups = {}, {}
        for seed, arm in jobs:
            out = base/f's{seed}'/arm
            result = read(out/'summary.json')
            if result['step'] != STEPS[arm] or result['rows'] != 512 or sha(out/'scores.npz') != result['score_sha256']:
                raise ValueError('Invalid terminal: '+arm)
            report[f's{seed}/{arm}'] = result
            backup = base/'backup'/f's{seed}'/arm/'terminal.pt'
            backup.parent.mkdir(parents=True)
            shutil.copyfile(out/'terminal.pt', backup)
            expected = read(out/'terminal.json')['sha256']
            if sha(backup) != expected:
                raise ValueError('Terminal backup mismatch')
            backups[str(backup)] = expected
        write_json(base/'report.json', report)
        write_json(base/'backup.json', backups)
    return experiment


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('queue', 'qualify', 'fit'))
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0, 1))
    p.add_argument('--arm', choices=ALL_ARMS)
    p.add_argument('--round', type=int, choices=tuple(ROUNDS), default=4)  # launcher passes no --round
    p.add_argument('--handoff', type=Path)
    p.add_argument('--handoff-sha256')
    a = p.parse_args()
    if a.action == 'queue':
        jobs = ROUNDS[a.round]
        try:
            supervise(a.base, make_prepare(jobs), make_experiment(jobs), a.handoff, a.handoff_sha256)
        except Exception as err:
            write_json(a.base/'failure.json', dict(reason=repr(err)))
            raise
    else:
        worker(a.base, a.seed, a.arm, a.action)
