"""Opt-in P7 throughput candidates; never imported by ordinary training.

The LSE candidate partitions ONE joint softmax, including its derivatives.
It does not average independently normalized attention outputs. Production
keeps its validated interleaved implementation until CUDA gates and timing pass.
"""
from contextlib import contextmanager


def attention_lse(q, k, v, layout, scaling, kernel):
    b, heads, length, dim = q.shape
    def flatten(x):
        return x.to(v.dtype).transpose(1, 2).reshape(b*length, x.shape[1], dim).contiguous()
    cu, maximum = layout
    output, lse = kernel(*map(flatten, (q, k, v)), cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=maximum, max_seqlen_k=maximum, causal=True,
        softmax_scale=scaling, return_lse=True)
    if lse.shape != (heads, b*length):
        raise ValueError(f'Unexpected pinned FA4 LSE layout: {lse.shape}')
    return output.reshape(b, length, heads, dim), lse.T.reshape(b, length, heads)


def merge_attention(native, proxy, native_lse, proxy_lse):
    import torch
    # Both branches include each query's own token, so LSEs are finite.
    # Keep LSE live: its adjoints are essential for the joint-softmax gradient.
    with torch.autocast(native.device.type, enabled=False):
        split = native.shape[2]-proxy.shape[2]
        weight = torch.sigmoid(native_lse[:, :, split:].float()-proxy_lse.float())[..., None]
        selected = weight*native[:, :, split:].float()+(1-weight)*proxy.float()
        return torch.cat((native[:, :, :split], selected.to(native.dtype)), dim=2)


def memory_attention(self, layer, u, head, prediction, mask, rotary, plan):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    from deep_kv.proxy_memory import proxy_mass
    if not self.proxy_enabled:
        return self.attention(layer, u, None, mask, rotary)
    a = layer.self_attn
    shape = (*u.shape[:-1], -1, a.head_dim)
    q = a.q_norm(a.q_proj(u).view(shape)).transpose(1, 2)
    k = a.k_norm(a.k_proj(u).view(shape)).transpose(1, 2)
    v = a.v_proj(u).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *rotary)
    kp, vp = head.entries(prediction, rotary, v[:, -2:])
    qp = q[:, -4:]
    if self._proxy_mass is not None:
        self._proxy_mass[str(a.layer_idx+1)] = proxy_mass(qp, k[:, -2:], kp, plan.documents, a.scaling)
    native, native_lse = attention_lse(q, k, v, mask, a.scaling, self.fa4_kernel)
    proxy, proxy_lse = attention_lse(qp, kp, vp, mask, a.scaling, self.fa4_kernel)
    if self._fa4_observer is not None:
        self._fa4_observer(q, k, v, mask)
        self._fa4_observer(qp, kp, vp, mask)
    return merge_attention(native, proxy, native_lse, proxy_lse)


@contextmanager
def implementation(mode, backend='inductor'):
    import torch
    import deep_kv.proxy_estimators as estimators
    from deep_kv.proxy import ProxyModel
    if mode in ('fused_aux', 'entries', 'kv'):
        from scripts.p7_fusion_trial import refinement
        with implementation('both', backend=backend), refinement(mode, backend=backend):
            yield
        return
    if mode not in ('eager', 'cosine', 'lse', 'both'):
        raise ValueError(mode)
    original_attention = ProxyModel.memory_attention
    original_cosine = estimators.cosine_loss
    global merge_attention
    original_merge = merge_attention
    try:
        if mode in ('cosine', 'both'):
            # The original function preserves live targets when target_grad=True.
            estimators.cosine_loss = torch.compile(original_cosine, backend=backend, fullgraph=True, dynamic=False)
        if mode in ('lse', 'both'):
            merge_attention = torch.compile(original_merge, backend=backend, fullgraph=True, dynamic=False)
            def candidate(self, *args, **kwargs):
                if self.arm != 'P7-simple-targetgrad' or self.attention_backend != 'fa4':
                    return original_attention(self, *args, **kwargs)
                return memory_attention(self, *args, **kwargs)
            ProxyModel.memory_attention = candidate
        yield
    finally:
        ProxyModel.memory_attention = original_attention
        estimators.cosine_loss = original_cosine
        merge_attention = original_merge


def train(mode, arguments):
    import hashlib
    from pathlib import Path
    import runpy
    import sys
    from transformers import TrainerCallback
    import deep_kv.proxy_training as training
    from scripts.p7_speed_study import write
    class Audit(TrainerCallback):
        def __init__(self): self.hashes=[]; self.graphs={}
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in (20,49):
                from torch._dynamo.utils import counters
                self.graphs[str(state.global_step)]=dict(counters['stats'])
        def on_train_end(self, args, state, control, **kwargs):
            write(Path(args.output_dir)/f'speed-trial-rank{args.process_index}.json',
                dict(mode=mode, data_hashes=self.hashes, graphs=self.graphs))
    original_init, original_step=training.ProxyTrainer.__init__,training.ProxyTrainer.training_step
    def initialize(self, *args, **kwargs):
        original_init(self,*args,**kwargs)
        self.speed_audit=Audit(); self.add_callback(self.speed_audit)
    def step(self, model, inputs, *args, **kwargs):
        # Outside measured updates20–49. Verify the identical data order at
        # both ends; no new tokenization/sampling or changes to the dataloader.
        if self.state.global_step in (0,49):
            h=hashlib.sha256()
            for key,value in sorted(inputs.items()):
                h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
            self.speed_audit.hashes.append(h.hexdigest())
        return original_step(self,model,inputs,*args,**kwargs)
    training.ProxyTrainer.__init__,training.ProxyTrainer.training_step=initialize,step
    with implementation('eager' if mode == 'configured' else mode):
        sys.argv = ['scripts.profile_proxy_training', *arguments]
        runpy.run_module('scripts.profile_proxy_training', run_name='__main__')


def gate(args):
    import gc
    import hashlib
    from pathlib import Path
    import torch
    from scripts.p7_speed_study import read, write
    from scripts.p6_compile_trial import compare
    from scripts.benchmark_document_training import compare as compare_model
    from scripts.check_trained_attention import acceptable
    from scripts.evaluate_recurrent_p6 import evaluation_runtime
    from scripts.proxy_speed_validation import deterministic_fa4
    from deep_kv.packing import isolated_data_collator
    from deep_kv.proxy_training import ProxyTrainer
    from eval.models import load_checkpoint, file_hash
    saved = read(Path(args.checkpoint).parent/'train_config.json')
    _, accelerator = evaluation_runtime(saved['training'], args.output)
    if accelerator.num_processes != 8 or accelerator.device.type != 'cuda':
        raise ValueError('Eight CUDA ranks required')
    rank = accelerator.process_index
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = Path(args.rows).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.rows_sha256:
        raise ValueError('Frozen validation rows changed')
    length, micro = (64, 2) if rank < 4 else (2048, 16)
    modes = ('both', 'fused_aux', 'entries', 'kv') if args.refinement else ('eager', 'cosine', 'lse', 'both')
    candidate = modes[rank % 4]
    rows = [{k: v[:length] for k, v in row['data'].items()} for row in read(args.rows)[:micro]]
    ctx = ProxyTrainer.context({k: v.to(accelerator.device) for k, v in isolated_data_collator(rows).items()})
    captures = []
    for mode in (modes[0], candidate):
        with implementation(mode), deterministic_fa4(True):
            adapter, _, metadata = load_checkpoint(args.checkpoint, accelerator.device, attention_backend='fa4')
            if metadata['arm'] != 'P7-simple-targetgrad' or metadata['step'] != 2500 or metadata['checkpoint_sha256'] != args.checkpoint_sha256:
                raise ValueError('Wrong source checkpoint')
            m = adapter.wrapped.train()
            m.checkpoint_layers = m.checkpoint_aux = m.checkpoint_lm = rank < 4
            outputs = {}
            def capture_hidden(module, inputs, value):
                outputs['hidden'] = value.detach().float().cpu()
                outputs['logits'] = m.backbone.lm_head(value[:, ::128]).detach().float().cpu()
            handle = m.backbone.model.norm.register_forward_hook(capture_hidden)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = m(ctx, collect_target_statistics=True)
                loss = out['lm_sum']/out['lm_count'] + .1*out['aux_sum']/out['aux_count']
            loss.backward(); handle.remove()
            gradients = {k: p.grad.detach().float().cpu().clone() for k, p in m.named_parameters() if p.grad is not None}
            if len(gradients) != len(list(m.named_parameters())) or not all(torch.isfinite(g).all() for g in gradients.values()):
                raise ValueError('Missing or nonfinite gradients')
            outputs.update({k: v.detach().cpu().clone() for k, v in out.items()})
            # CPU tests additionally check isolated LM/aux routing and an AdamW
            # update. This full-shape CUDA gate compares all parameter gradients.
            m.update_statistics(out['center_sums'], out['center_squares'], out['center_counts'])
            buffers = {k: v.detach().cpu().clone() for k, v in m.named_buffers()}
            captures.append(dict(**outputs, grads=gradients, buffers=buffers,
                loss=float(loss.detach()), targets=int(out['lm_count'])))
        del m, adapter, out, loss, gradients, outputs, buffers, handle
        gc.collect(); torch.cuda.empty_cache()
    a, b = captures
    comparisons = {'model': compare_model(a, b, False)}
    for group, prefix in (('proxy', 'heads.'), ('backbone', 'backbone.')):
        comparisons[group] = compare_model({**a, 'grads': {k: v for k, v in a['grads'].items() if k.startswith(prefix)}},
            {**b, 'grads': {k: v for k, v in b['grads'].items() if k.startswith(prefix)}}, False)
    # Existing FA4 model acceptance thresholds; no post-hoc widening.
    passed = all(acceptable(row) for row in comparisons.values())
    buffers = compare(a['buffers'], b['buffers'], relative=.02, peak=.02, exact=candidate in (modes[0], 'cosine', 'fused_aux'))
    passed = passed and not buffers['failures']
    exact = None
    if candidate in (modes[0], 'cosine', 'fused_aux'):
        keys = ('hidden', 'logits', 'lm_sum', 'lm_count', 'center_sums', 'center_squares', 'center_counts')
        exact = compare({k: a[k] for k in keys}, {k: b[k] for k in keys}, exact=True)
        passed = passed and not exact['failures']
    write(Path(args.output)/f'rank-{rank}.json', dict(status='passed' if passed else 'failed',
        mode=candidate, sequence_length=length, micro_batch=micro, checkpointing=rank < 4,
        comparisons=comparisons, buffers=buffers, exact_forward=exact, checkpoint=metadata))
    del captures, a, b
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        reports = [read(Path(args.output)/f'rank-{i}.json') for i in range(8)]
        accepted = [mode for mode in modes[1:]
                    if all(r['status']=='passed' for r in reports if r['mode']==mode)]
        if not all(r['status']=='passed' for r in reports if r['mode']==modes[0]):
            raise ValueError('Same-implementation numerical control failed')
        if file_hash(Path(args.checkpoint)/'model.safetensors') != args.checkpoint_sha256:
            raise ValueError('Source checkpoint changed')
        write(Path(args.output)/'summary.json', dict(status='passed' if accepted else 'failed',
            accepted=accepted, reports=reports, checkpoint_unchanged=True,
            thresholds=dict(loss_absolute=.01, gradient_relative_l2=.03, output_relative_l2=.02,
                            buffer_relative_l2=.02, buffer_peak_scaled=.02)))
    accelerator.wait_for_everyone()
    if read(Path(args.output)/'summary.json')['status'] != 'passed':
        raise ValueError('No numerical candidate accepted')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'gate'))
    p.add_argument('--implementation', choices=('eager', 'cosine', 'lse', 'both', 'configured', 'fused_aux', 'entries', 'kv'))
    p.add_argument('--refinement', action='store_true')
    p.add_argument('--checkpoint'); p.add_argument('--checkpoint-sha256')
    p.add_argument('--rows'); p.add_argument('--rows-sha256'); p.add_argument('--output')
    args, rest = p.parse_known_args()
    if rest[:1] == ['--']: rest = rest[1:]
    if args.mode == 'train': train(args.implementation, rest)
    else:
        if rest: p.error('Unexpected gate arguments')
        gate(args)
