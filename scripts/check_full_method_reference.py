"""Parity of our full-method ports against the official code (docs/FULL_LOOPED_METHODS_20261126.md).

Tiny configs on CPU in FP32: every parameter of the official model is randomized (norms, biases and routers
included), copied into our port, and outputs are compared on the same input.
  huginn: HF raven_modeling_minimal.py (huginn-0125). Logits at r = 1, 3, 5 with the same s0, and
          gradients of a truncated-backprop step (n no-grad + k grad iterations).
  huginn_optim: ELLISAdam (nebel config) against GH recpre/optim.py over six steps.
  ouro:   HF modeling_ouro.py (Ouro-1.4B). Per-step logits and exit-gate logits over T steps.
  mor:    official repo (MoRLlamaForCausalLM + middle_cycle sharing + expert routers). Training-mode
          logits and the per-step BCE sampling losses (top-k routing).

Usage (each in an environment with the reference's transformers version; untrusted reference code is
imported from an explicit path, run with python -I):
  python -I scripts/check_full_method_reference.py huginn <dir with huginn_ref/> --out result.json
  python -I scripts/check_full_method_reference.py huginn_optim <GH recpre/ dir> --out result.json
  python -I scripts/check_full_method_reference.py ouro <dir with ouro_ref/> --out result.json
  python -I scripts/check_full_method_reference.py mor <official repo root> --out result.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from efficient_llm.warm_order.full_methods import make_full_model  # noqa: E402
from efficient_llm.warm_order.model import Config  # noqa: E402


def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    seen = set()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim == 1 and ('norm' in name or 'ln' in name):
                p.copy_(1 + .2 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(.08 * torch.randn(p.shape, generator=g))


def load(ours, mapping):
    state = ours.state_dict()
    missing = set(state) - set(mapping)
    if missing:
        raise ValueError(f'Unmapped port parameters: {sorted(missing)}')
    with torch.no_grad():
        for name, tensor in mapping.items():
            if state[name].shape != tensor.shape:
                raise ValueError(f'{name}: {tuple(state[name].shape)} vs {tuple(tensor.shape)}')
            state[name].copy_(tensor)


def maxdiff(a, b):
    return float((a.double() - b.double()).abs().max())


def huginn(ref_dir):
    sys.path.insert(0, ref_dir)
    from huginn_ref.raven_config_minimal import RavenConfig
    from huginn_ref.raven_modeling_minimal import RavenForCausalLM
    torch.manual_seed(0)
    cfg = RavenConfig(n_embd=64, n_heads=4, n_layers=4, block_size=64, vocab_size=256, padding_multiple=256,
                      intermediate_size=96, n_layers_in_prelude=1, n_layers_in_recurrent_block=2,
                      n_layers_in_coda=1, mean_recurrence=3, mean_backprop_depth=2, tie_embeddings=True,
                      rope_base=50000)
    ref = RavenForCausalLM(cfg).float().eval()
    randomize(ref, 1)
    c = Config(d=64, heads=4, kv_heads=4, head_dim=16, ff=96, vocab=256, prefix=1, length=64)
    spec = dict(full=dict(kind='huginn', prelude=1, core=2, coda=1, mean_recurrence=3, backprop=2))
    ours = make_full_model(c, 'huginn_parity', spec).eval()
    rs = ref.state_dict()
    mapping = {}
    for k, v in rs.items():
        if k in ('freqs_cis', 'lm_head.weight'):
            continue
        k2 = k.replace('transformer.wte.', 'embed.').replace('transformer.core_block.', 'core.').replace('transformer.', '')
        mapping[k2] = v
    load(ours, mapping)
    x = torch.randint(0, 256, (2, 24))
    torch.manual_seed(3)
    s0 = ref.initialize_state(torch.zeros(2, 24, 64))
    ours.state_override = s0
    out = {}
    with torch.no_grad():
        for r in (1, 3, 5):
            a = ref(x, input_states=s0, num_steps=r).logits
            ours.eval_mode = dict(steps=r)
            b = F.linear(ours.readout_hidden(ours.hidden(x)), ours.embed.weight)
            out[f'logits_r{r}'] = maxdiff(a, b)
    ours.eval_mode = {}
    # truncated backprop: n = 2 no-grad + k = 2 grad iterations, CE on all positions
    y = torch.randint(0, 256, (2, 24))
    ref.train()
    ref.zero_grad()
    la = ref(x, input_states=s0, num_steps=(torch.tensor(2), torch.tensor(2)), labels=y).loss
    la.backward()
    ours.train()
    ours.zero_grad()
    ours.train_steps = (2, 2)
    logits = F.linear(ours.readout_hidden(ours.hidden(x)), ours.embed.weight).float()
    lb = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    lb.backward()
    out['train_loss'] = maxdiff(la.detach(), lb.detach())
    ref_grads = {k.replace('transformer.wte.', 'embed.').replace('transformer.core_block.', 'core.').replace('transformer.', ''): p.grad
                 for k, p in ref.named_parameters() if p.grad is not None}
    out['grad_max'] = max(maxdiff(ref_grads[n], p.grad) for n, p in ours.named_parameters() if n in ref_grads)
    out['grad_params'] = len(ref_grads)
    return out


def huginn_optim(recpre_dir):
    """ELLISAdam (nebel config) vs GH recpre/optim.py, 6 steps on random gradients with a schedule including lr 0.
    optim.py imports .soap (absent from the copy, unused by ELLISAdam): loaded in a temp package with a stub."""
    import importlib
    import shutil
    import tempfile
    from efficient_llm.warm_order.full_methods import ELLISAdam
    tmp = Path(tempfile.mkdtemp())
    (tmp/'gh_optim').mkdir()
    shutil.copy(Path(recpre_dir)/'optim.py', tmp/'gh_optim'/'optim.py')
    (tmp/'gh_optim'/'__init__.py').write_text('')
    (tmp/'gh_optim'/'soap.py').write_text('class SOAP:\n    pass\n')
    sys.path.insert(0, str(tmp))
    ref_cls = importlib.import_module('gh_optim.optim').ELLISAdam
    g = torch.Generator().manual_seed(0)
    shapes = [(8, 6), (6,), (3, 4, 5)]
    init = [torch.randn(s, generator=g) for s in shapes]
    pa = [torch.nn.Parameter(t.clone()) for t in init]
    pb = [torch.nn.Parameter(t.clone()) for t in init]
    peak, wd = 3e-3, 4e-5
    ref = ref_cls([dict(params=pa[::2], weight_decay=wd), dict(params=pa[1::2], weight_decay=0.)], lr=peak,
                  betas=(.9, .95), weight_decay=wd, update_clipping=True, atan_adam=True, running_init=True,
                  tensor_wise_finite_check=False, tensor_wise_gradient_normalization=False, decouple_wd=True)
    ours = ELLISAdam([dict(params=pb[::2], weight_decay=wd), dict(params=pb[1::2], weight_decay=0.)], lr=peak)
    worst = 0.
    for step, lr in enumerate([0., peak / 2, peak, peak, peak / 3, 0.]):
        grads = [torch.randn(s, generator=g) * (10 if step == 3 else 1) for s in shapes]  # step 3: clipping active
        for opt, ps in ((ref, pa), (ours, pb)):
            for p, gr in zip(ps, grads):
                p.grad = gr.clone()
            for group in opt.param_groups:
                group['lr'] = torch.tensor(lr) if opt is ref else lr
            opt.step()
        worst = max(worst, max(maxdiff(a, b) for a, b in zip(pa, pb)))
    return dict(param_max_after_6_steps=worst, moved=maxdiff(pa[0], init[0]))


def ouro(ref_dir):
    sys.path.insert(0, ref_dir)
    from ouro_ref.configuration_ouro import OuroConfig
    from ouro_ref.modeling_ouro import OuroForCausalLM
    torch.manual_seed(0)
    T = 3
    cfg = OuroConfig(vocab_size=101, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                     rope_theta=10000.0, total_ut_steps=T, early_exit_threshold=1.0, tie_word_embeddings=False,
                     use_cache=False, rms_norm_eps=1e-6)
    cfg._attn_implementation = 'eager'
    ref = OuroForCausalLM(cfg).eval()
    randomize(ref, 2)
    c = Config(d=64, heads=4, kv_heads=2, head_dim=16, ff=96, vocab=101, prefix=1, length=64)
    ours = make_full_model(c, 'ouro_parity', dict(full=dict(kind='ouro', layers=2, steps=T))).eval()
    mapping = {k.replace('model.embed_tokens.', 'embed.').replace('model.', ''): v for k, v in ref.state_dict().items()}
    load(ours, mapping)
    x = torch.randint(0, 101, (2, 20))
    with torch.no_grad():
        _, hs, gates = ref.model(input_ids=x, use_cache=False)
        hs2, gates2 = ours.states(x, T)
        out = {f'logits_step{t + 1}': maxdiff(ref.lm_head(hs[t]), F.linear(hs2[t], ours.lm_head.weight)) for t in range(T)}
        out.update({f'gate_step{t + 1}': maxdiff(gates[t].squeeze(-1), gates2[t]) for t in range(T)})
    return out


def mor(ref_dir):
    sys.path.insert(0, ref_dir)
    from omegaconf import OmegaConf
    from transformers import LlamaConfig
    from model.mor_model.modeling_llama import MoRLlamaForCausalLM
    from model.sharing_strategy.llama import sharing_strategy
    torch.manual_seed(0)
    nr, base = 3, 2
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=1 + nr * base + 1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
                         rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=True, initializer_range=0.02,
                         hidden_act='silu', use_cache=False)
    cfg = OmegaConf.create({
        'precision': 'fp32', 'num_warmup_steps': 0, 'gradient_accumulation_steps': 1,
        'recursive': {'enable': True, 'base_depth': None, 'num_recursion': nr, 'sharing': 'middle_cycle',
                      'ln_share': True, 'initialization': 'random'},
        'kv_sharing': {'enable': False},
        'mor': {'enable': True, 'type': 'expert', 'capacity': ','.join(str((nr - i) / nr) for i in range(nr)),
                'rand_router': False, 'router_type': 'linear', 'z_loss': False, 'z_coeff': 0.001, 'temp': 1.0,
                'expert': {'cap_warmup_step': 0, 'router_func': 'sigmoid', 'alpha': 0.1, 'sampling': 'aux_loss',
                           'include_first': True, 'coeff': 0.001, 'gating': 'weighted'}}})
    ref = MoRLlamaForCausalLM._from_config(config, attn_implementation='eager', torch_dtype=torch.float32)
    ref, _ = sharing_strategy(cfg, ref)
    ref.transform_layer_to_mor_expert(cfg)
    randomize(ref, 3)
    L = ref.model.layers
    c = Config(d=64, heads=4, kv_heads=2, head_dim=16, ff=96, vocab=128, prefix=1, length=64)
    ours = make_full_model(c, 'mor_parity', dict(full=dict(kind='mor', base_depth=base, recursions=nr)))

    def layer(prefix, module):
        return {f'{prefix}.{k}': v for k, v in module.state_dict().items()}

    mapping = {'embed.weight': ref.model.embed_tokens.weight, 'norm.weight': ref.model.norm.weight}
    mapping.update(layer('first', L[0]))
    mapping.update(layer('last', L[-1]))
    for j in range(base):
        mapping.update(layer(f'middle.{j}', L[1].block[j]))
    for r in range(nr):
        mapping[f'routers.{r}.weight'] = L[1 + r].mor_router.router.weight
    load(ours, mapping)
    x = torch.randint(0, 128, (2, 18))
    out = {}
    for mode in ('eval', 'train'):  # the official code routes top-k in both modes
        getattr(ref, mode)()
        ours.train()  # our top-k path (training mode); eval mode is the causal rule
        with torch.no_grad():
            r = ref(input_ids=x, labels=x, use_cache=False, return_dict=True)
            h = ours.readout_hidden(ours.hidden(x))
            out[f'logits_{mode}'] = maxdiff(r.logits, F.linear(h, ours.embed.weight))
            if mode == 'train':
                out['sampling_loss'] = maxdiff(r.sampling_loss, sum(ours.aux_terms))
    # the causal rule with the top-k decisions forced must equal the top-k path for selected tokens
    ours.train()
    with torch.no_grad():
        ours.hidden(x)
        local = ours.decisions
        prev, masks = None, []
        for r_, loc in enumerate(local):
            sel = loc[..., 0] if prev is None else prev.gather(1, loc[..., 0])
            m = torch.zeros(x.shape, dtype=torch.bool).scatter_(1, sel, True)
            masks.append(m)
            prev = sel
        a = ours.readout_hidden(ours.hidden(x))
        ours.eval()
        ours.forced_decisions = masks
        b = ours.readout_hidden(ours.hidden(x))
        ours.forced_decisions = None
    out['causal_rule_vs_topk_forced'] = maxdiff(a, b)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('method', choices=('huginn', 'huginn_optim', 'ouro', 'mor'))
    p.add_argument('reference')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    torch.set_default_dtype(torch.float32)
    result = dict(method=a.method, torch=torch.__version__, **globals()[a.method](a.reference))
    try:
        import transformers
        result['transformers'] = transformers.__version__
    except ImportError:
        pass
    print(json.dumps(result, indent=1))
    if a.out:
        a.out.write_text(json.dumps(result, indent=1) + '\n')
