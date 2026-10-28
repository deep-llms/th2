"""Pinned native OpenSci model, physical pruning and full-vocabulary pure KL."""
from contextlib import contextmanager, nullcontext
import inspect
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from efficient_llm.diagnostic_data import sha256
from .protocol import parent_ref


def autocast(device, enabled=True):
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled and torch.device(device).type == 'cuda' else nullcontext()


def assert_native(model, *, full=False):
    if model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        raise ValueError('Tied embeddings required')
    if model.config.use_cache or model.config._attn_implementation != 'sdpa':
        raise ValueError('A2 requires native SDPA without cache')
    if any(layer.self_attn.attention_dropout != 0 for layer in model.model.layers):
        raise ValueError('Dropout must be zero at attention construction')
    if full and (len(model.model.layers) != 24 or sum(p.numel() for p in model.parameters()) != 1714377728):
        raise ValueError('Unexpected released OpenSci shape/count')
    if full:
        expected = dict(hidden_size=2048, intermediate_size=8192, num_attention_heads=32,
            num_key_value_heads=32, head_dim=64, vocab_size=50304, hidden_act='silu',
            attention_bias=True, mlp_bias=True, qk_layernorm=True, tie_word_embeddings=True,
            rope_scaling=None, rms_norm_eps=1e-5, layer_norm_eps=1e-5)
        if any(getattr(model.config, key, None) != value for key, value in expected.items()):
            raise ValueError('Parent structure differs from shared-check architecture')


def load_parent(assets, dataset, parent, device='cpu', *, teacher=False, theta=None):
    """Manifest pins Hub identities; directories contain controller-downloaded files.

    Import ONLY main's exact code, construct with branch config, then load branch
    weights using the native class. This is the offline equivalent of code_revision.
    """
    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    if transformers.__version__ != '4.57.1':
        raise RuntimeError('A2 requires isolated transformers==4.57.1; do not change the old train env')
    entry = assets[f'{dataset}/{parent}']
    for key,value in parent_ref(dataset,parent).items():
        if entry[key] != value: raise ValueError('Parent/code revision mismatch')
    weights, code = Path(entry['weights_dir']), Path(entry['code_dir'])
    for name,digest in entry['files'].items():
        if sha256(weights/name) != digest: raise ValueError(f'Asset mismatch: {name}')
    if not {'config.json','model.safetensors'} <= entry['files'].keys():
        raise ValueError('Weight/config checksums missing')
    for name,digest in entry['code_files'].items():
        if sha256(code/name) != digest: raise ValueError('Main source digest mismatch')
    if not {'configuration_opensci.py','modeling_opensci.py'} <= entry['code_files'].keys():
        raise ValueError('Code checksums missing')
    cls = get_class_from_dynamic_module('modeling_opensci.OpensciForCausalLM',str(code),local_files_only=True)
    if sha256(inspect.getfile(cls)) != entry['code_files']['modeling_opensci.py']:
        raise ValueError('Imported branch/stale code instead of pinned main')
    if sha256(inspect.getfile(cls.config_class)) != entry['code_files']['configuration_opensci.py']:
        raise ValueError('Imported config code differs from pinned main')
    config = cls.config_class.from_dict(json.loads((weights/'config.json').read_text()))
    config.attention_dropout = 0.; config.use_cache = False
    if theta is not None: config.rope_theta = theta
    model, loading = cls.from_pretrained(str(weights), config=config, torch_dtype=torch.bfloat16 if teacher else torch.float32,
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    if (set(loading['missing_keys']) - {'lm_head.weight'} or loading['unexpected_keys']
        or loading.get('mismatched_keys') or loading.get('error_msgs')):
        raise ValueError('Incomplete or incompatible checkpoint load: '+str(loading))
    model = model.to(device)
    assert_native(model,full=True)
    if teacher: model.eval().requires_grad_(False)
    return model


def prune(model, removed):
    layers = model.model.layers
    if hasattr(model.config,'a2_kept_layers'): raise ValueError('Already pruned; reload original parent')
    if len(set(removed)) != len(removed) or any(i not in range(len(layers)) for i in removed):
        raise ValueError('Invalid original layer indices')
    keep = [i for i in range(len(layers)) if i not in removed]
    if not keep: raise ValueError('Cannot remove every layer')
    model.model.layers = nn.ModuleList([layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    model.config.a2_kept_layers = keep
    if [layer.self_attn.layer_idx for layer in model.model.layers] != keep:
        raise ValueError('Original layer identities changed')
    assert_native(model)
    return keep


@contextmanager
def bypass(model, indices):
    """Independent hook reference for physical removal; no parameter mutation."""
    handles = []
    def identity(module,args,kwargs,output):
        x = args[0] if args else kwargs['hidden_states']
        return (x, *output[1:]) if isinstance(output,tuple) else x
    try:
        for i in indices:
            handles.append(model.model.layers[i].register_forward_hook(identity, with_kwargs=True))
        yield
    finally:
        for handle in handles: handle.remove()


def kl_chunk(hidden, weight, teacher_hidden, teacher_weight):
    logits = F.linear(hidden,weight).float()
    with torch.no_grad():
        teacher_log = F.linear(teacher_hidden,teacher_weight).float().log_softmax(-1)
    return F.kl_div(logits.log_softmax(-1), teacher_log, reduction='sum', log_target=True)


def forward_kl(hidden, weight, teacher_hidden, teacher_weight, chunk=128, *, recompute_head=True):
    """All positions, including final/EOS. Chunking is ONLY over positions.

    Checkpoint just the head/loss to avoid retaining T*vocab logits for backward;
    decoder activations are not recomputed. BF16 autocast is preserved by PyTorch.
    """
    if (hidden.shape != teacher_hidden.shape or hidden.ndim != 3 or weight.shape != teacher_weight.shape
        or teacher_hidden.requires_grad or teacher_weight.requires_grad or chunk < 1):
        raise ValueError('Invalid KD tensors or unfrozen teacher')
    loss = hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,hidden.shape[1],chunk):
        args = (hidden[:,start:start+chunk],weight,teacher_hidden[:,start:start+chunk],teacher_weight)
        if torch.is_grad_enabled() and recompute_head:
            term = checkpoint(kl_chunk,*args,use_reentrant=False,preserve_rng_state=False)
        else: term = kl_chunk(*args)
        loss = loss + term
    return loss/(hidden.shape[0]*hidden.shape[1])


class KDModel(nn.Module):
    """DDP must see the complete student forward, including the tied head."""
    def __init__(self,student,chunk=128):
        super().__init__(); self.student=student; self.config=student.config; self.chunk=chunk
    def forward(self,input_ids,teacher_hidden,teacher_weight):
        hidden = self.student.model(input_ids, use_cache=False, return_dict=True).last_hidden_state
        return {'loss': forward_kl(hidden,self.student.lm_head.weight,teacher_hidden,teacher_weight,self.chunk)}


@torch.no_grad()
def round_to_saved_precision(model):
    for p in model.parameters():
        if p.dtype != torch.float32: raise ValueError('Expected FP32 master weights')
        p.copy_(p.to(torch.bfloat16).float())


def save_student(model,path):
    """Save BF16, preserving live FP32 tensors. Native HF handles tied parameters."""
    path=Path(path)
    if path.exists(): raise ValueError('Refuse checkpoint overwrite')
    model.register_for_auto_class('AutoModelForCausalLM')
    model.config.register_for_auto_class('AutoConfig')
    state={k:v.detach().cpu().to(torch.bfloat16) if v.is_floating_point() else v.detach().cpu()
           for k,v in model.state_dict().items()}
    model.save_pretrained(path,state_dict=state,safe_serialization=True)


def reload_student(path,device='cpu'):
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa',attention_dropout=0.,use_cache=False).to(device)
    keep=model.config.a2_kept_layers
    if len(keep)!=len(model.model.layers): raise ValueError('Saved layer map mismatch')
    for layer,original in zip(model.model.layers,keep,strict=True): layer.self_attn.layer_idx=original
    assert_native(model)
    return model
