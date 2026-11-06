"""Native SmolLM3 pruning, document isolation, and the unchanged A2 forward KL."""
from pathlib import Path
import torch
from efficient_llm.a2.model import forward_kl
from .common import SPEC_SHA

def prune(model, removed):
    layers = model.model.layers
    keep = [i for i in range(len(layers)) if i not in removed]
    flags = [bool(layer.self_attn.use_rope) for layer in layers]
    model.model.layers = torch.nn.ModuleList([layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    model.config.no_rope_layers = [int(flags[i]) for i in keep]
    model.config.layer_types = [layer.attention_type for layer in model.model.layers]
    model.config.a2_family_kept_layers = keep
    model.config.a2_family_spec_sha256 = SPEC_SHA
    # The live modules retain original layer_idx/use_rope; native reload restores map.
    return keep

def document_attention(starts, dtype=torch.float32):
    starts = starts.bool().clone()
    starts[:, 0] = True
    length = starts.shape[1]
    positions = torch.arange(length, device=starts.device).expand_as(starts)
    origins = torch.where(starts, positions, 0).cummax(dim=1).values
    position_ids = positions-origins
    segments = starts.long().cumsum(dim=1)
    allowed = (segments[:, :, None] == segments[:, None, :]) & (positions[:, :, None] >= positions[:, None, :])
    mask = torch.zeros(allowed.shape, dtype=dtype, device=starts.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[:, None], position_ids

class KDModel(torch.nn.Module):
    def __init__(self, student, chunk=128):
        super().__init__()
        self.student, self.config, self.chunk = student, student.config, chunk

    def forward(self, input_ids, teacher_hidden, teacher_weight, attention_mask=None, position_ids=None):
        hidden = self.student.model(input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, use_cache=False, return_dict=True).last_hidden_state
        return {'loss': forward_kl(hidden, self.student.lm_head.weight,
                                  teacher_hidden, teacher_weight, self.chunk)}

def save_student(model, path):
    if Path(path).exists():
        raise ValueError('Refuse checkpoint overwrite')
    state = {name: p.detach().cpu().to(torch.bfloat16) if p.is_floating_point() else p.detach().cpu()
             for name, p in model.state_dict().items()}
    model.save_pretrained(path, state_dict=state, safe_serialization=True)

def reload_student(path, device='cpu'):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                     torch_dtype=torch.float32, attn_implementation='sdpa', use_cache=False).to(device)
    keep = model.config.a2_family_kept_layers
    if len(keep)!=len(model.model.layers):
        raise ValueError('Saved layer map mismatch')
    for layer, original in zip(model.model.layers, keep, strict=True):
        layer.self_attn.layer_idx = original
    return model
