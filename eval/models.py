"""Inference interface for the exact trained DeepKV/proxy computation.

Training intentionally returns chunked loss statistics. This adapter exposes
the same final hidden states through the existing vocabulary projection.
"""
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, PreTrainedModel, Qwen3Config
from transformers.modeling_outputs import CausalLMOutput

from deep_kv.model import Context, DeepKV
from deep_kv.proxy import ProxyModel, ProxySettings


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


class EvaluationModel(PreTrainedModel):
    """Likelihood evaluation only; no training, KV cache, or generation API."""
    config_class = Qwen3Config
    _supports_sdpa = True

    def __init__(self, wrapped, max_length, autocast_dtype=None):
        config = copy.deepcopy(wrapped.backbone.config)
        config.max_position_embeddings = min(max_length, config.max_position_embeddings)
        super().__init__(config)
        self.wrapped = wrapped
        self.autocast_dtype = autocast_dtype
        self.eval()

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None,
                segments=None, use_cache=False, past_key_values=None, **kwargs):
        if self.training or past_key_values is not None or use_cache:
            raise ValueError('Evaluation adapter requires eval mode and no KV cache')
        if kwargs:
            raise ValueError(f'Unsupported inference arguments: {sorted(kwargs)}')
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or not 0 < input_ids.shape[1] <= self.config.max_position_embeddings:
            raise ValueError('Expected nonempty [batch, length] input within the trained context limit')
        batch, length = input_ids.shape
        if attention_mask is not None and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('attention_mask must contain only zero or one')
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if valid.shape != input_ids.shape or not bool(valid.any(-1).all()):
            raise ValueError('Every evaluation row needs at least one valid token')
        valid_starts = valid & F.pad(~valid[:, :-1], (1, 0), value=True)
        if bool((valid_starts.sum(-1) != 1).any()):
            raise ValueError('Padding must be outside a contiguous span of valid tokens')
        # Each benchmark request is one document. Never interpret literal EOS
        # inside a prompt as a boundary; packed callers supply explicit segments.
        docs = torch.zeros_like(input_ids) if segments is None else segments.clone()
        if docs.shape != input_ids.shape or docs.dtype not in (torch.int32, torch.int64) or bool((docs < 0).any()):
            raise ValueError('Expected nonnegative segment IDs matching input IDs')
        docs = docs.masked_fill(~valid, int(docs.max()) + 1)
        # EMS requires complete chunks; Context requires at least two positions.
        chunk = (self.wrapped.settings.chunk_size if isinstance(self.wrapped, ProxyModel)
                 and (self.wrapped.family == 'P3' or self.wrapped.arm == 'P7-ems') else 1)
        padded_length = ((max(2, length) + chunk - 1) // chunk) * chunk
        extra = padded_length - length
        ids = F.pad(input_ids, (0, extra), value=0)
        docs = F.pad(docs, (0, extra), value=int(docs.max()) + 1)
        offsets = torch.arange(padded_length, device=ids.device).expand(batch, -1)
        starts = torch.ones_like(docs, dtype=torch.bool)
        starts[:, 1:] = docs[:, 1:] != docs[:, :-1]
        # A repeated document ID after another document would make dense SDPA
        # reconnect fragments while FA4 treats them separately. Reject it.
        for row in range(batch):
            document_ids = docs[row, :length][starts[row, :length] & valid[row]]
            if document_ids.unique().numel() != document_ids.numel():
                raise ValueError('Each document ID must occupy one contiguous segment per row')
        positions = offsets - offsets.masked_fill(~starts, 0).cummax(-1).values
        if position_ids is not None and not torch.equal(position_ids, positions[:, :length]):
            raise ValueError('Position IDs must reset at each document/padding boundary')
        # Padding is a separate dummy document. The unchanged proxy path can
        # treat all positions as packed; real queries cannot attend to padding.
        context = Context(ids, torch.ones_like(ids, dtype=torch.bool), positions, docs)
        if self.wrapped.causal_attention:
            if segments is not None or attention_mask is not None and not bool(valid.all()):
                raise ValueError('Legacy causal model does not support segmented or padded requests')
            context.segments = None
        with torch.autocast(ids.device.type, dtype=self.autocast_dtype,
                            enabled=self.autocast_dtype is not None):
            hidden = self.wrapped.hidden_states(context)[0]
            logits = self.wrapped.backbone.lm_head(hidden[:, :length])
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError('Labels must match input IDs')
            eligible = valid[:, :-1] & valid[:, 1:] & (docs[:, :length-1] == docs[:, 1:length])
            targets = labels[:, 1:].masked_fill(~eligible, -100)
            if not bool(targets.ne(-100).any()):
                raise ValueError('No scored next-token targets')
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    def generate(self, *args, **kwargs):
        raise NotImplementedError('Custom checkpoints currently support likelihood tasks, not generation')


def load_checkpoint(checkpoint, device='cpu', dtype=None, attention_backend=None):
    """Return model, tokenizer source, and provenance; strict-load custom states."""
    checkpoint = Path(checkpoint).resolve()
    recipe_path = next((p / 'train_config.json' for p in (checkpoint, checkpoint.parent)
                        if (p / 'train_config.json').is_file()), None)
    weights = checkpoint / 'model.safetensors'
    if recipe_path is None:
        if weights.is_file():
            with safe_open(weights, framework='pt') as state:
                if any(key.startswith('backbone.') for key in state.keys()):
                    raise ValueError('Custom checkpoint is missing its train_config.json recipe')
        if attention_backend == 'fa4':
            raise ValueError('FA4 selection is supported only for custom proxy checkpoints')
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), dtype=dtype or 'auto', attn_implementation=attention_backend or 'sdpa',
            local_files_only=True).to(device).eval()
        return model, str(checkpoint), dict(checkpoint=str(checkpoint), kind='huggingface',
                                           attention_backend=attention_backend or 'sdpa')
    saved = json.loads(recipe_path.read_text())
    pilot = saved['pilot']
    if saved['model_config']['model_type'] != 'qwen3':
        raise ValueError('Custom checkpoint must be a Qwen3 model')
    config = Qwen3Config.from_dict(saved['model_config'])
    config._attn_implementation = 'sdpa'
    config.use_cache = False
    backend = attention_backend or pilot.get('attention_backend', 'sdpa')
    if dtype is None and saved['training'].get('bf16'):
        dtype = torch.bfloat16
    if backend == 'fa4' and (torch.device(device).type != 'cuda' or dtype != torch.bfloat16):
        raise ValueError('FA4 evaluation requires CUDA BF16; choose SDPA explicitly for CPU checks')
    options = {key: pilot[key] for key in ('consumer', 'deep_target', 'lm_chunk', 'causal_attention')}
    options.update(checkpoint_layers=False, checkpoint_lm=False, checkpoint_aux=False,
                   seed=saved['training']['seed'])
    if pilot.get('proxy_screen', False):
        settings = ProxySettings(**{f.name: pilot['proxy_' + f.name] for f in fields(ProxySettings)
                                    if 'proxy_' + f.name in pilot})
        model = ProxyModel.from_scratch(config, pilot['arm'], proxy_settings=settings,
            sequence_length=saved['data']['block_size'], attention_backend=backend, **options)
    else:
        if backend != 'sdpa':
            raise ValueError('Legacy DeepKV checkpoints require SDPA')
        model = DeepKV.from_scratch(config, pilot['arm'], **options)
    state = load_file(str(weights), device='cpu')
    # Accept only genuinely tied aliases; missing proxy weights/buffers fail.
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        present = [name for name in names if name in state]
        if present:
            source = state[present[0]]
            if any(not torch.equal(source, state[name]) for name in present[1:]):
                raise ValueError(f'Checkpoint tied weights disagree: {names}')
            for name in names:
                state.setdefault(name, source)
    model.load_state_dict(state, strict=True)
    adapter = EvaluationModel(model, saved['data']['block_size'], dtype).to(device).eval()
    tokenizer_source = (str(checkpoint) if (checkpoint / 'tokenizer_config.json').is_file()
                        else saved['model']['tokenizer_name'])
    trainer_state = checkpoint / 'trainer_state.json'
    metadata = dict(kind='custom', arm=pilot['arm'], checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(weights), recipe=str(recipe_path), recipe_sha256=file_hash(recipe_path),
        step=json.loads(trainer_state.read_text())['global_step'] if trainer_state.is_file() else None,
        attention_backend=backend, trained_attention_backend=pilot.get('attention_backend', 'sdpa'),
        autocast_dtype=str(dtype), master_dtype='torch.float32', context_length=saved['data']['block_size'],
        boundary_policy='explicit segments; otherwise one document per request', positions='reset_per_document')
    return adapter, tokenizer_source, metadata
