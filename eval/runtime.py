"""Shared evaluation-only utilities; never imported by pretraining."""
import os
from pathlib import Path
import re


def offline():
    # Set before importing datasets/transformers/lm_eval, not just before loading.
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK'):
        os.environ[key] = '1'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def languages(value):
    values = value.split(',') if isinstance(value, str) else list(value)
    if not values or len(set(values)) != len(values) or any(
            not re.fullmatch('[a-z]{2,3}', lang) for lang in values):
        raise ValueError('Select unique language codes, e.g. en or en,vi,zh')
    return values


def load_checkpoint(checkpoint, tokenizer_path=None, device='cpu', precision='fp32'):
    offline()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import capacity_allocation  # noqa: F401: register both custom HF classes
    checkpoint = Path(checkpoint).resolve(strict=True)
    tokenizer_path = Path(tokenizer_path or checkpoint).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError('A local fast tokenizer with EOS is required')
    tokenizer.model_max_length = 10**30
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    if precision not in ('fp32', 'bf16') or device not in ('cpu', 'cuda'):
        raise ValueError('Choose cpu/cuda and fp32/bf16')
    if device == 'cuda' and (not torch.cuda.is_available() or
                            (precision == 'bf16' and not torch.cuda.is_bf16_supported())):
        raise RuntimeError('Requested CUDA/precision is unavailable')
    # FP32 master weights for fine-tuning; BF16 is an explicit autocast policy.
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.float32,
        attn_implementation='sdpa').to(device).eval()
    if len(tokenizer) > model.config.vocab_size or model.config.eos_token_id != tokenizer.eos_token_id:
        raise ValueError('Checkpoint/tokenizer vocabulary mismatch')
    return model, tokenizer


def checkpoint_identity(path):
    from capacity_allocation.data import sha256
    path = Path(path).resolve(strict=True)
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.suffix == '.safetensors' or p.name in ('config.json', 'model.safetensors.index.json')))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Expected local safetensors checkpoint')
    return dict(path=str(path), files={p.name: sha256(p) for p in files})


def checkpoint_specs(values):
    result = {}
    for value in values:
        name, separator, path = value.partition('=')
        if not separator or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in result:
            raise ValueError('Use unique NAME=/absolute/checkpoint entries')
        path = Path(path)
        if not path.is_absolute() or not (path/'config.json').is_file():
            raise ValueError(f'Missing local checkpoint config: {path}')
        result[name] = str(path.resolve())
    if not result:
        raise ValueError('Select checkpoints explicitly')
    return result
