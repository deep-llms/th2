"""Likelihood adapter for already verified native A2 parent/student models.

Checkpoint loading and GPU ownership remain the caller's responsibility. Use
a2.model.load_parent / reload_student with validated input receipts; never the
old Qwen/KV loader. This module does not launch a benchmark suite.
"""
from importlib.metadata import version

from eval.kv_harness import harness_class as likelihood_harness_class


def harness_class():
    import torch
    from efficient_llm.a2.model import autocast

    class A2Harness(likelihood_harness_class()):
        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                raise ValueError('Only causal likelihood requests are supported')
            # Match A2 validation: FP32 parameters, BF16 CUDA autocast. CPU
            # fixtures stay FP32; they are not evidence for GPU precision.
            with torch.inference_mode(), autocast(inps.device):
                return self.model(input_ids=inps, use_cache=False,
                    position_ids=torch.arange(inps.shape[1], device=inps.device)
                    .expand(inps.shape[0], -1)).logits

    return A2Harness


def make_harness(model, tokenizer, *, batch_size=8, max_length=4096):
    """Wrap an already loaded native model without recasting its parameters."""
    import torch
    from efficient_llm.a2.model import assert_native

    if version('lm-eval') != '0.4.13' or version('transformers') != '4.57.1':
        raise ValueError('Require A2 Transformers4.57.1 and lm-eval0.4.13')
    if (batch_size < 1 or not 2 <= max_length <= 4096
            or max_length > model.config.max_position_embeddings):
        raise ValueError('Invalid batch/context')
    assert_native(model)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('A2 evaluation requires FP32 parameters with BF16 CUDA autocast')
    if tokenizer.eos_token_id != 0:
        raise ValueError('Require the verified GPT-NeoX tokenizer with EOS0')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    lm = harness_class()(pretrained=model, tokenizer=tokenizer, backend='causal',
        device=str(model.device), batch_size=batch_size, max_length=max_length,
        add_bos_token=False, softmax_dtype='float32', mixed_precision_dtype=None,
        # Harness 0.4.13 groups identical input tokens even when requests split
        # context/continuation differently, then reuses already sliced logits.
        # Disable that reuse; ordinary batched likelihood computation is intact.
        logits_cache=False)
    if any(p.dtype != torch.float32 for p in lm.model.parameters()):
        raise ValueError('Harness changed model precision')
    return lm
