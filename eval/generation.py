"""Deterministic, uncached decoding through the complete trained architecture.

Full-prefix recomputation deliberately avoids introducing a proxy KV-cache
implementation. It is slower, but uses the same causal forward as likelihood
evaluation. Prompts are never silently shortened to accommodate generation.
"""
import torch


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=256, until=()):
    if model.training:
        raise ValueError('Generation requires eval mode')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    if isinstance(until, str):
        until = [until]
    if any(not isinstance(s, str) or not s for s in until):
        raise ValueError('Stop strings must be nonempty strings')
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        if tokenizer.eos_token_id is None:
            raise ValueError('Empty prompt requires an EOS prefix')
        ids = [tokenizer.eos_token_id]
    limit = min(2048, model.config.max_position_embeddings)
    if len(ids) + max_new_tokens > limit:
        raise ValueError(f'Prompt ({len(ids)}) plus generation budget ({max_new_tokens}) '
                         f'exceeds context {limit}; choose a smaller fixed shot count/budget')
    tokens = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    generated = []
    text = ''
    for _ in range(max_new_tokens):
        logits = model(input_ids=tokens, use_cache=False).logits[:, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite generation logits')
        next_id = int(logits.argmax(-1))
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        stops = [text.index(s) for s in until if s in text]
        if stops:
            return text[:min(stops)]
        tokens = torch.cat((tokens, tokens.new_tensor([[next_id]])), dim=1)
    return text


def harness_model_class():
    from lm_eval.models.huggingface import HFLM

    class FullPrefixHFLM(HFLM):
        def generate_until(self, requests, disable_tqdm=False):
            results = []
            for request in requests:
                prompt, options = request.args
                options = dict(options)
                until = options.pop('until', [])
                budget = options.pop('max_gen_toks', 256)
                if options.pop('do_sample', False) or options.pop('temperature', 0) not in (0, 0.0):
                    raise ValueError('Benchmark protocol requires greedy decoding')
                if options:
                    raise ValueError(f'Unsupported generation settings: {sorted(options)}')
                answer = generate_answer(self.model, self.tokenizer, prompt, budget, until)
                results.append(answer)
                self.cache_hook.add_partial('generate_until', request.args, answer)
            return results

    return FullPrefixHFLM
