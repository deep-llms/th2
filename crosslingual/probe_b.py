"""Probe B — translation-vs-random cosine gap at the embedding layer.

Isolated-word representations (single-token by default): mean cosine of
translation-equivalent pairs minus mean cosine of random cross-language
pairs, with a Mann-Whitney U p-value and per-language-pair breakdown.

Ported from cross_lingual_embeddings_hub diagnostics/test_probe_ab.py
(Probe B half; Probe A is hub-architecture-specific and is replaced in this
project by a theta-based variant — see the anchor-usage analysis).

Caveat from the source project: this metric is necessary but not sufficient
("gameable" — a plain linear transform doubled it); read together with T6/T8.
"""

import random

import torch
import torch.nn.functional as F

from crosslingual.utils import LANGS, get_layer_reps

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@torch.no_grad()
def run_probe_b(model, tokenizer, translations, device, layer_idx=None,
                single_token_only=True):
    word_reps = {}   # (tuple_idx, lang) -> representation
    token_counts = {}

    for t_idx, (en_word, trans) in enumerate(translations):
        for lang, word in {"en": en_word, **trans}.items():
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            token_counts[(t_idx, lang)] = len(ids)
            input_ids = torch.tensor([ids], device=device)
            rep = get_layer_reps(model, input_ids, layer_idx)
            word_reps[(t_idx, lang)] = rep.squeeze(0).float().mean(dim=0)

    if single_token_only:
        for key in [k for k, n in token_counts.items() if n > 1]:
            word_reps.pop(key, None)

    trans_cos, per_pair = [], {}
    for t_idx in range(len(translations)):
        langs_in = [l for l in LANGS if (t_idx, l) in word_reps]
        for i, l1 in enumerate(langs_in):
            for l2 in langs_in[i + 1:]:
                cos = F.cosine_similarity(
                    word_reps[(t_idx, l1)].unsqueeze(0),
                    word_reps[(t_idx, l2)].unsqueeze(0)).item()
                trans_cos.append(cos)
                pair_key = f"{min(l1, l2)}-{max(l1, l2)}"
                per_pair.setdefault(pair_key, []).append(cos)

    random.seed(42)
    rand_cos = []
    all_keys = list(word_reps.keys())
    for _ in range(len(trans_cos)):
        k1, k2 = random.sample(all_keys, 2)
        if k1[1] == k2[1]:
            continue
        rand_cos.append(F.cosine_similarity(
            word_reps[k1].unsqueeze(0), word_reps[k2].unsqueeze(0)).item())

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    if trans_cos and rand_cos and HAVE_SCIPY:
        _, pval = stats.mannwhitneyu(trans_cos, rand_cos,
                                     alternative="greater")
    else:
        pval = None

    return {
        "cos_translation": mean(trans_cos),
        "cos_random": mean(rand_cos),
        "cos_gap": mean(trans_cos) - mean(rand_cos),
        "pvalue": pval,
        "n_translation": len(trans_cos),
        "n_random": len(rand_cos),
        "per_pair": {p: {"mean": mean(v), "n": len(v)}
                     for p, v in sorted(per_pair.items())},
        "layer": "embedding" if layer_idx is None else layer_idx,
    }
