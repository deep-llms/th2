"""Shared utilities for cross-lingual transfer tests.

Ported and adapted from /disk/thuat/cross_lingual_embeddings_hub
(diagnostics/test_utils.py). Key differences from the source project:

- Model loading routes through eval/eval_checkpoint.load_model, which handles
  both plain HF checkpoints (baseline) and our compositional checkpoints
  (embedding.pt + EmbeddingShim). Every model in this project modifies only
  the input embedding, so hidden_states[0] — the embedding output — IS the
  custom module's output, and all tests measure models through the standard
  forward pass (output_hidden_states=True / .logits). No hub_info plumbing.
- The tokenizer is always explicit (no silent fallback).

Methodology notes inherited from the source project (hard-won lessons):
- Loanword/cognate pairs (translation == English word) must be removed
  per-pair; leaving them inflates alignment metrics ~20x.
- Report related (en-de, en-vi) separately from distant (en-zh, en-ru, en-ar).
- en-vi has very few single-token pairs under the Qwen3 tokenizer — do not
  over-read it.
- Single-token words only for word-level tests (multi-token mean-pooling
  dilutes signal ~3x).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

LANGS = ["en", "vi", "zh", "ru", "de", "ar"]
NON_EN_LANGS = ["vi", "zh", "ru", "de", "ar"]
RELATED_PAIRS = ["en-de", "en-vi"]
DISTANT_PAIRS = ["en-zh", "en-ar", "en-ru"]

DEFAULT_TRANSLATIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "frequent_translations_llm.json")


def load_model(checkpoint_path, device="cuda", dtype=torch.bfloat16):
    """Load a checkpoint (plain HF or compositional) ready for eval."""
    from eval.eval_checkpoint import load_model as _load
    return _load(checkpoint_path, device, dtype=dtype)


def get_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def get_layer_reps(model, input_ids, layer_idx=None):
    """Hidden states at one layer via a standard forward pass.

    layer_idx=None -> embedding output (hidden_states[0]; for our models this
    is the compositional module's output). layer_idx=k -> after transformer
    layer k (hidden_states[k+1]).
    """
    outputs = model(input_ids=input_ids, output_hidden_states=True,
                    return_dict=True)
    if layer_idx is None:
        return outputs.hidden_states[0]
    return outputs.hidden_states[layer_idx + 1]


# ---------------------------------------------------------------------------
# Translation tuples
# ---------------------------------------------------------------------------

def load_translations(path=None, single_token_only=False, tokenizer=None):
    """Load (en_word, {lang: word}) tuples from the LLM translation file."""
    with open(path or DEFAULT_TRANSLATIONS) as f:
        data = json.load(f)
    tuples = list(data["all_five_tuples"].items())
    if single_token_only:
        if tokenizer is None:
            raise ValueError("tokenizer required when single_token_only=True")
        tuples = _filter_single_token(tuples, tokenizer)
    return tuples


def _filter_single_token(tuples, tokenizer):
    filtered = []
    for en_word, translations in tuples:
        if len(tokenizer(en_word, add_special_tokens=False)["input_ids"]) != 1:
            continue
        if all(len(tokenizer(w, add_special_tokens=False)["input_ids"]) == 1
               for w in translations.values()):
            filtered.append((en_word, translations))
    return filtered


def filter_loanwords_per_pair(tuples):
    """Drop only the language pairs where translation == English word.

    ("bus", {"vi": "bus", "zh": "公共汽车"}) -> ("bus", {"zh": "公共汽车"})
    Tuples left with no translations are dropped.
    """
    filtered = []
    for en_word, translations in tuples:
        clean = {lang: w for lang, w in translations.items()
                 if w.lower() != en_word.lower()}
        if clean:
            filtered.append((en_word, clean))
    return filtered


# ---------------------------------------------------------------------------
# CSLS (Cross-domain Similarity Local Scaling)
# ---------------------------------------------------------------------------

def compute_csls_scores(source_embs, target_embs, k=10):
    """CSLS(e, x) = 2*cos(e, x) - r(e) - r(x); inputs must be L2-normalized.

    r(e) = mean cos of e to its k nearest TARGET neighbors,
    r(x) = mean cos of x to its k nearest SOURCE neighbors.
    Returns an (N_src, N_tgt) score matrix.
    """
    cos_matrix = source_embs @ target_embs.T
    k_src = min(k, target_embs.shape[0])
    r_source = cos_matrix.topk(k_src, dim=1).values.mean(dim=1)
    k_tgt = min(k, source_embs.shape[0])
    r_target = cos_matrix.T.topk(k_tgt, dim=1).values.mean(dim=1)
    return 2 * cos_matrix - r_source.unsqueeze(1) - r_target.unsqueeze(0)


def summarize_pairs(results_per_lang, metric="p_at_1"):
    """Mean of a metric over all/related/distant pairs (pairs with n>0 only)."""
    vals, rel, dist = [], [], []
    for pair, res in results_per_lang.items():
        if res.get("n_pairs", res.get("n", 0)) == 0:
            continue
        v = res[metric]
        vals.append(v)
        if pair in RELATED_PAIRS:
            rel.append(v)
        elif pair in DISTANT_PAIRS:
            dist.append(v)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"mean": mean(vals), "related_mean": mean(rel),
            "distant_mean": mean(dist)}
