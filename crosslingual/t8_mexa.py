"""T8 — MEXA sentence-level alignment on FLORES-200 (Kargaran et al. 2024).

Per layer: position-weighted sentence embeddings for N parallel En/XX
sentences; MEXA score = fraction of pairs whose cosine strictly beats every
off-diagonal entry in BOTH its row and column (mutual-NN, hubness-robust).

Ported from cross_lingual_embeddings_hub diagnostics/test_t8_mexa.py, which
was verified line-by-line against the official cisnlp/MEXA implementation
(position weights w_t = t * attention_mask, float64 cosines).
"""

import os

import numpy as np
import torch

from crosslingual.utils import NON_EN_LANGS

LANG_TO_FLORES = {
    "en": "eng_Latn", "vi": "vie_Latn", "zh": "zho_Hans",
    "ru": "rus_Cyrl", "de": "deu_Latn", "ar": "arb_Arab",
}

DEFAULT_FLORES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "flores200")


def load_flores_from_dir(flores_dir, lang, n_sentences=100):
    code = LANG_TO_FLORES.get(lang, lang)
    for name in (f"{code}.txt", f"{lang}.txt", f"{code}.devtest"):
        path = os.path.join(flores_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines[:n_sentences]
    return None


def load_flores_all(flores_dir=None, n_sentences=100):
    flores_dir = flores_dir or DEFAULT_FLORES_DIR
    out = {}
    for lang in ["en"] + NON_EN_LANGS:
        sents = load_flores_from_dir(flores_dir, lang, n_sentences)
        if sents:
            out[lang] = sents
    return out


@torch.no_grad()
def compute_sentence_embeddings(model, tokenizer, sentences, device):
    """Position-weighted sentence embedding per layer (official MEXA formula).

    Returns {layer_idx: (n_sentences, d) float64 array}; layer 0 = embedding
    output.
    """
    all_layer_embs = {}
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", padding=True).to(device)
        attention_mask = inputs["attention_mask"]
        seq_len = inputs["input_ids"].shape[1]
        outputs = model(**inputs, output_hidden_states=True)

        positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)
        weights = attention_mask * positions
        weight_sum = weights.sum(dim=-1).unsqueeze(-1)

        for layer_idx, hs in enumerate(outputs["hidden_states"]):
            emb = (torch.sum(hs * weights.unsqueeze(-1), dim=1)
                   / weight_sum).squeeze(0)
            emb = emb.to(torch.float32).cpu().numpy().astype(np.float64)
            all_layer_embs.setdefault(layer_idx, []).append(emb)

    return {k: np.stack(v) for k, v in all_layer_embs.items()}


def compute_mexa_score(pivot_embs, target_embs):
    """Fraction of pairs where c_ii strictly beats all off-diagonal entries in
    both its row and its column of the cosine matrix."""
    N = pivot_embs.shape[0]
    assert target_embs.shape[0] == N
    p = pivot_embs / np.linalg.norm(pivot_embs, axis=1, keepdims=True)
    t = target_embs / np.linalg.norm(target_embs, axis=1, keepdims=True)
    sim = p @ t.T
    if N == 1:
        return 1.0
    correct = 0
    off = ~np.eye(N, dtype=bool)
    for i in range(N):
        diag = sim[i, i]
        row_max = sim[i][off[i]].max()
        col_max = sim[:, i][off[:, i]].max()
        if diag > row_max and diag > col_max:
            correct += 1
    return correct / N


def run_t8(model, tokenizer, flores_sentences, device, layers=None):
    """MEXA for all en-X pairs. Returns per-pair per-layer scores + summary."""
    if "en" not in flores_sentences:
        return {"error": "No English FLORES sentences"}

    lang_embs = {}
    for lang, sentences in flores_sentences.items():
        lang_embs[lang] = compute_sentence_embeddings(
            model, tokenizer, sentences, device)

    if layers is None:
        layers = sorted(lang_embs["en"].keys())

    results = {}
    for lang in NON_EN_LANGS:
        if lang not in lang_embs:
            continue
        per_layer = {}
        for layer_idx in layers:
            en = lang_embs["en"][layer_idx]
            tgt = lang_embs[lang][layer_idx]
            n = min(len(en), len(tgt))
            per_layer[layer_idx] = compute_mexa_score(en[:n], tgt[:n])
        scores = list(per_layer.values())
        best_layer = max(per_layer, key=per_layer.get)
        results[f"en-{lang}"] = {
            "per_layer": {str(k): v for k, v in per_layer.items()},
            "mean_over_layers": sum(scores) / len(scores),
            "best_layer": best_layer,
            "best_score": per_layer[best_layer],
            "n_sentences": n,
        }

    means = [r["mean_over_layers"] for r in results.values()]
    bests = [r["best_score"] for r in results.values()]
    results["summary"] = {
        "avg_mean_over_layers": sum(means) / len(means) if means else 0.0,
        "avg_best_score": sum(bests) / len(bests) if bests else 0.0,
    }
    return results
