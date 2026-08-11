"""T6 — Bilingual Lexicon Induction (translation retrieval) via CSLS.

For each English word, rank all single-token target-language candidate words
by CSLS similarity of their representations; check whether the true
translation is retrieved. Reports P@1 / P@5 per language pair plus
related/distant means.

Ported from cross_lingual_embeddings_hub diagnostics/test_t6_bli_retrieval.py
(hub_info removed; measurement layer is an explicit argument, default the
embedding output = our compositional module's output).
"""

import torch
import torch.nn.functional as F

from crosslingual.utils import (NON_EN_LANGS, compute_csls_scores,
                                get_layer_reps, summarize_pairs)


@torch.no_grad()
def build_word_embeddings(model, tokenizer, words, device, layer_idx=None,
                          batch_size=256):
    """Representations for single-token words; returns (embs, valid_words).

    embs is (N_valid, d) float32, L2-normalized. Multi-token words dropped.
    """
    all_ids, valid_words = [], []
    for word in words:
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            all_ids.append(ids[0])
            valid_words.append(word)
    if not all_ids:
        return torch.empty(0, model.config.hidden_size), []

    reps = []
    for start in range(0, len(all_ids), batch_size):
        batch = all_ids[start:start + batch_size]
        input_ids = torch.tensor(batch, device=device).unsqueeze(1)  # (B, 1)
        r = get_layer_reps(model, input_ids, layer_idx)
        reps.append(r.squeeze(1).float())
    embs = F.normalize(torch.cat(reps, dim=0), dim=-1)
    return embs, valid_words


def run_bli_for_lang_pair(model, tokenizer, translations, target_lang, device,
                          layer_idx=None, csls_k=10):
    en_words, tgt_words = [], []
    en_word_set, tgt_word_set = {}, {}
    pairs = []
    for en_word, trans in translations:
        if target_lang not in trans:
            continue
        tgt_word = trans[target_lang]
        en_ids = tokenizer(en_word, add_special_tokens=False)["input_ids"]
        tgt_ids = tokenizer(tgt_word, add_special_tokens=False)["input_ids"]
        if len(en_ids) != 1 or len(tgt_ids) != 1:
            continue
        if en_word.lower() == tgt_word.lower():  # loanword
            continue
        if en_word not in en_word_set:
            en_word_set[en_word] = len(en_words)
            en_words.append(en_word)
        if tgt_word not in tgt_word_set:
            tgt_word_set[tgt_word] = len(tgt_words)
            tgt_words.append(tgt_word)
        pairs.append((en_word, tgt_word))

    if not pairs:
        return {"p_at_1": 0.0, "p_at_5": 0.0, "n_pairs": 0, "n_candidates": 0}

    en_embs, en_valid = build_word_embeddings(
        model, tokenizer, en_words, device, layer_idx)
    tgt_embs, tgt_valid = build_word_embeddings(
        model, tokenizer, tgt_words, device, layer_idx)
    en_emb_idx = {w: i for i, w in enumerate(en_valid)}
    tgt_emb_idx = {w: i for i, w in enumerate(tgt_valid)}

    csls = compute_csls_scores(en_embs, tgt_embs, k=csls_k)

    hits_1 = hits_5 = n_eval = 0
    for en_word, tgt_word in pairs:
        if en_word not in en_emb_idx or tgt_word not in tgt_emb_idx:
            continue
        scores = csls[en_emb_idx[en_word]]
        ranked = scores.argsort(descending=True)
        rank = (ranked == tgt_emb_idx[tgt_word]).nonzero(as_tuple=True)[0]
        if len(rank) == 0:
            continue
        rank = rank[0].item()
        n_eval += 1
        hits_1 += rank == 0
        hits_5 += rank < 5

    if n_eval == 0:
        return {"p_at_1": 0.0, "p_at_5": 0.0, "n_pairs": 0,
                "n_candidates": len(tgt_valid)}
    return {"p_at_1": hits_1 / n_eval, "p_at_5": hits_5 / n_eval,
            "n_pairs": n_eval, "n_candidates": len(tgt_valid)}


def run_t6(model, tokenizer, translations, device, layer_idx=None, csls_k=10):
    """BLI for all en->X pairs. Returns {per_lang, summary}."""
    per_lang = {}
    for lang in NON_EN_LANGS:
        per_lang[f"en-{lang}"] = run_bli_for_lang_pair(
            model, tokenizer, translations, lang, device, layer_idx, csls_k)
    summary = summarize_pairs(per_lang, "p_at_1")
    summary["mean_p5"] = summarize_pairs(per_lang, "p_at_5")["mean"]
    return {"per_lang": per_lang, "summary": summary,
            "layer": "embedding" if layer_idx is None else layer_idx,
            "csls_k": csls_k}
