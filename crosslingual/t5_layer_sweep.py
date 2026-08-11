"""T5 — per-layer cross-lingual similarity sweep with in-context occurrences.

For each layer: mean cosine of translation pairs minus mean cosine of random
en-X pairs, where each word's representation is mean-pooled over up to
max_occurrences real occurrences in monolingual eval text (128-token windows
centered on the occurrence). Locates the layer where alignment peaks.

Ported from cross_lingual_embeddings_hub diagnostics/test_t5_layer_sweep.py.
Caveats from the source project: read the GAP, not absolute cosines (deep
layers are anisotropic); the random control is not strictly frequency-matched;
this is a setup/diagnostic tool, not a verdict metric.
"""

import os
import random

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk

from crosslingual.utils import NON_EN_LANGS


def word_token_ids(tokenizer, word):
    """Single-token ids for a word in both its bare and space-prefixed forms.

    GPT-style BPE tokenizes "cat" and " cat" to different ids; in running
    text the space-prefixed form dominates, so searching only the bare id
    (as the source project did) finds almost nothing mid-sentence.
    """
    ids = []
    for form in (word, " " + word):
        toks = tokenizer(form, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            ids.append(toks[0])
    return ids


def find_word_occurrences(token_ids, sequences, max_occurrences=50):
    occurrences = []
    for seq_idx, ids in enumerate(sequences):
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for tid in token_ids:
            mask |= ids == tid
        for pos in mask.nonzero(as_tuple=True)[0]:
            occurrences.append((seq_idx, pos.item()))
            if len(occurrences) >= max_occurrences:
                return occurrences
    return occurrences


@torch.no_grad()
def get_hidden_states_at_positions(model, input_ids_batch, positions, layers,
                                   device):
    outputs = model(input_ids=input_ids_batch.to(device),
                    output_hidden_states=True, return_dict=True)
    result = {}
    for layer_idx in layers:
        hs = (outputs.hidden_states[0] if layer_idx is None
              else outputs.hidden_states[layer_idx + 1])
        result[layer_idx] = torch.stack(
            [hs[i, pos] for i, pos in enumerate(positions)]).float()
    return result


def collect_word_representations(model, tokenizer, word, sequences, layers,
                                 device, max_occurrences=50, context_len=128):
    token_ids = word_token_ids(tokenizer, word)
    if not token_ids:
        return None
    occurrences = find_word_occurrences(token_ids, sequences, max_occurrences)
    if not occurrences:
        return None

    all_states = {l: [] for l in layers}
    for start in range(0, len(occurrences), 16):
        batch_occ = occurrences[start:start + 16]
        batch_inputs, batch_positions = [], []
        for seq_idx, pos in batch_occ:
            seq = sequences[seq_idx]
            s = max(0, pos - context_len // 2)
            e = min(len(seq), s + context_len)
            s = max(0, e - context_len)
            batch_inputs.append(seq[s:e])
            batch_positions.append(pos - s)
        max_len = max(len(x) for x in batch_inputs)
        padded = torch.zeros(len(batch_inputs), max_len, dtype=torch.long)
        for i, inp in enumerate(batch_inputs):
            padded[i, :len(inp)] = inp
        states = get_hidden_states_at_positions(
            model, padded, batch_positions, layers, device)
        for l in layers:
            all_states[l].append(states[l])

    return {l: torch.cat(all_states[l], dim=0).mean(dim=0) for l in layers}


def load_eval_sequences(eval_dir, tokenizer, lang, max_sequences=2000,
                        block_size=128):
    lang_path = os.path.join(eval_dir, lang)
    if not os.path.isdir(lang_path):
        return []
    shard_dirs = sorted(
        os.path.join(lang_path, d) for d in os.listdir(lang_path)
        if d.startswith("shard_") and os.path.isdir(os.path.join(lang_path, d)))
    ds = (concatenate_datasets([load_from_disk(sd) for sd in shard_dirs])
          if shard_dirs else load_from_disk(lang_path))

    sequences, buffer = [], []
    for example in ds:
        if len(sequences) >= max_sequences:
            break
        text = example.get("text", "")
        if not text:
            continue
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        while len(buffer) >= block_size:
            sequences.append(torch.tensor(buffer[:block_size],
                                          dtype=torch.long))
            buffer = buffer[block_size:]
            if len(sequences) >= max_sequences:
                break
    return sequences


def run_t5(model, tokenizer, translations, eval_dir, layers, device,
           max_occurrences=50, max_sequences=2000, context_len=128):
    eval_seqs = {}
    for lang in ["en"] + NON_EN_LANGS:
        seqs = load_eval_sequences(eval_dir, tokenizer, lang, max_sequences,
                                   context_len)
        if seqs:
            eval_seqs[lang] = seqs
    if "en" not in eval_seqs:
        raise ValueError(f"No English eval data in {eval_dir}")

    word_reps = {}
    for en_word, trans in translations:
        reps = collect_word_representations(
            model, tokenizer, en_word, eval_seqs["en"], layers, device,
            max_occurrences, context_len)
        if reps is not None:
            word_reps[(en_word, "en")] = reps
        for lang, word in trans.items():
            if lang not in eval_seqs:
                continue
            reps = collect_word_representations(
                model, tokenizer, word, eval_seqs[lang], layers, device,
                max_occurrences, context_len)
            if reps is not None:
                word_reps[(word, lang)] = reps

    per_layer = {}
    for layer_idx in layers:
        trans_cos = []
        for en_word, trans in translations:
            if (en_word, "en") not in word_reps:
                continue
            en_vec = word_reps[(en_word, "en")][layer_idx]
            for lang, word in trans.items():
                if (word, lang) not in word_reps:
                    continue
                trans_cos.append(F.cosine_similarity(
                    en_vec.unsqueeze(0),
                    word_reps[(word, lang)][layer_idx].unsqueeze(0)).item())

        random.seed(42)
        rand_cos = []
        en_keys = [k for k in word_reps if k[1] == "en"]
        non_en_keys = [k for k in word_reps if k[1] != "en"]
        if en_keys and non_en_keys:
            for _ in range(len(trans_cos)):
                v1 = word_reps[random.choice(en_keys)][layer_idx]
                v2 = word_reps[random.choice(non_en_keys)][layer_idx]
                rand_cos.append(F.cosine_similarity(
                    v1.unsqueeze(0), v2.unsqueeze(0)).item())

        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        layer_name = "embedding" if layer_idx is None else f"layer_{layer_idx}"
        per_layer[layer_name] = {
            "translation_cos_mean": mean(trans_cos),
            "random_cos_mean": mean(rand_cos),
            "gap": mean(trans_cos) - mean(rand_cos),
            "n_translation_pairs": len(trans_cos),
            "n_random_pairs": len(rand_cos),
        }

    peak = max(per_layer, key=lambda k: per_layer[k]["gap"]) if per_layer else None
    return {"per_layer": per_layer, "peak_layer": peak,
            "peak_gap": per_layer[peak]["gap"] if peak else 0.0,
            "n_words_found": len(word_reps)}
