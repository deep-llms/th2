"""Checkpoint diagnostics for the Product-Code Hashed Chinese failure.

This implements Checks 3--5 in ``docs/hashed_zh_diagnostics.md``:

* prediction mass leaked to tokens sharing one or more hash buckets;
* cosine similarity and norm statistics for effective embedding rows;
* learned-gate distributions and their relationship to importance/PPL.

Checks 1, 2, and 6 intentionally remain in ``ppl_bytoken.py`` and
``ppl_bins.py`` so they use the project's established sliding-window and
binning paths.  With ``--eval-dir`` this script evaluates prediction leakage;
without it, only state-dict diagnostics are run.  The output is one strict
JSON file suitable for archiving with the other evaluation results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from compositional.product_code import ProductCodeEmbed, hashed_codes
from compositional.tied_head import TiedProductCodeHead
from eval.ppl_bins import frequency_rank_order, load_run
from eval.ppl_bytoken import validate_tokenizer_ids


LANGUAGE_COUNT_PREFIX = "counts_"
PAIR_CATEGORIES = ("zh_zh", "zh_other", "other_other")


def _summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "p01": None,
                "p50": None, "p99": None, "min": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("summary input contains non-finite values")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _pearson(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("correlation vectors must have identical shapes")
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if left.size < 2:
        return None
    left_scale = float(np.max(np.abs(left)))
    right_scale = float(np.max(np.abs(right)))
    if left_scale == 0 or right_scale == 0:
        return None
    left = left / left_scale
    right = right / right_scale
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator == 0:
        return None
    return float((left @ right) / denominator)


def load_language_counts(path, vocab_size):
    """Load per-language counts and return normalized shares + dominance."""
    with np.load(path, allow_pickle=False) as archive:
        languages = sorted(
            key.removeprefix(LANGUAGE_COUNT_PREFIX)
            for key in archive.files
            if key.startswith(LANGUAGE_COUNT_PREFIX)
        )
        if not languages:
            raise ValueError(f"{path}: no {LANGUAGE_COUNT_PREFIX}<lang> arrays")
        rows = []
        for language in languages:
            values = np.asarray(
                archive[f"{LANGUAGE_COUNT_PREFIX}{language}"], dtype=np.float64
            )
            if values.shape != (vocab_size,):
                raise ValueError(
                    f"{path}: counts_{language} has shape {values.shape}, "
                    f"expected ({vocab_size},)"
                )
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{path}: invalid counts_{language}")
            total = float(values.sum())
            if total <= 0:
                raise ValueError(f"{path}: counts_{language} has zero mass")
            rows.append(values / total)
    normalized = np.stack(rows, axis=0)
    observed = normalized.sum(axis=0) > 0
    dominant = np.argmax(normalized, axis=0).astype(np.int16)
    dominant[~observed] = -1
    return languages, normalized, dominant


def load_importance(path, key, vocab_size):
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"{path}: missing array {key!r}")
        values = np.asarray(archive[key], dtype=np.float64)
    if values.shape != (vocab_size,):
        raise ValueError(
            f"{path}: importance shape {values.shape}, expected ({vocab_size},)"
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: importance must be finite and nonnegative")
    return values


class BucketMateIndex:
    """CPU index from each (hash, bucket) to its tail-row members."""

    def __init__(self, tail_ids, codes, num_buckets):
        self.tail_ids = np.asarray(tail_ids, dtype=np.int64)
        self.codes = np.asarray(codes, dtype=np.int64)
        self.num_buckets = int(num_buckets)
        if self.codes.ndim != 2 or self.codes.shape[0] != self.tail_ids.size:
            raise ValueError("codes and tail_ids have incompatible shapes")
        if self.codes.size and (
            self.codes.min() < 0 or self.codes.max() >= self.num_buckets
        ):
            raise ValueError("codes contain an out-of-range bucket")
        self.num_hashes = int(self.codes.shape[1])
        self._members = []
        for hash_index in range(self.num_hashes):
            buckets = [[] for _ in range(self.num_buckets)]
            for row, bucket in enumerate(self.codes[:, hash_index]):
                buckets[int(bucket)].append(row)
            self._members.append([
                np.asarray(rows, dtype=np.int64) for rows in buckets
            ])
        self._cache = {}

    def mate_rows(self, tail_row):
        tail_row = int(tail_row)
        cached = self._cache.get(tail_row)
        if cached is not None:
            return cached
        if not 0 <= tail_row < self.tail_ids.size:
            raise IndexError(f"tail row {tail_row} is out of range")
        pieces = [
            self._members[index][int(self.codes[tail_row, index])]
            for index in range(self.num_hashes)
        ]
        rows = np.unique(np.concatenate(pieces))
        rows = rows[rows != tail_row]
        self._cache[tail_row] = rows
        return rows

    def mate_token_ids(self, tail_row):
        return self.tail_ids[self.mate_rows(tail_row)]


def fixed_random_token_ids(index, tail_row, size, seed):
    """Deterministic same-size tail baseline excluding target and mates."""
    tail_row, size = int(tail_row), int(size)
    if size < 0:
        raise ValueError("random-set size must be nonnegative")
    excluded_rows = set(index.mate_rows(tail_row).tolist())
    excluded_rows.add(tail_row)
    available = index.tail_ids.size - len(excluded_rows)
    if size > available:
        raise ValueError(
            f"cannot draw {size} random rows from {available} eligible rows"
        )
    target_id = int(index.tail_ids[tail_row])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), target_id]))
    chosen = set()
    while len(chosen) < size:
        need = size - len(chosen)
        for candidate in rng.integers(
            0, index.tail_ids.size, size=max(32, need * 2), endpoint=False
        ):
            candidate = int(candidate)
            if candidate not in excluded_rows:
                chosen.add(candidate)
                if len(chosen) == size:
                    break
    return index.tail_ids[np.fromiter(chosen, dtype=np.int64, count=size)]


def _padded_ids(rows, device):
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return None, None
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    valid = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        row = np.asarray(row, dtype=np.int64)
        ids[index, :row.size] = torch.from_numpy(row)
        valid[index, :row.size] = True
    return ids.to(device), valid.to(device)


@torch.no_grad()
def analyze_logits(logits, target_ids, mate_ids, random_ids, batch_size=64):
    """Return per-position leakage values without materializing probabilities."""
    if logits.ndim != 2:
        raise ValueError("logits must be a [positions, vocab] matrix")
    target_ids = np.asarray(target_ids, dtype=np.int64)
    if target_ids.shape != (logits.shape[0],):
        raise ValueError("target_ids must contain one id per logits row")
    if len(mate_ids) != logits.shape[0] or len(random_ids) != logits.shape[0]:
        raise ValueError("mate/random lists must contain one row per position")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output = defaultdict(list)
    device = logits.device
    for start in range(0, logits.shape[0], batch_size):
        end = min(start + batch_size, logits.shape[0])
        block = logits[start:end].float()
        target = torch.as_tensor(target_ids[start:end], device=device)
        target_logit = block.gather(1, target[:, None]).squeeze(1)
        log_z = torch.logsumexp(block, dim=1)

        mate_index, mate_valid = _padded_ids(mate_ids[start:end], device)
        random_index, random_valid = _padded_ids(random_ids[start:end], device)
        if mate_index is None or random_index is None:
            raise ValueError("every analyzed target must have a non-empty mate set")
        mate_logits = block.gather(1, mate_index).masked_fill(~mate_valid, -torch.inf)
        random_logits = block.gather(1, random_index).masked_fill(
            ~random_valid, -torch.inf
        )
        mate_log_mass = torch.logsumexp(mate_logits, dim=1) - log_z
        random_log_mass = torch.logsumexp(random_logits, dim=1) - log_z
        target_log_prob = target_logit - log_z
        argmax = block.argmax(dim=1)

        values = {
            "p_target": target_log_prob.exp(),
            "p_mates": mate_log_mass.exp(),
            "p_random": random_log_mass.exp(),
            "mates_over_random": (mate_log_mass - random_log_mass).double().exp(),
            "mates_over_target": (mate_log_mass - target_log_prob).double().exp(),
            "log_mates_over_random": mate_log_mass - random_log_mass,
            "log_mates_over_target": mate_log_mass - target_log_prob,
            "target_rank": (block > target_logit[:, None]).sum(dim=1) + 1,
            "argmax_is_mate": (
                (mate_index == argmax[:, None]) & mate_valid
            ).any(dim=1),
        }
        for name, tensor in values.items():
            array = tensor.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite prediction diagnostic {name}")
            output[name].append(array)
    return {name: np.concatenate(parts) for name, parts in output.items()}


def summarize_prediction_values(values):
    count = int(values["p_target"].size)
    if count == 0:
        raise ValueError("prediction diagnostic received zero positions")
    return {
        "num_tail_positions": count,
        "mean_p_target": float(values["p_target"].mean()),
        "mean_p_mates": float(values["p_mates"].mean()),
        "mean_p_random": float(values["p_random"].mean()),
        "mean_mates_over_random": float(values["mates_over_random"].mean()),
        "geomean_mates_over_random": float(
            math.exp(float(values["log_mates_over_random"].mean()))
        ),
        "mean_mates_over_target": float(values["mates_over_target"].mean()),
        "geomean_mates_over_target": float(
            math.exp(float(values["log_mates_over_target"].mean()))
        ),
        "mean_target_rank": float(values["target_rank"].mean()),
        "median_target_rank": float(np.median(values["target_rank"])),
        "argmax_bucket_mate_fraction": float(values["argmax_is_mate"].mean()),
    }


@torch.no_grad()
def prediction_diagnostics(
    model,
    tokenizer,
    eval_dir,
    languages,
    embed,
    mate_index,
    *,
    block_size,
    stride,
    device,
    random_seed,
    logits_batch_size,
    max_positions,
    verification_gap_limit,
):
    """Run Check 3 with the exact project sliding-window target selection."""
    results = {}
    random_cache = {}
    tail_row_cpu = embed.tail_row.detach().cpu().numpy()

    for language in languages:
        dataset_path = os.path.join(eval_dir, language)
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"missing eval language directory {dataset_path}")
        dataset = load_from_disk(dataset_path)
        input_ids = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        if input_ids.size(1) < block_size:
            raise ValueError(
                f"{language}: {input_ids.size(1)} tokens is below block size "
                f"{block_size}"
            )

        collected = defaultdict(list)
        total_positions = 0
        observed_max_gap = 0.0
        truncated = False
        prev_end = 0
        starts = range(0, input_ids.size(1), stride)
        for begin in tqdm(starts, desc=f"  leakage-{language}"):
            end = min(begin + block_size, input_ids.size(1))
            target_length = end - prev_end
            chunk = input_ids[:, begin:end].to(device)
            labels = chunk.clone()
            labels[:, :-target_length] = -100
            outputs = model(chunk, labels=labels)
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            counted = shift_labels != -100
            all_logits = shift_logits[counted]
            all_targets = shift_labels[counted]
            manual_loss = F.cross_entropy(
                all_logits.float(), all_targets, reduction="mean"
            ).item()
            observed_max_gap = max(
                observed_max_gap, abs(manual_loss - outputs.loss.item())
            )
            total_positions += int(all_targets.numel())

            target_cpu = all_targets.detach().cpu().numpy()
            target_tail_rows = tail_row_cpu[target_cpu]
            selected = target_tail_rows >= 0
            if selected.any():
                selected_rows = target_tail_rows[selected]
                selected_targets = target_cpu[selected]
                selected_mask = torch.from_numpy(selected).to(all_logits.device)
                selected_logits = all_logits[selected_mask]
                if max_positions:
                    remaining = max_positions - sum(
                        part.size for part in collected["p_target"]
                    )
                    if remaining <= 0:
                        truncated = True
                        break
                    if selected_rows.size > remaining:
                        truncated = True
                    selected_rows = selected_rows[:remaining]
                    selected_targets = selected_targets[:remaining]
                    selected_logits = selected_logits[:remaining]

                mates = []
                randoms = []
                for tail_row in selected_rows:
                    tail_row = int(tail_row)
                    mate_tokens = mate_index.mate_token_ids(tail_row)
                    if mate_tokens.size == 0:
                        raise ValueError(
                            f"tail row {tail_row} has no bucket mates"
                        )
                    mates.append(mate_tokens)
                    cache_key = (tail_row, int(mate_tokens.size))
                    if cache_key not in random_cache:
                        random_cache[cache_key] = fixed_random_token_ids(
                            mate_index, tail_row, mate_tokens.size, random_seed
                        )
                    randoms.append(random_cache[cache_key])
                batch_values = analyze_logits(
                    selected_logits,
                    selected_targets,
                    mates,
                    randoms,
                    batch_size=logits_batch_size,
                )
                for name, values in batch_values.items():
                    collected[name].append(values)

            prev_end = end
            del outputs, shift_logits, all_logits
            if end == input_ids.size(1):
                break

        values = {
            name: np.concatenate(parts) if parts else np.empty(0)
            for name, parts in collected.items()
        }
        if not values or values.get("p_target", np.empty(0)).size == 0:
            raise ValueError(f"{language}: no tail target positions were analyzed")
        summary = summarize_prediction_values(values)
        summary.update({
            "num_all_positions_seen": total_positions,
            "analyzed_tail_positions_over_all_positions_seen": (
                summary["num_tail_positions"] / total_positions
            ),
            "max_loss_verification_gap": observed_max_gap,
            "truncated_at_max_positions": truncated,
        })
        if summary["max_loss_verification_gap"] > verification_gap_limit:
            raise RuntimeError(
                f"{language}: manual-vs-model loss gap "
                f"{summary['max_loss_verification_gap']:.6g} exceeds "
                f"{verification_gap_limit:.6g}"
            )
        results[language] = summary
    return results


def materialize_table(embed, chunk_size):
    if chunk_size <= 0:
        raise ValueError("materialize chunk size must be positive")
    table = torch.empty(
        (embed.vocab_size, embed.embed_dim), dtype=torch.float32,
        device="cpu",
    )
    device = embed.bias.device
    with torch.no_grad():
        for start in tqdm(
            range(0, embed.vocab_size, chunk_size), desc="  materialize-table"
        ):
            end = min(start + chunk_size, embed.vocab_size)
            ids = torch.arange(start, end, device=device)
            table[start:end] = embed.materialize(ids).float().cpu()
    return table


def _pair_category(left, right, zh_index):
    left, right = int(left), int(right)
    if left < 0 or right < 0:
        return None
    if left == zh_index and right == zh_index:
        return "zh_zh"
    if (left == zh_index) != (right == zh_index):
        return "zh_other"
    return "other_other"


def sample_shared_pairs(index, dominant_tail, zh_index, sample_size, seed):
    """Token-centric samples from the union of each anchor's four buckets."""
    rng = np.random.default_rng(seed)
    pairs = {name: set() for name in PAIR_CATEGORIES}
    zh_rows = np.flatnonzero(dominant_tail == zh_index)
    other_rows = np.flatnonzero(
        (dominant_tail >= 0) & (dominant_tail != zh_index)
    )
    if zh_rows.size == 0 or other_rows.size == 0:
        raise ValueError("not enough zh/other tail rows for shared-pair sampling")
    specs = {
        "zh_zh": (zh_rows, lambda values: values == zh_index),
        "zh_other": (zh_rows, lambda values: (values >= 0) & (values != zh_index)),
        "other_other": (
            other_rows,
            lambda values: (values >= 0) & (values != zh_index),
        ),
    }
    max_attempts = max(100_000, sample_size * 50)
    for category, (anchor_pool, eligible) in specs.items():
        for _ in range(max_attempts):
            if len(pairs[category]) >= sample_size:
                break
            left = int(anchor_pool[int(rng.integers(anchor_pool.size))])
            mates = index.mate_rows(left)
            mates = mates[eligible(dominant_tail[mates])]
            if mates.size == 0:
                continue
            right = int(mates[int(rng.integers(mates.size))])
            pairs[category].add((min(left, right), max(left, right)))
    return {
        name: np.asarray(sorted(values), dtype=np.int64).reshape(-1, 2)
        for name, values in pairs.items()
    }


def pairs_sharing_two_or_more(codes, num_buckets):
    """Enumerate the sparse set of pairs matching in at least two hashes."""
    codes = np.asarray(codes, dtype=np.int64)
    pair_keys = []
    num_rows, num_hashes = codes.shape
    for first in range(num_hashes):
        for second in range(first + 1, num_hashes):
            key = codes[:, first] * int(num_buckets) + codes[:, second]
            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            boundaries = np.flatnonzero(np.diff(sorted_key)) + 1
            for group in np.split(order, boundaries):
                if group.size < 2:
                    continue
                for left_index in range(group.size - 1):
                    left = int(group[left_index])
                    for right in group[left_index + 1:]:
                        right = int(right)
                        pair_keys.append(
                            min(left, right) * num_rows + max(left, right)
                        )
    if not pair_keys:
        return np.empty((0, 2), dtype=np.int64)
    unique = np.unique(np.asarray(pair_keys, dtype=np.int64))
    return np.column_stack((unique // num_rows, unique % num_rows))


def stratify_and_sample_pairs(
    pairs, dominant_tail, zh_index, sample_size, seed
):
    rng = np.random.default_rng(seed)
    grouped = {name: [] for name in PAIR_CATEGORIES}
    for left, right in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
        category = _pair_category(
            dominant_tail[left], dominant_tail[right], zh_index
        )
        if category is not None:
            grouped[category].append((int(left), int(right)))
    output = {}
    for name, values in grouped.items():
        array = np.asarray(values, dtype=np.int64).reshape(-1, 2)
        if array.shape[0] > sample_size:
            array = array[rng.choice(array.shape[0], sample_size, replace=False)]
        output[name] = array
    return output


def sample_random_pairs(codes, dominant_tail, zh_index, sample_size, seed):
    """Language-matched random pairs that share no Product-Code coordinate."""
    codes = np.asarray(codes, dtype=np.int64)
    dominant_tail = np.asarray(dominant_tail)
    if codes.ndim != 2 or codes.shape[0] != dominant_tail.size:
        raise ValueError("codes and dominant_tail have incompatible shapes")
    rng = np.random.default_rng(seed)
    zh = np.flatnonzero(dominant_tail == zh_index)
    other = np.flatnonzero((dominant_tail >= 0) & (dominant_tail != zh_index))
    if zh.size < 2 or other.size < 2:
        raise ValueError("not enough zh/other tail rows for random pair baselines")

    specs = {
        "zh_zh": (zh, zh),
        "zh_other": (zh, other),
        "other_other": (other, other),
    }
    output = {}
    for name, (left_pool, right_pool) in specs.items():
        pairs = set()
        max_attempts = max(100_000, sample_size * 50)
        for _ in range(max_attempts):
            if len(pairs) >= sample_size:
                break
            left = int(left_pool[int(rng.integers(left_pool.size))])
            right = int(right_pool[int(rng.integers(right_pool.size))])
            if left == right or np.any(codes[left] == codes[right]):
                continue
            pairs.add((min(left, right), max(left, right)))
        if len(pairs) < sample_size:
            raise ValueError(
                f"could only draw {len(pairs)} of {sample_size} non-mate "
                f"random pairs for category {name}"
            )
        output[name] = np.asarray(sorted(pairs), dtype=np.int64)
    return output


def cosine_summary(table, tail_ids, pair_groups, chunk_size=4096):
    tail_ids = np.asarray(tail_ids, dtype=np.int64)
    output = {}
    for category, pairs in pair_groups.items():
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        values = []
        for start in range(0, pairs.shape[0], chunk_size):
            block = pairs[start:start + chunk_size]
            left = table[torch.from_numpy(tail_ids[block[:, 0]])]
            right = table[torch.from_numpy(tail_ids[block[:, 1]])]
            values.append(F.cosine_similarity(left, right, dim=1).numpy())
        output[category] = _summary(
            np.concatenate(values) if values else np.empty(0)
        )
    return output


def gate_group_summary(gates):
    gates = np.asarray(gates, dtype=np.float64)
    if gates.ndim != 2:
        raise ValueError("gates must be a matrix")
    if gates.shape[0] == 0:
        return {
            "num_tokens": 0,
            "gate_values": _summary([]),
            "mean_abs_offset": _summary([]),
            "fraction_any_below_0.1": None,
            "fraction_any_above_3": None,
        }
    return {
        "num_tokens": int(gates.shape[0]),
        "gate_values": _summary(gates),
        "mean_abs_offset": _summary(np.abs(gates - 1).mean(axis=1)),
        "fraction_any_below_0.1": float((gates < 0.1).any(axis=1).mean()),
        "fraction_any_above_3": float((gates > 3).any(axis=1).mean()),
    }


def state_diagnostics(
    embed,
    language_counts_path,
    importance_path,
    importance_key,
    bytoken_path,
    *,
    pair_samples,
    pair_seed,
    materialize_chunk_size,
    rank_populations,
):
    """Run Checks 4 and 5 from the checkpoint's exact tied embedding."""
    vocab_size = embed.vocab_size
    languages, normalized, dominant = load_language_counts(
        language_counts_path, vocab_size
    )
    if "zh" not in languages:
        raise ValueError("language counts do not contain counts_zh")
    zh_index = languages.index("zh")
    importance = load_importance(importance_path, importance_key, vocab_size)
    order = frequency_rank_order(importance)
    rank = np.empty(vocab_size, dtype=np.int64)
    rank[order] = np.arange(vocab_size)

    tail_ids = embed.tail_ids.detach().cpu().numpy()
    head_ids = embed.head_ids.detach().cpu().numpy()
    expected_head_ids = np.sort(order[:embed.head_size])
    if not np.array_equal(head_ids, expected_head_ids):
        raise ValueError(
            "checkpoint head_ids do not match the supplied importance artifact"
        )
    codes = embed.codes.detach().cpu().numpy()
    dominant_tail = dominant[tail_ids]
    gate_offsets = embed.gate_offsets.detach().float().cpu().numpy()
    gates = 1.0 + gate_offsets
    mate_index = BucketMateIndex(tail_ids, codes, embed.num_buckets)

    table = materialize_table(embed, materialize_chunk_size)
    norms = table.norm(dim=1).numpy()
    zh_tail = dominant_tail == zh_index
    other_tail = (dominant_tail >= 0) & ~zh_tail
    unknown_tail = dominant_tail < 0

    shared = sample_shared_pairs(
        mate_index, dominant_tail, zh_index, pair_samples, pair_seed
    )
    two_plus_all = pairs_sharing_two_or_more(codes, embed.num_buckets)
    two_plus = stratify_and_sample_pairs(
        two_plus_all, dominant_tail, zh_index, pair_samples, pair_seed + 1
    )
    random_pairs = sample_random_pairs(
        codes, dominant_tail, zh_index, pair_samples, pair_seed + 2
    )

    similarities = {
        "share_1plus_codes": cosine_summary(table, tail_ids, shared),
        "share_2plus_codes": cosine_summary(table, tail_ids, two_plus),
        "random_tail_pairs": cosine_summary(table, tail_ids, random_pairs),
    }

    population_sum = sum(rank_populations)
    if population_sum != vocab_size or any(value <= 0 for value in rank_populations):
        raise ValueError(
            f"rank populations must be positive and sum to {vocab_size}, "
            f"got {rank_populations} (sum={population_sum})"
        )
    rank_groups = {}
    start = 0
    for population in rank_populations:
        end = start + population
        mask = (rank[tail_ids] >= start) & (rank[tail_ids] < end)
        label = f"rank[{start}:{end})"
        rank_groups[label] = {
            "zh": gate_group_summary(gates[mask & zh_tail]),
            "other": gate_group_summary(gates[mask & other_tail]),
        }
        start = end

    deviation = np.abs(gates - 1).mean(axis=1)
    gate_correlations = {
        "importance_rank_vs_mean_abs_offset_all_tail": _pearson(
            rank[tail_ids], deviation
        ),
        "importance_rank_vs_mean_abs_offset_zh_tail": _pearson(
            rank[tail_ids][zh_tail], deviation[zh_tail]
        ),
        "importance_rank_vs_mean_abs_offset_other_tail": _pearson(
            rank[tail_ids][other_tail], deviation[other_tail]
        ),
        "per_language_bytoken": {},
    }
    if bytoken_path:
        bytoken = load_run(bytoken_path)
        gate_l2 = np.linalg.norm(gates, axis=1)
        for language, (nll_sum, count) in sorted(bytoken.items()):
            if nll_sum.shape != (vocab_size,):
                raise ValueError(
                    f"{bytoken_path}: {language} vocab size does not match checkpoint"
                )
            observed = count[tail_ids] > 0
            token_nll = nll_sum[tail_ids][observed] / count[tail_ids][observed]
            token_ppl = np.exp(np.minimum(token_nll, 700.0))
            gate_correlations["per_language_bytoken"][language] = {
                "num_observed_tail_types": int(observed.sum()),
                "num_token_ppl_clipped_at_exp_700": int((token_nll > 700).sum()),
                "mean_abs_offset_vs_token_log_ppl": _pearson(
                    deviation[observed], token_nll
                ),
                "mean_abs_offset_vs_token_ppl": _pearson(
                    deviation[observed], token_ppl
                ),
                "effective_gate_l2_vs_token_log_ppl": _pearson(
                    gate_l2[observed], token_nll
                ),
                "effective_gate_l2_vs_token_ppl": _pearson(
                    gate_l2[observed], token_ppl
                ),
            }

    return {
        "language_metadata": {
            "languages": languages,
            "unknown_dominant_language_types": int((dominant < 0).sum()),
            "zh_dominant_head_rows": int((dominant[head_ids] == zh_index).sum()),
            "zh_dominant_tail_rows": int(zh_tail.sum()),
            "other_dominant_tail_rows": int(other_tail.sum()),
            "unknown_dominant_tail_rows": int(unknown_tail.sum()),
            "head_token_mass_by_language": {
                language: float(normalized[index, head_ids].sum())
                for index, language in enumerate(languages)
            },
        },
        "row_norms": {
            "head_all": _summary(norms[head_ids]),
            "head_zh": _summary(norms[head_ids][dominant[head_ids] == zh_index]),
            "tail_zh": _summary(norms[tail_ids][zh_tail]),
            "tail_other": _summary(norms[tail_ids][other_tail]),
            "tail_unknown": _summary(norms[tail_ids][unknown_tail]),
        },
        "cosine_similarity": similarities,
        "pair_sampling": {
            "requested_per_category": pair_samples,
            "share_1plus_realized": {
                name: int(values.shape[0]) for name, values in shared.items()
            },
            "share_2plus_total_before_sampling": int(two_plus_all.shape[0]),
            "share_2plus_realized": {
                name: int(values.shape[0]) for name, values in two_plus.items()
            },
            "random_realized": {
                name: int(values.shape[0]) for name, values in random_pairs.items()
            },
            "strategy": (
                "1+ is token-centric anchor/mate sampling; 2+ enumerates exact "
                "pairwise code intersections; random pairs match language category "
                "and share no code coordinate"
            ),
        },
        "gates": {
            "zh_tail": gate_group_summary(gates[zh_tail]),
            "other_tail": gate_group_summary(gates[other_tail]),
            "unknown_tail": gate_group_summary(gates[unknown_tail]),
            "by_importance_tier": rank_groups,
            "correlations": gate_correlations,
        },
        "mate_index": mate_index,
    }


def _json_ready_state(state):
    state = dict(state)
    state.pop("mate_index", None)
    return state


def validate_hashed_tied_model(model, comp_config):
    """Return the exact diagnosed embedder or reject a mismatched checkpoint."""
    if comp_config.get("arm") != "product_code":
        raise ValueError(
            f"checkpoint arm is {comp_config.get('arm')!r}, expected 'product_code'"
        )
    if comp_config.get("product_code_assignment") != "hashed":
        raise ValueError(
            "checkpoint Product-Code assignment is "
            f"{comp_config.get('product_code_assignment')!r}, expected 'hashed'"
        )
    if comp_config.get("tie_output") is not True:
        raise ValueError("checkpoint must use an exactly tied output head")
    embed = model.model.embed_tokens.embed
    if not isinstance(embed, ProductCodeEmbed):
        raise TypeError(f"loaded embedding is {type(embed)}, not ProductCodeEmbed")
    if (
        not isinstance(model.lm_head, TiedProductCodeHead)
        or model.lm_head.embed is not embed
    ):
        raise TypeError(
            "loaded output head is not tied to the diagnosed Product-Code embedding"
        )
    seed = int(comp_config.get("product_code_seed", 0))
    expected_codes = hashed_codes(
        embed.tail_ids.detach().cpu(),
        embed.num_hashes,
        embed.num_buckets,
        seed=seed,
    )
    if not torch.equal(embed.codes.detach().cpu(), expected_codes):
        raise ValueError(
            "checkpoint codes do not match deterministic hashed assignment "
            f"for product_code_seed={seed}"
        )
    return embed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--language-counts", default="resources/token_freq_sample10.npz"
    )
    parser.add_argument(
        "--importance", default="resources/token_importance_langbalanced.npz"
    )
    parser.add_argument("--importance-key", default="counts")
    parser.add_argument(
        "--bytoken", default=None,
        help="Hashed eval_ppl_bytoken.npz from Check 1 (enables gate/PPL correlation)",
    )
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--langs", nargs="+", default=["zh", "de"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--logits-batch-size", type=int, default=64)
    parser.add_argument("--max-verification-gap", type=float, default=1e-3)
    parser.add_argument(
        "--max-positions", type=int, default=0,
        help="tail positions per language; 0 analyzes every position (production default)",
    )
    parser.add_argument("--pair-samples", type=int, default=20000)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--materialize-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--rank-populations", default="2048,6144,24576,119168"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stride = args.stride or args.block_size // 2
    if args.block_size <= 1 or not 0 < stride <= args.block_size:
        raise ValueError("require block_size > 1 and 0 < stride <= block_size")
    if args.max_positions < 0:
        raise ValueError("--max-positions must be nonnegative")
    if (
        args.max_verification_gap < 0
        or not math.isfinite(args.max_verification_gap)
    ):
        raise ValueError("--max-verification-gap must be finite and nonnegative")
    if args.pair_samples <= 0:
        raise ValueError("--pair-samples must be positive")
    if len(args.langs) != len(set(args.langs)):
        raise ValueError("--langs must not contain duplicates")
    rank_populations = tuple(
        int(value) for value in args.rank_populations.split(",") if value
    )

    dtype = torch.bfloat16 if args.bf16 else None
    load_device = args.device if args.eval_dir else "cpu"
    from compositional.loading import load_compositional_model
    model, comp_config = load_compositional_model(
        args.checkpoint, device=load_device, dtype=dtype
    )
    embed = validate_hashed_tied_model(model, comp_config)

    state = state_diagnostics(
        embed,
        args.language_counts,
        args.importance,
        args.importance_key,
        args.bytoken,
        pair_samples=args.pair_samples,
        pair_seed=args.pair_seed,
        materialize_chunk_size=args.materialize_chunk_size,
        rank_populations=rank_populations,
    )
    report = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "arm": comp_config.get("arm"),
            "assignment": comp_config.get("product_code_assignment"),
            "product_code_seed": int(comp_config.get("product_code_seed", 0)),
            "tie_output": comp_config.get("tie_output"),
            "vocab_size": embed.vocab_size,
            "embed_dim": embed.embed_dim,
            "head_size": embed.head_size,
            "tail_size": embed.tail_size,
            "num_hashes": embed.num_hashes,
            "num_buckets": embed.num_buckets,
            "language_counts": args.language_counts,
            "importance": args.importance,
            "importance_key": args.importance_key,
            "bytoken": args.bytoken,
            "rank_populations": rank_populations,
            "pair_samples": args.pair_samples,
            "pair_seed": args.pair_seed,
            "prediction_random_seed": args.random_seed,
            "random_baseline": "same size as mate union; excludes target and all mates",
        },
        "state": _json_ready_state(state),
    }

    if args.eval_dir:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name or args.checkpoint
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        report["metadata"]["tokenizer"] = validate_tokenizer_ids(
            tokenizer, embed.vocab_size
        )
        report["metadata"]["prediction_languages"] = args.langs
        report["metadata"]["block_size"] = args.block_size
        report["metadata"]["stride"] = stride
        report["metadata"]["max_positions"] = args.max_positions
        report["prediction_leakage"] = prediction_diagnostics(
            model,
            tokenizer,
            args.eval_dir,
            args.langs,
            embed,
            state["mate_index"],
            block_size=args.block_size,
            stride=stride,
            device=args.device,
            random_seed=args.random_seed,
            logits_batch_size=args.logits_batch_size,
            max_positions=args.max_positions,
            verification_gap_limit=args.max_verification_gap,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
