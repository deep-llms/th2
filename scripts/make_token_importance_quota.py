#!/usr/bin/env python3
"""Build the deterministic Hashed-v2 coverage-quota head ordering.

The first ``--v1-head-size`` entries reproduce the language-balanced v1
head.  Remaining observed tokens are greedily assigned to the language with
the lowest covered token mass, using that language's highest-share unselected
token.  Tokens unseen in every language are placed last by token id.

Usage:
  python scripts/make_token_importance_quota.py \
      --source resources/token_freq_sample10.npz \
      --v1-importance resources/token_importance_langbalanced.npz \
      --output resources/token_importance_quota.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_LANGUAGES = ("ar", "de", "en", "ru", "vi", "zh")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an artifact without importing the torch-dependent package."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_descending_order(values: np.ndarray) -> np.ndarray:
    """Return ``(-value, token_id)`` order for a validated vector."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("importance/count shares must be one-dimensional")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("importance/count shares must be finite and nonnegative")
    token_ids = np.arange(values.size, dtype=np.int64)
    return np.lexsort((token_ids, -values)).astype(np.int64, copy=False)


def load_language_shares(source: str | Path, prefix: str = "counts_"):
    """Load per-language training counts and normalize each to unit mass."""
    with np.load(source) as archive:
        languages = sorted(
            key[len(prefix):] for key in archive.files if key.startswith(prefix)
        )
        if not languages:
            raise ValueError(f"{source} has no arrays beginning with {prefix!r}")
        if tuple(languages) != EXPECTED_LANGUAGES:
            raise ValueError(
                "coverage quota requires exactly the six training languages "
                f"{EXPECTED_LANGUAGES}, got {tuple(languages)}"
            )
        arrays = []
        vocab_size = None
        for language in languages:
            counts = np.asarray(archive[f"{prefix}{language}"], dtype=np.float64)
            if counts.ndim != 1:
                raise ValueError(f"counts for {language} must be one-dimensional")
            if vocab_size is None:
                vocab_size = counts.size
            elif counts.size != vocab_size:
                raise ValueError("all per-language count arrays must have one shape")
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError(f"invalid counts for language {language}")
            total = float(counts.sum())
            if total <= 0:
                raise ValueError(f"counts for language {language} have zero mass")
            arrays.append(counts / total)
    return languages, np.stack(arrays, axis=0)


def load_importance(path: str | Path, key: str, vocab_size: int) -> np.ndarray:
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"{path} has no array named {key!r}")
        importance = np.asarray(archive[key], dtype=np.float64)
    if importance.shape != (vocab_size,):
        raise ValueError(
            f"v1 importance must have shape ({vocab_size},), got {importance.shape}"
        )
    if not np.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("v1 importance must be finite and nonnegative")
    return importance


def coverage_quota_order(shares: np.ndarray, v1_importance: np.ndarray,
                         v1_head_size: int = 2048) -> np.ndarray:
    """Construct the floor-quota ordering from normalized language shares."""
    shares = np.asarray(shares, dtype=np.float64)
    if shares.ndim != 2:
        raise ValueError("shares must have shape (num_languages, vocab_size)")
    num_languages, vocab_size = shares.shape
    if num_languages <= 0 or vocab_size <= 0:
        raise ValueError("shares must contain languages and vocabulary entries")
    if not np.isfinite(shares).all() or (shares < 0).any():
        raise ValueError("shares must be finite and nonnegative")
    totals = shares.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"each language share vector must sum to one, got {totals}")
    v1_importance = np.asarray(v1_importance, dtype=np.float64)
    if v1_importance.shape != (vocab_size,):
        raise ValueError("v1 importance shape does not match the shares")
    v1_head_size = int(v1_head_size)
    if not 0 < v1_head_size < vocab_size:
        raise ValueError("v1_head_size must be between 1 and vocabulary size - 1")

    v1_order = _stable_descending_order(v1_importance)
    initial = v1_order[:v1_head_size]
    selected = np.zeros(vocab_size, dtype=np.bool_)
    selected[initial] = True
    ordering = initial.tolist()
    coverage = shares[:, initial].sum(axis=1)

    # Each row is independently stable by descending within-language share,
    # with token id breaking ties.  Pointers only move forward, so the full
    # greedy pass is O(LV), not O(V^2).
    language_orders = [
        _stable_descending_order(shares[index]) for index in range(num_languages)
    ]
    pointers = np.zeros(num_languages, dtype=np.int64)
    positive_anywhere = np.any(shares > 0, axis=0)
    if not positive_anywhere[initial].all():
        raise ValueError("the preserved v1 head unexpectedly contains zero-mass tokens")
    remaining_positive = int(np.count_nonzero(positive_anywhere & ~selected))

    while remaining_positive:
        eligible = []
        for language in range(num_languages):
            order = language_orders[language]
            pointer = int(pointers[language])
            while pointer < vocab_size:
                token_id = int(order[pointer])
                if not selected[token_id] and shares[language, token_id] > 0:
                    break
                pointer += 1
            pointers[language] = pointer
            if pointer < vocab_size:
                eligible.append(language)
        if not eligible:
            raise RuntimeError(
                f"no language can place the {remaining_positive} positive-mass tokens"
            )

        # Language index is the deterministic tie break; callers construct it
        # from sorted language names.
        language = min(eligible, key=lambda index: (coverage[index], index))
        token_id = int(language_orders[language][pointers[language]])
        pointers[language] += 1
        selected[token_id] = True
        ordering.append(token_id)
        coverage += shares[:, token_id]
        remaining_positive -= 1

    # Unobserved vocabulary entries cannot change any coverage.  Keep them
    # last and stable by token id.
    ordering.extend(np.flatnonzero(~selected).tolist())
    ordering = np.asarray(ordering, dtype=np.int64)

    if ordering.shape != (vocab_size,) or np.unique(ordering).size != vocab_size:
        raise AssertionError("quota ordering is not a permutation of the vocabulary")
    if not np.array_equal(ordering[:v1_head_size], initial):
        raise AssertionError("quota ordering does not preserve the ordered v1 head")
    positive_count = int(np.count_nonzero(positive_anywhere))
    if not positive_anywhere[ordering[:positive_count]].all():
        raise AssertionError("a zero-mass token was placed before an observed token")
    expected_zero_tail = np.flatnonzero(~positive_anywhere)
    if not np.array_equal(ordering[positive_count:], expected_zero_tail):
        raise AssertionError("zero-mass tokens are not last in token-id order")
    return ordering


def importance_from_order(ordering: np.ndarray) -> np.ndarray:
    """Encode an exact permutation as strictly descending float64 values."""
    ordering = np.asarray(ordering, dtype=np.int64)
    vocab_size = ordering.size
    if ordering.ndim != 1 or np.unique(ordering).size != vocab_size:
        raise ValueError("ordering must be a one-dimensional permutation")
    if ordering.min(initial=0) < 0 or ordering.max(initial=-1) >= vocab_size:
        raise ValueError("ordering has out-of-range token ids")
    importance = np.empty(vocab_size, dtype=np.float64)
    importance[ordering] = np.arange(vocab_size, 0, -1, dtype=np.float64)
    return importance


def _token_id_sha256(token_ids: np.ndarray) -> str:
    values = np.asarray(token_ids, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_provenance(languages, shares, ordering, head_sizes, *, source,
                     v1_importance, v1_head_size, output):
    coverage_table = {}
    composition_table = {}
    dominant = np.argmax(shares, axis=0)
    previous = np.zeros(len(languages), dtype=np.float64)
    for head_size in head_sizes:
        coverage = shares[:, ordering[:head_size]].sum(axis=1)
        if np.any(coverage + 1e-15 < previous):
            raise AssertionError("language coverage decreased with a larger head")
        previous = coverage
        coverage_table[str(head_size)] = {
            language: float(value)
            for language, value in zip(languages, coverage)
        }
        counts = np.bincount(dominant[ordering[:head_size]], minlength=len(languages))
        composition_table[str(head_size)] = {
            language: int(value)
            for language, value in zip(languages, counts)
        }

    v1_head = ordering[:v1_head_size]
    return {
        "algorithm": "coverage_floor_quota_v1_head_preserving",
        "algorithm_version": 1,
        "source": str(source),
        "source_sha256": file_sha256(source),
        "v1_importance_source": str(v1_importance),
        "v1_importance_sha256": file_sha256(v1_importance),
        "output": str(output),
        "languages": list(languages),
        "vocab_size": int(ordering.size),
        "v1_head_size": int(v1_head_size),
        "v1_ordered_prefix_sha256_int64_le": _token_id_sha256(v1_head),
        "v1_head_ids_sorted_sha256_int64_le": _token_id_sha256(np.sort(v1_head)),
        "reported_head_sizes": [int(value) for value in head_sizes],
        "coverage_by_head_size": coverage_table,
        "dominant_language_definition": (
            "argmax(counts_lang / sum(counts_lang)); language-name tie break"
        ),
        "head_dominant_language_composition": composition_table,
        "first_placed_token_ids": ordering[:32].tolist(),
        "last_placed_token_ids": ordering[-32:].tolist(),
        "zero_mass_token_count": int(np.count_nonzero(~np.any(shares > 0, axis=0))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="resources/token_freq_sample10.npz")
    parser.add_argument(
        "--v1-importance",
        default="resources/token_importance_langbalanced.npz",
    )
    parser.add_argument("--v1-key", default="counts")
    parser.add_argument("--v1-head-size", type=int, default=2048)
    parser.add_argument(
        "--report-head-sizes", type=int, nargs="+", default=[2048, 4096, 6144, 8192]
    )
    parser.add_argument("--output", default="resources/token_importance_quota.npz")
    args = parser.parse_args()

    languages, shares = load_language_shares(args.source)
    vocab_size = shares.shape[1]
    v1_importance = load_importance(args.v1_importance, args.v1_key, vocab_size)
    head_sizes = sorted(set(int(value) for value in args.report_head_sizes))
    if args.v1_head_size not in head_sizes:
        raise ValueError("--report-head-sizes must include --v1-head-size")
    if any(value <= 0 or value >= vocab_size for value in head_sizes):
        raise ValueError("reported head sizes must be between 1 and vocabulary size - 1")

    ordering = coverage_quota_order(shares, v1_importance, args.v1_head_size)
    importance = importance_from_order(ordering)
    if not np.array_equal(_stable_descending_order(importance), ordering):
        raise AssertionError("importance values do not reconstruct the quota ordering")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, counts=importance)
    provenance = build_provenance(
        languages,
        shares,
        ordering,
        head_sizes,
        source=args.source,
        v1_importance=args.v1_importance,
        v1_head_size=args.v1_head_size,
        output=output,
    )
    provenance["output_sha256"] = file_sha256(output)
    json_path = output.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
