#!/usr/bin/env python3
"""Build a language-balanced token importance vector from per-language counts.

    imp(w) = sum_l counts_l(w) / sum_w counts_l(w)

Each language's counts are normalized to total mass 1 before summing, so the
ordering reflects the equal-language weighting of the mean-PPL objective
rather than the 85.7%-English training mix. Uses training counts only. The
output is accepted by ``load_frequency_counts`` (key ``counts``, float64) and
is the head-selection input for ``--arm product_code``.

Usage:
  python scripts/make_token_importance.py \
      --source resources/token_freq_sample10.npz \
      --output resources/token_importance_langbalanced.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compositional.compression_init import file_sha256  # noqa: E402
from compositional.product_code import head_tail_partition  # noqa: E402


def language_balanced_importance(archive, prefix="counts_"):
    langs = sorted(key[len(prefix):] for key in archive.files if key.startswith(prefix))
    if not langs:
        raise ValueError("no per-language count arrays found")
    total = None
    for lang in langs:
        counts = archive[f"{prefix}{lang}"].astype(np.float64)
        if counts.ndim != 1 or (counts < 0).any() or counts.sum() <= 0:
            raise ValueError(f"invalid counts for language {lang}")
        share = counts / counts.sum()
        total = share if total is None else total + share
    return langs, total / len(langs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="resources/token_freq_sample10.npz")
    parser.add_argument("--output", default="resources/token_importance_langbalanced.npz")
    parser.add_argument("--head-size", type=int, default=2048,
                        help="report language composition of this head size")
    args = parser.parse_args()

    with np.load(args.source) as archive:
        langs, importance = language_balanced_importance(archive)
        dominant = np.argmax(
            np.stack([archive[f"counts_{lang}"] for lang in langs]), axis=0
        )
        raw = archive["counts"].astype(np.float64)
    # Same partition rule the training arm applies, so the reported head is
    # exactly the trained head.
    head = head_tail_partition(importance, args.head_size)[0].numpy()
    raw_head = head_tail_partition(raw, args.head_size)[0].numpy()
    composition = {
        lang: int((dominant[head] == index).sum()) for index, lang in enumerate(langs)
    }
    raw_composition = {
        lang: int((dominant[raw_head] == index).sum()) for index, lang in enumerate(langs)
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, counts=importance)
    provenance = {
        "source": args.source,
        "source_sha256": file_sha256(args.source),
        "languages": langs,
        "formula": "mean over languages of counts_lang / sum(counts_lang)",
        "vocab_size": int(importance.size),
        "head_size_reported": args.head_size,
        "head_dominant_language_composition": composition,
        "raw_frequency_head_composition": raw_composition,
        "output_sha256": file_sha256(output),
    }
    with open(output.with_suffix(".json"), "w") as handle:
        json.dump(provenance, handle, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
