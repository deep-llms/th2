#!/usr/bin/env python3
"""Product-quantization tail codes from a trained dense embedding table.

Selects the dense head with the same language-balanced importance rule as
``--arm product_code``, then runs per-sub-vector k-means on the remaining
(tail) rows of the dense checkpoint's embedding table. The output is the
``--product_code_codes_path`` artifact for ``--product_code_assignment pq``.

Fidelity boundary: the resulting run is *post-hoc-informed* from-scratch
training (codes derived from a dense teacher). Report it as such.

Usage (on the node that holds the dense checkpoint):
  python scripts/make_pq_codes.py \
      --checkpoint /path/to/dense_tied_baseline/checkpoint-10000 \
      --importance resources/token_importance_langbalanced.npz \
      --head-size 2048 --num-hashes 4 --num-buckets 4096 \
      --output resources/pq_codes_h2048_4x4096.pt --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compositional.compression_init import (  # noqa: E402
    file_sha256,
    load_frequency_counts,
    tensor_sha256,
)
from compositional.product_code import head_tail_partition, pq_codes  # noqa: E402

TABLE_KEYS = ("model.embed_tokens.weight", "lm_head.weight")


def load_dense_table(checkpoint_dir):
    from safetensors import safe_open

    path = os.path.join(checkpoint_dir, "model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in TABLE_KEYS:
            if key in keys:
                return handle.get_tensor(key).float(), key
    raise KeyError(f"{path} has none of {TABLE_KEYS}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--importance", default="resources/token_importance_langbalanced.npz")
    parser.add_argument("--importance-key", default="counts")
    parser.add_argument("--head-size", type=int, default=2048)
    parser.add_argument("--num-hashes", type=int, default=4)
    parser.add_argument("--num-buckets", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity-factor", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    table, table_key = load_dense_table(args.checkpoint)
    vocab_size, embed_dim = table.shape
    importance = load_frequency_counts(
        args.importance, vocab_size, key=args.importance_key, pseudocount=0.0
    )
    head_ids, tail_ids = head_tail_partition(importance, args.head_size)

    tail_table = table[tail_ids].to(args.device)
    codes = pq_codes(
        tail_table, args.num_hashes, args.num_buckets,
        iters=args.iters, seed=args.seed, capacity_factor=args.capacity_factor,
    ).cpu()
    occupancy = torch.stack([
        torch.bincount(codes[:, index], minlength=args.num_buckets)
        for index in range(args.num_hashes)
    ])
    provenance = {
        "checkpoint": args.checkpoint,
        "table_key": table_key,
        "table_sha256": tensor_sha256(table),
        "importance": args.importance,
        "importance_sha256": file_sha256(args.importance),
        "head_size": args.head_size,
        "num_hashes": args.num_hashes,
        "num_buckets": args.num_buckets,
        "iters": args.iters,
        "seed": args.seed,
        "capacity_factor": args.capacity_factor,
        "tail_size": int(tail_ids.numel()),
        "max_bucket_occupancy": int(occupancy.max()),
        "empty_buckets": int((occupancy == 0).sum()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"codes": codes, "tail_ids": tail_ids, "head_ids": head_ids,
                "provenance": provenance}, output)
    provenance["output_sha256"] = file_sha256(output)
    with open(output.with_suffix(".json"), "w") as handle:
        json.dump(provenance, handle, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
