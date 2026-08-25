#!/usr/bin/env python3
"""Convert a dense tied checkpoint into a strict compact baseline artifact.

This utility performs the topology-changing boundary after dense pretraining
(and, for faithful P-VQ, after its dense clustering curriculum).  The output is
a plain embedding state dictionary accepted by
``train_compositional.py --embedding_init_path``.  A JSON file beside it records
source provenance and every reproduction decision.

Examples
--------
P-VQ using final assignments saved by the curriculum::

    python scripts/convert_dense_baseline.py \
      --method pvq --checkpoint /path/to/dense-curriculum/checkpoint-10000 \
      --output /path/to/pvq_init.pt --pvq_shared_dim 768 --pvq_num_codes 128 \
      --pvq_assignments /path/to/pvq_curriculum_state.pt

GroupReduce weighted block-SVD with paper-style refinement::

    python scripts/convert_dense_baseline.py \
      --method groupreduce --checkpoint /path/to/dense/checkpoint-10000 \
      --output /path/to/groupreduce_init.pt \
      --frequency_path resources/token_freq_sample10.npz \
      --groupreduce_num_groups 20 --target_params 19579904
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compositional.compression_init import (
    allocate_frequency_proportional_ranks,
    capacity_constrained_kmeans,
    frequency_group_ids,
    group_parameter_count,
    initialize_groupreduce_from_dense,
    initialize_pvq_from_dense,
    load_frequency_counts,
    refine_groupreduce_from_dense,
    tensor_sha256,
)


def _parse_ranks(value):
    if not value:
        return ()
    try:
        ranks = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("--groupreduce_ranks must be comma-separated integers") from error
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("all GroupReduce ranks must be positive")
    return ranks


def _load_assignment_tensor(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if torch.is_tensor(value):
        return value.to(torch.long)
    if not isinstance(value, dict):
        raise ValueError("assignment artifact must be a tensor or state dictionary")
    for key in ("assignments", "code_assignments", "final_assignments"):
        if key in value and torch.is_tensor(value[key]):
            return value[key].to(torch.long)
    raise ValueError(
        "assignment artifact has no assignments/code_assignments/final_assignments tensor"
    )


def _load_dense_tied_table(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if not getattr(config, "tie_word_embeddings", False):
        raise ValueError(
            "Source checkpoint config does not declare tie_word_embeddings=true; "
            "it is not the dense tied source required by these conversions"
        )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    if output_module is None or not hasattr(output_module, "weight"):
        raise ValueError("source model has no conventional output embedding weight")
    output_weight = output_module.weight
    if input_weight.shape != output_weight.shape:
        raise ValueError("source input/output table shapes differ")
    if input_weight.data_ptr() != output_weight.data_ptr() and not torch.equal(
        input_weight, output_weight
    ):
        raise ValueError("source input/output table values are not exactly tied")
    dense = input_weight.detach().float().cpu().contiguous()
    del model
    return dense, config


def _pvq(args, dense):
    if not 0 < args.pvq_shared_dim < dense.size(1):
        raise ValueError("--pvq_shared_dim must be strictly between 0 and hidden size")
    if not 0 < args.pvq_num_codes <= dense.size(0):
        raise ValueError("--pvq_num_codes must be between 1 and vocabulary size")

    if args.pvq_assignments:
        assignments = _load_assignment_tensor(args.pvq_assignments)
        assignment_method = "provided_final_curriculum_assignments"
    else:
        device = args.cluster_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA clustering requested but CUDA is unavailable")
        _, assignments = capacity_constrained_kmeans(
            dense[:, :args.pvq_shared_dim],
            args.pvq_num_codes,
            num_iters=args.cluster_iters,
            num_restarts=args.cluster_restarts,
            seed=args.seed,
            chunk_size=args.cluster_chunk_size,
            device=device,
        )
        assignment_method = (
            "scalable_capacity_repair_kmeans_not_original_hungarian_balanced_kmeans"
        )
    if assignments.shape != (dense.size(0),):
        raise ValueError(
            f"P-VQ assignments must have shape ({dense.size(0)},), got "
            f"{tuple(assignments.shape)}"
        )
    counts = torch.bincount(assignments, minlength=args.pvq_num_codes)
    if counts.numel() != args.pvq_num_codes or torch.any(counts == 0):
        raise ValueError("P-VQ assignments contain an empty or invalid code")
    state = initialize_pvq_from_dense(
        dense,
        args.pvq_shared_dim,
        args.pvq_num_codes,
        assignments,
    )
    metadata = {
        "variant": "pvq_compact",
        "shared_dim": args.pvq_shared_dim,
        "num_codes": args.pvq_num_codes,
        "assignment_method": assignment_method,
        "cluster_size_min": int(counts.min().item()),
        "cluster_size_max": int(counts.max().item()),
        "float_parameters": sum(
            tensor.numel() for key, tensor in state.items()
            if key != "assignments"
        ),
        "assignment_entries": assignments.numel(),
    }
    return state, metadata


def _groupreduce(args, dense):
    if not args.frequency_path:
        raise ValueError("GroupReduce conversion requires --frequency_path")
    counts = load_frequency_counts(
        args.frequency_path,
        dense.size(0),
        key=args.frequency_key,
        pseudocount=args.frequency_pseudocount,
    )
    group_ids = frequency_group_ids(counts, args.groupreduce_num_groups)
    ranks = _parse_ranks(args.groupreduce_ranks)
    if ranks:
        if len(ranks) != args.groupreduce_num_groups:
            raise ValueError("provide exactly one GroupReduce rank per group")
    else:
        if args.target_params <= 0:
            raise ValueError(
                "provide --target_params or explicit --groupreduce_ranks"
            )
        ranks = allocate_frequency_proportional_ranks(
            counts,
            group_ids,
            dense.size(1),
            args.target_params,
        )

    if args.groupreduce_refine_iters > 0:
        state, final_group_ids, history = refine_groupreduce_from_dense(
            dense.to(args.groupreduce_device),
            counts.to(args.groupreduce_device),
            group_ids.to(args.groupreduce_device),
            ranks,
            max_iters=args.groupreduce_refine_iters,
            move_fraction=args.groupreduce_move_fraction,
            min_candidates=args.groupreduce_min_candidates,
            chunk_size=args.cluster_chunk_size,
        )
    else:
        state = initialize_groupreduce_from_dense(
            dense.to(args.groupreduce_device),
            counts.to(args.groupreduce_device),
            group_ids,
            ranks,
        )
        final_group_ids = group_ids
        history = []
    final_sizes = torch.bincount(
        final_group_ids, minlength=args.groupreduce_num_groups
    )
    metadata = {
        "variant": "groupreduce_posthoc_tied_adaptation",
        "source_method_note": (
            "Original GroupReduce compressed input and output separately; this "
            "artifact applies its weighted block-SVD to one exactly tied table."
        ),
        "frequency_path": os.path.abspath(args.frequency_path),
        "frequency_key": args.frequency_key,
        "frequency_pseudocount": args.frequency_pseudocount,
        "num_groups": args.groupreduce_num_groups,
        "ranks": list(ranks),
        "group_sizes": final_sizes.tolist(),
        "float_parameters": group_parameter_count(
            final_group_ids, ranks, dense.size(1)
        ),
        "refinement_iterations": history,
    }
    return state, metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("pvq", "groupreduce"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pvq_shared_dim", type=int, default=768)
    parser.add_argument("--pvq_num_codes", type=int, default=128)
    parser.add_argument("--pvq_assignments")
    parser.add_argument("--cluster_device", default="cuda")
    parser.add_argument("--cluster_iters", type=int, default=20)
    parser.add_argument("--cluster_restarts", type=int, default=1)
    parser.add_argument("--cluster_chunk_size", type=int, default=4096)

    parser.add_argument("--frequency_path")
    parser.add_argument("--frequency_key", default="counts")
    parser.add_argument("--frequency_pseudocount", type=float, default=1.0)
    parser.add_argument("--groupreduce_num_groups", type=int, default=20)
    parser.add_argument("--groupreduce_ranks", default="")
    parser.add_argument("--target_params", type=int, default=0)
    parser.add_argument("--groupreduce_refine_iters", type=int, default=5)
    parser.add_argument("--groupreduce_move_fraction", type=float, default=0.10)
    parser.add_argument("--groupreduce_min_candidates", type=int, default=1)
    parser.add_argument("--groupreduce_device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    dense, config = _load_dense_tied_table(args.checkpoint)
    if args.method == "pvq":
        state, method_metadata = _pvq(args, dense)
    else:
        state, method_metadata = _groupreduce(args, dense)

    metadata = {
        "format_version": 1,
        "method": args.method,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_table_shape": list(dense.shape),
        "source_table_sha256": tensor_sha256(dense),
        "source_tie_word_embeddings": bool(config.tie_word_embeddings),
        "seed": args.seed,
        **method_metadata,
    }
    torch.save(state, output)
    metadata_path = output.with_suffix(output.suffix + ".json")
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"saved {output}")
    print(f"saved {metadata_path}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
