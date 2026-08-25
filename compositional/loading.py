"""Load a compositional checkpoint into a single model for evaluation.

Supports two checkpoint layouts:

1. HF Trainer checkpoint (the actual layout):
     output_dir/checkpoint-N/     — HF model files + embedding.pt
                                      (+ output_head.pt for independent LR output)
     output_dir/train_config.json — saved by save_train_config()

2. Standalone dir (if restructured):
     dir/                         — HF model files + embedding.pt + train_config.json
                                      (+ output_head.pt for independent LR output)

In layout 1, pass the checkpoint dir as output_dir and config_path separately.
In layout 2, everything is in one dir.
"""

import json
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .embeddings import (
    OriginalANT,
    ANTEmbed,
    ResidualANTEmbed,
    V0Embed,
    V1Embed,
    V2Embed,
    IsolationControlEmbed,
    LowRankEmbed,
    SharedLocalEmbed,
    PureLocalEmbed,
)
from .compressed_baselines import (
    PVQEmbed,
    SlimEmbed,
    GroupReduceEmbed,
    TTEmbedding,
    balanced_exact_modes,
    balanced_padded_modes,
)
from .tied_head import (
    INDEPENDENT_OUTPUT_FILENAME,
    IndependentLowRankHead,
)


class EmbeddingShim(nn.Module):
    """Wraps a compositional embedding so it can be installed as embed_tokens.

    nn.Embedding.forward(input_ids) -> (B, L, d)
    Compositional.forward(input_ids) -> (e, theta)
    This shim returns only e, making model(input_ids) work transparently.
    """

    def __init__(self, embed_module):
        super().__init__()
        self.embed = embed_module

    def forward(self, input_ids):
        e, _ = self.embed(input_ids)
        return e


def _parse_int_list(value):
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(part) for part in value)


def _build_arm_from_config(comp_config, vocab_size, embed_dim, state=None):
    """Rebuild the embedding module from saved train_config.json."""
    tc = comp_config
    arm = tc["arm"]
    shared = dict(
        d_x=tc.get("d_x", 128),
        d_k=tc.get("d_k", 64),
        gamma=tc.get("gamma", 1.0),
    )
    K = tc.get("K", 4096)
    num_heads = tc.get("num_heads", 1)

    if arm in ("lowrank", "global_lowrank"):
        return LowRankEmbed(vocab_size, embed_dim, rank=tc.get("d_x", 128))
    if arm == "shared_local":
        return SharedLocalEmbed(
            vocab_size,
            embed_dim,
            shared_rank=tc.get("shared_rank", 64),
            local_rank=tc.get("local_embed_rank", tc.get("local_rank", 64)),
            num_groups=tc.get("num_groups", 4),
        )
    if arm == "pure_local":
        return PureLocalEmbed(
            vocab_size,
            embed_dim,
            rank=tc.get("pure_local_rank", tc.get("local_rank", 128)),
            num_groups=tc.get("num_groups", 16),
        )
    if arm == "pvq":
        assignments = None if state is None else state.get("assignments")
        return PVQEmbed(
            vocab_size,
            embed_dim,
            shared_dim=tc.get("pvq_shared_dim", 768),
            num_codes=tc.get("pvq_num_codes", 128),
            assignments=assignments,
            assignment_seed=tc.get("pvq_assignment_seed", 42),
        )
    if arm == "slim":
        mapping = None if state is None else state.get("mapping")
        return SlimEmbed(
            vocab_size,
            embed_dim,
            num_components=tc.get("slim_num_components", 8),
            num_subvectors=tc.get("slim_num_subvectors", 0) or vocab_size,
            mapping=mapping,
            mapping_seed=tc.get("slim_mapping_seed", 42),
        )
    if arm == "groupreduce":
        if state is not None:
            group_ids = state.get("group_ids")
            ranks = []
            group = 0
            while f"left_factors.{group}" in state:
                ranks.append(state[f"left_factors.{group}"].shape[1])
                group += 1
        else:
            group_ids = None
            ranks = _parse_int_list(tc.get("groupreduce_ranks", ""))
            if not ranks:
                ranks = (tc.get("d_x", 128),) * tc.get(
                    "groupreduce_num_groups", 4
                )
        return GroupReduceEmbed(
            vocab_size,
            embed_dim,
            group_ranks=ranks,
            group_ids=group_ids,
        )
    if arm == "tt":
        if state is not None and any(
            key.startswith("cores.") for key in state
        ):
            core_keys = sorted(
                (key for key in state if key.startswith("cores.")),
                key=lambda key: int(key.split(".")[1]),
            )
            shapes = [state[key].shape for key in core_keys]
            vocab_modes = tuple(shape[1] for shape in shapes)
            embedding_modes = tuple(shape[2] for shape in shapes)
            ranks = (shapes[0][0],) + tuple(shape[3] for shape in shapes)
        else:
            order = tc.get("tt_order", 3)
            vocab_modes = _parse_int_list(tc.get("tt_vocab_shape", ""))
            embedding_modes = _parse_int_list(
                tc.get("tt_embedding_shape", "")
            )
            vocab_modes = vocab_modes or balanced_padded_modes(vocab_size, order)
            embedding_modes = embedding_modes or balanced_exact_modes(embed_dim, order)
            ranks = _parse_int_list(tc.get("tt_ranks", "")) or tc.get(
                "tt_rank", 128
            )
        return TTEmbedding(
            vocab_size,
            embed_dim,
            vocab_modes=vocab_modes,
            embedding_modes=embedding_modes,
            tt_ranks=ranks,
            target_std=(tc.get("tt_target_std", 0.0) or None),
            implementation=tc.get("tt_implementation", "materialize"),
        )
    if arm == "original_ant":
        return OriginalANT(vocab_size, K, embed_dim)
    if arm == "ant":
        return ANTEmbed(vocab_size, K, embed_dim, **shared, num_heads=num_heads)
    if arm == "residual_ant":
        return ResidualANTEmbed(vocab_size, K, embed_dim, **shared, num_heads=num_heads)
    if arm == "v0":
        return V0Embed(vocab_size, K, embed_dim, **shared,
                       max_k=tc.get("max_k", 16), mode=tc.get("v0_mode", "post"))
    if arm == "v1":
        return V1Embed(vocab_size, K, embed_dim, **shared,
                       max_k=tc.get("max_k", 16), query=tc.get("v1_query", "content"))
    if arm == "v2":
        return V2Embed(vocab_size, K, embed_dim, **shared,
                       num_heads=num_heads, localenc=tc.get("localenc", "attn"))
    if arm == "isolation_control":
        return IsolationControlEmbed(vocab_size, K, embed_dim, **shared,
                                     num_heads=num_heads, localenc=tc.get("localenc", "attn"))
    raise ValueError(f"Unknown arm: {arm}")


def _find_config_path(checkpoint_dir):
    """Find train_config.json — in checkpoint dir or parent."""
    local = os.path.join(checkpoint_dir, "train_config.json")
    if os.path.isfile(local):
        return local
    parent = os.path.join(os.path.dirname(checkpoint_dir), "train_config.json")
    if os.path.isfile(parent):
        return parent
    return None


def _checkpoint_model_state_keys(checkpoint_dir):
    """Read saved tensor names without materializing safetensor weights."""
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors import safe_open
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            return set(handle.keys())

    safetensors_index = os.path.join(
        checkpoint_dir, "model.safetensors.index.json"
    )
    if os.path.isfile(safetensors_index):
        with open(safetensors_index) as handle:
            return set(json.load(handle)["weight_map"])

    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(pytorch_path):
        return set(torch.load(
            pytorch_path, map_location="cpu", weights_only=True
        ))

    pytorch_index = os.path.join(
        checkpoint_dir, "pytorch_model.bin.index.json"
    )
    if os.path.isfile(pytorch_index):
        with open(pytorch_index) as handle:
            return set(json.load(handle)["weight_map"])
    return set()


def _load_checkpoint_tensors(checkpoint_dir, tensor_names):
    """Load selected HF checkpoint tensors, including from sharded states."""
    requested = set(tensor_names)
    if not requested:
        return {}

    loaded = {}
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors import safe_open
        with safe_open(
            safetensors_path, framework="pt", device="cpu"
        ) as handle:
            available = set(handle.keys())
            missing = requested - available
            if missing:
                raise FileNotFoundError(
                    "HF checkpoint is missing custom tensors: "
                    f"{sorted(missing)}"
                )
            return {name: handle.get_tensor(name) for name in requested}

    safetensors_index = os.path.join(
        checkpoint_dir, "model.safetensors.index.json"
    )
    if os.path.isfile(safetensors_index):
        with open(safetensors_index) as handle:
            weight_map = json.load(handle)["weight_map"]
        missing = requested - set(weight_map)
        if missing:
            raise FileNotFoundError(
                "HF checkpoint is missing custom tensors: "
                f"{sorted(missing)}"
            )
        tensors_by_shard = {}
        for name in requested:
            tensors_by_shard.setdefault(weight_map[name], []).append(name)
        from safetensors import safe_open
        for shard_name, names in tensors_by_shard.items():
            with safe_open(
                os.path.join(checkpoint_dir, shard_name),
                framework="pt",
                device="cpu",
            ) as handle:
                loaded.update({name: handle.get_tensor(name) for name in names})
        return loaded

    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(pytorch_path):
        state = torch.load(
            pytorch_path, map_location="cpu", weights_only=True
        )
        missing = requested - set(state)
        if missing:
            raise FileNotFoundError(
                "HF checkpoint is missing custom tensors: "
                f"{sorted(missing)}"
            )
        return {name: state[name] for name in requested}

    pytorch_index = os.path.join(
        checkpoint_dir, "pytorch_model.bin.index.json"
    )
    if os.path.isfile(pytorch_index):
        with open(pytorch_index) as handle:
            weight_map = json.load(handle)["weight_map"]
        missing = requested - set(weight_map)
        if missing:
            raise FileNotFoundError(
                "HF checkpoint is missing custom tensors: "
                f"{sorted(missing)}"
            )
        tensors_by_shard = {}
        for name in requested:
            tensors_by_shard.setdefault(weight_map[name], []).append(name)
        for shard_name, names in tensors_by_shard.items():
            shard_state = torch.load(
                os.path.join(checkpoint_dir, shard_name),
                map_location="cpu",
                weights_only=True,
            )
            loaded.update({name: shard_state[name] for name in names})
        return loaded

    raise FileNotFoundError(
        f"No supported HF model state in {checkpoint_dir}"
    )


def _validate_sidecar_values(checkpoint_dir, state, prefix, filename):
    """Require a sidecar to be an exact duplicate of its saved HF tensors."""
    full_names = {prefix + key for key in state}
    hf_state = _load_checkpoint_tensors(checkpoint_dir, full_names)
    mismatches = []
    for key, sidecar_tensor in state.items():
        hf_tensor = hf_state[prefix + key]
        if not torch.equal(sidecar_tensor, hf_tensor):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            f"{filename} does not match the corresponding HF checkpoint "
            f"tensors: {sorted(mismatches)}"
        )


def _infer_comp_config_from_state(state):
    """Infer arm + hyperparams from embedding.pt tensor names/shapes.

    Supports legacy, manually copied, or interrupted checkpoints whose
    train_config.json is absent. Gamma is not recoverable from weights; the
    training default (1.0) is assumed. V0/V1 cannot be distinguished from
    weights alone and are not handled here.
    """
    keys = set(state.keys())

    if {"codebook", "exclusive", "assignments"}.issubset(keys):
        return {
            "arm": "pvq",
            "pvq_shared_dim": state["codebook"].shape[1],
            "pvq_num_codes": state["codebook"].shape[0],
        }

    if {"subvectors", "mapping"}.issubset(keys):
        return {
            "arm": "slim",
            "slim_num_components": state["subvectors"].shape[0],
            "slim_num_subvectors": (
                state["subvectors"].shape[0]
                * state["subvectors"].shape[1]
            ),
        }

    if "group_ids" in keys and any(
        key.startswith("left_factors.") for key in keys
    ):
        left_keys = sorted(
            (key for key in keys if key.startswith("left_factors.")),
            key=lambda key: int(key.split(".")[1]),
        )
        return {
            "arm": "groupreduce",
            "groupreduce_num_groups": len(left_keys),
            "groupreduce_ranks": ",".join(
                str(state[key].shape[1]) for key in left_keys
            ),
        }

    if "cores.0" in keys:
        core_keys = sorted(
            (key for key in keys if key.startswith("cores.")),
            key=lambda key: int(key.split(".")[1]),
        )
        shapes = [state[key].shape for key in core_keys]
        return {
            "arm": "tt",
            "tt_order": len(shapes),
            "tt_vocab_shape": ",".join(str(shape[1]) for shape in shapes),
            "tt_embedding_shape": ",".join(str(shape[2]) for shape in shapes),
            "tt_ranks": ",".join(str(shape[3]) for shape in shapes[:-1]),
        }

    if "T" in keys:
        return {"arm": "original_ant", "K": state["T"].shape[1]}

    if "local_weight" in keys and "shared_proj.weight" in keys:
        local_rank = state["local_weight"].shape[-1]
        return {
            "arm": "shared_local",
            "shared_rank": state["token_factors"].shape[-1] - local_rank,
            "local_embed_rank": local_rank,
            "num_groups": state["local_weight"].shape[0],
        }

    if {"token_factors", "local_weight", "bias"}.issubset(keys):
        return {
            "arm": "pure_local",
            "pure_local_rank": state["local_weight"].shape[-1],
            "num_groups": state["local_weight"].shape[0],
        }

    if "proj.weight" in keys:  # LowRankEmbed: X + proj, no codebook A
        return {"arm": "lowrank", "d_x": state["X"].shape[1]}

    cfg = {"K": state["A"].shape[0], "gamma": 1.0}
    if "X" in keys:
        cfg["d_x"] = state["X"].shape[1]
    if "W_q" in keys:
        cfg["d_k"] = state["W_q"].shape[1]
        cfg["num_heads"] = 1
    elif "W_q_mh" in keys:
        cfg["d_k"] = state["W_q_mh"].shape[2]
        cfg["num_heads"] = state["W_q_mh"].shape[0]

    if any(k.startswith("localenc.") for k in keys):
        if "localenc.Wq_a" in keys:
            cfg["localenc"] = "attn"
        elif "localenc.convs.0.weight" in keys:
            cfg["localenc"] = "conv"
        else:
            cfg["localenc"] = "conv_lite"
        cfg["arm"] = "isolation_control" if "W_ctl" in keys else "v2"
    elif "Wq_sat" in keys:
        raise ValueError(
            "V0/V1 checkpoints cannot be identified from weights alone — "
            "provide train_config.json")
    elif "W_up.weight" in keys:
        cfg["arm"] = "residual_ant"
    else:
        cfg["arm"] = "ant"

    return cfg


def is_compositional(checkpoint_dir):
    """Recognize custom checkpoints, including damaged ones, before loading.

    A missing ``embedding.pt`` must not make evaluation silently fall back to
    a stock model with randomly initialized native embeddings. Sidecar/config
    evidence routes such checkpoints through ``load_compositional_model``,
    which then raises a precise missing-artifact error.
    """
    if os.path.isfile(os.path.join(checkpoint_dir, "embedding.pt")):
        return True
    if os.path.isfile(os.path.join(
        checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME
    )):
        return True
    config_path = _find_config_path(checkpoint_dir)
    if config_path is not None:
        try:
            with open(config_path) as handle:
                if isinstance(
                    json.load(handle).get("compositional"), dict
                ):
                    return True
        except (OSError, ValueError, AttributeError):
            pass

    # Last-resort corruption detection: custom registered tensors survive in
    # the HF model state even if both sidecars and train_config.json were lost.
    # Route that checkpoint through the strict compositional loader so it fails
    # for the missing artifacts instead of evaluating random native weights.
    try:
        model_keys = _checkpoint_model_state_keys(checkpoint_dir)
    except (OSError, ValueError, KeyError):
        return False
    return (
        any(key.startswith("model.embed_tokens.embed.") for key in model_keys)
        or any(key in {"lm_head.X", "lm_head.proj_weight"}
               for key in model_keys)
    )


def load_compositional_model(checkpoint_dir, device="cuda", dtype=None):
    """Load a compositional checkpoint as a ready-to-use model.

    Args:
        checkpoint_dir: Path to the checkpoint directory containing model files
                        and embedding.pt. The arm config comes from
                        train_config.json (checkpoint dir or parent) when
                        present, otherwise it is inferred from embedding.pt.
        device: Target device.
        dtype: Parameter dtype (default: from config).

    Returns:
        (model, comp_config) where model(input_ids) works normally.
    """
    embedding_path = os.path.join(checkpoint_dir, "embedding.pt")
    if not os.path.isfile(embedding_path):
        raise FileNotFoundError(f"No embedding.pt in {checkpoint_dir}")

    state = torch.load(embedding_path, map_location="cpu", weights_only=True)
    output_head_path = os.path.join(
        checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME
    )
    has_independent_output = os.path.isfile(output_head_path)

    config_path = _find_config_path(checkpoint_dir)
    if config_path is not None:
        with open(config_path) as f:
            full_config = json.load(f)
        comp_config = full_config["compositional"]
        configured_independent = comp_config.get(
            "independent_lowrank_output", False
        )
        if has_independent_output and not configured_independent:
            raise ValueError(
                f"Found {INDEPENDENT_OUTPUT_FILENAME}, but train_config.json "
                "does not declare an independent low-rank output head"
            )
    else:
        comp_config = _infer_comp_config_from_state(state)
        if has_independent_output:
            output_state = torch.load(
                output_head_path, map_location="cpu", weights_only=True
            )
            if set(output_state) != {"X", "proj_weight"}:
                raise ValueError(
                    f"Unrecognized independent output state in {output_head_path}: "
                    f"{sorted(output_state)}"
                )
            comp_config.update({
                "tie_output": False,
                "independent_lowrank_output": True,
                "output_rank": output_state["X"].shape[1],
            })

    _validate_sidecar_values(
        checkpoint_dir,
        state,
        "model.embed_tokens.embed.",
        "embedding.pt",
    )
    if has_independent_output:
        output_integrity_state = torch.load(
            output_head_path, map_location="cpu", weights_only=True
        )
        _validate_sidecar_values(
            checkpoint_dir,
            output_integrity_state,
            "lm_head.",
            INDEPENDENT_OUTPUT_FILENAME,
        )

    config = AutoConfig.from_pretrained(checkpoint_dir)
    load_kwargs = dict(config=config, torch_dtype=dtype)
    # Always inspect the stock-model load report. Custom heads appear as
    # unexpected keys while the native lm_head is missing; validating that
    # topology prevents stale config/sidecars from silently selecting a random
    # dense or tied head.
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, output_loading_info=True, **load_kwargs
    )
    unexpected_head_keys = {
        key for key in loading_info["unexpected_keys"]
        if key.startswith("lm_head.")
    }
    expected_independent_keys = {"lm_head.X", "lm_head.proj_weight"}
    if unexpected_head_keys and unexpected_head_keys != expected_independent_keys:
        raise ValueError(
            "Unrecognized output-head tensors in HF checkpoint: "
            f"{sorted(unexpected_head_keys)}"
        )
    hf_has_independent_output = (
        unexpected_head_keys == expected_independent_keys
    )
    hf_missing_native_output = any(
        key == "lm_head.weight" for key in loading_info["missing_keys"]
    )

    if config_path is None:
        # A tied checkpoint intentionally has no lm_head tensors. Loading info
        # lets old/interrupted checkpoints recover tie_output, while explicit
        # independent keys require their sidecar rather than being mistaken for
        # a tied head.
        if hf_has_independent_output and not has_independent_output:
            raise FileNotFoundError(
                "HF checkpoint contains an independent low-rank output head "
                f"but {INDEPENDENT_OUTPUT_FILENAME} is missing"
            )
        if has_independent_output and not hf_has_independent_output:
            raise ValueError(
                f"{INDEPENDENT_OUTPUT_FILENAME} exists, but the HF checkpoint "
                "does not contain the matching independent head topology"
            )
        supported_tied_arms = {
            "lowrank", "global_lowrank", "shared_local", "pure_local",
            "pvq", "slim", "groupreduce", "tt", "original_ant", "ant",
            "residual_ant"
        }
        if not has_independent_output:
            comp_config["tie_output"] = (
                comp_config["arm"] in supported_tied_arms
                and hf_missing_native_output
            )
        print(f"  No train_config.json — inferred from checkpoint: {comp_config}")
    else:
        configured_tied = comp_config.get("tie_output", False)
        configured_independent = comp_config.get(
            "independent_lowrank_output", False
        )
        if configured_independent:
            if not hf_has_independent_output or not hf_missing_native_output:
                raise ValueError(
                    "train_config.json requests independent low-rank output, "
                    "but the HF checkpoint has a different head topology"
                )
        elif configured_tied:
            if hf_has_independent_output or not hf_missing_native_output:
                raise ValueError(
                    "train_config.json requests tied output, but the HF "
                    "checkpoint has a different head topology"
                )
        elif hf_has_independent_output or hf_missing_native_output:
            raise ValueError(
                "train_config.json requests a native dense output, but the HF "
                "checkpoint has a compressed/missing head"
            )

    embed = _build_arm_from_config(
        comp_config, config.vocab_size, config.hidden_size, state=state
    )
    embed.load_state_dict(state)

    # The compositional weights live in a separate embedding.pt, so unlike the
    # backbone they are not cast by from_pretrained().  Match the dtype that HF
    # actually selected for the model when the caller leaves dtype unspecified
    # (for example, a BF16 checkpoint whose config records BF16).  Otherwise the
    # first projection in the embedding fails with a Float/BFloat16 mismatch.
    model_dtype = next(model.parameters()).dtype
    embed = embed.to(dtype=dtype if dtype is not None else model_dtype)

    model.model.embed_tokens = EmbeddingShim(embed)

    uses_independent_output = comp_config.get(
        "independent_lowrank_output", False
    )
    if comp_config.get("tie_output", False) and uses_independent_output:
        raise ValueError(
            "Checkpoint config cannot request both tied and independent output"
        )

    if uses_independent_output:
        if comp_config["arm"] != "lowrank":
            raise ValueError(
                "Independent low-rank output currently requires arm=lowrank"
            )
        if not has_independent_output:
            raise FileNotFoundError(
                f"Checkpoint requests independent low-rank output but has no "
                f"{INDEPENDENT_OUTPUT_FILENAME}: {checkpoint_dir}"
            )
        independent_head = IndependentLowRankHead(embed)
        expected_rank = comp_config.get("output_rank", embed.X.shape[1])
        if expected_rank != embed.X.shape[1]:
            raise ValueError(
                f"Independent output rank {expected_rank} does not match input "
                f"rank {embed.X.shape[1]}"
            )
        output_state = torch.load(
            output_head_path, map_location="cpu", weights_only=True
        )
        independent_head.load_state_dict(output_state, strict=True)
        independent_head = independent_head.to(
            dtype=dtype if dtype is not None else model_dtype
        )
        model.lm_head = independent_head
    elif comp_config.get("tie_output", False):
        from .tied_head import make_tied_head
        model.lm_head = make_tied_head(embed, comp_config["arm"],
                                       config.vocab_size)

    model.to(device)
    model.eval()

    return model, comp_config
