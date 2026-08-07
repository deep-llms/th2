"""Load a compositional checkpoint into a single model for evaluation.

Supports two checkpoint layouts:

1. HF Trainer checkpoint (the actual layout):
     output_dir/checkpoint-N/     — HF model files + embedding.pt
     output_dir/train_config.json — saved by save_train_config()

2. Standalone dir (if restructured):
     dir/                         — HF model files + embedding.pt + train_config.json

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
    V0Embed,
    V1Embed,
    V2Embed,
    IsolationControlEmbed,
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


def _build_arm_from_config(comp_config, vocab_size, embed_dim):
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

    if arm == "original_ant":
        return OriginalANT(vocab_size, K, embed_dim)
    if arm == "ant":
        return ANTEmbed(vocab_size, K, embed_dim, **shared, num_heads=num_heads)
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


def _infer_comp_config_from_state(state):
    """Infer arm + hyperparams from embedding.pt tensor names/shapes.

    Needed because train_config.json is only written when training finishes
    normally — runs killed at a target step (run_experiments.py) never write it.
    gamma is not recoverable from weights; the training default (1.0) is assumed.
    V0/V1 cannot be distinguished from weights alone and are not handled here.
    """
    keys = set(state.keys())

    if "T" in keys:
        return {"arm": "original_ant", "K": state["T"].shape[1]}

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
    else:
        cfg["arm"] = "ant"

    return cfg


def is_compositional(checkpoint_dir):
    """Check if a checkpoint is compositional (has embedding.pt)."""
    return os.path.isfile(os.path.join(checkpoint_dir, "embedding.pt"))


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

    config_path = _find_config_path(checkpoint_dir)
    if config_path is not None:
        with open(config_path) as f:
            full_config = json.load(f)
        comp_config = full_config["compositional"]
    else:
        comp_config = _infer_comp_config_from_state(state)
        print(f"  No train_config.json — inferred from embedding.pt: {comp_config}")

    config = AutoConfig.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, config=config, torch_dtype=dtype)

    embed = _build_arm_from_config(comp_config, config.vocab_size, config.hidden_size)
    embed.load_state_dict(state)

    if dtype is not None:
        embed = embed.to(dtype)

    model.model.embed_tokens = EmbeddingShim(embed)
    model.to(device)
    model.eval()

    return model, comp_config
