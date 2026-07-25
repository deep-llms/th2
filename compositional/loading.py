"""Load a compositional checkpoint into a single model for evaluation.

The training script saves the backbone and embedding separately:
  output_dir/backbone/       — HF model (embed_tokens missing)
  output_dir/embedding.pt    — compositional embedding state_dict
  output_dir/train_config.json — training args (tells us which arm was used)

This module rebuilds the full model by:
  1. Loading the backbone
  2. Rebuilding the embedding module from the saved config
  3. Loading embedding.pt
  4. Installing an embed_tokens shim so model(input_ids) works transparently
     (including with lm_eval benchmarks that call model(input_ids) internally)
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


def _build_arm_from_config(train_config, vocab_size, embed_dim):
    """Rebuild the embedding module from saved train_config.json."""
    tc = train_config
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


def load_compositional_model(output_dir, device="cuda", dtype=None):
    """Load a compositional checkpoint as a ready-to-use model.

    Args:
        output_dir: The training output directory containing backbone/,
                    embedding.pt, and train_config.json.
        device: Target device.
        dtype: Parameter dtype for the backbone (default: from config).

    Returns:
        (model, tokenizer, train_config) where model(input_ids) works normally.
    """
    backbone_dir = os.path.join(output_dir, "backbone")
    embedding_path = os.path.join(output_dir, "embedding.pt")
    config_path = os.path.join(output_dir, "train_config.json")

    if not os.path.isdir(backbone_dir):
        raise FileNotFoundError(f"No backbone/ directory in {output_dir}")
    if not os.path.isfile(embedding_path):
        raise FileNotFoundError(f"No embedding.pt in {output_dir}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No train_config.json in {output_dir}")

    with open(config_path) as f:
        full_config = json.load(f)
    train_config = full_config["training"]

    config = AutoConfig.from_pretrained(backbone_dir)
    model = AutoModelForCausalLM.from_pretrained(backbone_dir, config=config, torch_dtype=dtype)

    embed = _build_arm_from_config(train_config, config.vocab_size, config.hidden_size)
    state = torch.load(embedding_path, map_location="cpu", weights_only=True)
    embed.load_state_dict(state)

    if dtype is not None:
        embed = embed.to(dtype)

    model.model.embed_tokens = EmbeddingShim(embed)
    model.to(device)
    model.eval()

    return model, train_config
