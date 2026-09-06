"""Variable private codes with a common feature width and one tied projection."""

import logging
import math
import struct

import torch
from torch import nn
from torch.nn import functional as F

from .compression_init import (
    file_sha256, frequency_group_ids_from_populations, load_frequency_counts,
)
from .nonlinear_factorizations import _parameter_free_rms_norm


class UnifiedRankLiftEmbed(nn.Module):
    """e_i = P [z_i, SiLU(A_g RMS(z_i)) * B_g RMS(z_i)] + b.

    Groups vary only in private code/lift widths. All features have width m;
    the head assembles V x m features in token order, never grouped N x V
    logits. There are no per-group final projections or cross-step caches.
    """

    def __init__(self, vocab_size, embed_dim, code_dims=(384, 256, 128, 112),
                 feature_dim=460, group_ids=None, rms_eps=1e-6):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.feature_dim = int(feature_dim)
        self.code_dims = tuple(int(c) for c in code_dims)
        self.rms_eps = float(rms_eps)
        if min(self.vocab_size, self.embed_dim, self.feature_dim) <= 0:
            raise ValueError("Unified RankLift dimensions must be positive")
        if not self.code_dims or any(c <= 0 or c > self.feature_dim
                                     for c in self.code_dims):
            raise ValueError("code_dims must be nonempty and in [1, feature_dim]")
        if not math.isfinite(self.rms_eps) or self.rms_eps <= 0:
            raise ValueError("rms_eps must be finite and positive")
        self.num_groups = len(self.code_dims)
        self.lift_dims = tuple(self.feature_dim - c for c in self.code_dims)
        if group_ids is None:
            raise ValueError("Unified RankLift requires explicit group_ids")
        groups = torch.as_tensor(group_ids).detach().cpu().clone()
        if groups.dtype not in (torch.int32, torch.int64):
            raise ValueError("group_ids must contain integers")
        groups = groups.long()
        if groups.shape != (self.vocab_size,):
            raise ValueError("group_ids shape does not match vocabulary")
        if groups.min() < 0 or groups.max() >= self.num_groups:
            raise ValueError("group_ids out of range")
        sizes = torch.bincount(groups, minlength=self.num_groups)
        if torch.any(sizes == 0):
            raise ValueError("every Unified RankLift group must be used")
        self.group_sizes = tuple(sizes.tolist())
        order = torch.argsort(groups, stable=True)
        inverse = torch.argsort(order)
        offsets = torch.empty_like(groups)
        start = 0
        for size in self.group_sizes:
            offsets[order[start:start + size]] = torch.arange(size)
            start += size
        for name, value in (("group_ids", groups), ("group_offsets", offsets),
                            ("token_ids_by_group", order),
                            ("inverse_grouped_order", inverse),
                            ("unified_code_dims", torch.tensor(self.code_dims)),
                            ("unified_rms_eps_bits", torch.tensor(
                                struct.unpack("<q", struct.pack("<d", self.rms_eps))[0],
                                dtype=torch.int64))):
            self.register_buffer(name, value, persistent=True)
        self.token_codes = nn.ParameterList([
            nn.Parameter(torch.empty(n, c))
            for n, c in zip(self.group_sizes, self.code_dims)
        ])
        self.lift_a = nn.ModuleList([
            nn.Linear(c, q, bias=True) if q else nn.Identity()
            for c, q in zip(self.code_dims, self.lift_dims)
        ])
        self.lift_b = nn.ModuleList([
            nn.Linear(c, q, bias=True) if q else nn.Identity()
            for c, q in zip(self.code_dims, self.lift_dims)
        ])
        self.projection = nn.Linear(self.feature_dim, self.embed_dim, bias=True)
        self.reset_parameters()
        self.register_load_state_dict_post_hook(
            lambda module, incompatible: module.validate_structure())

    def reset_parameters(self):
        for codes in self.token_codes:
            nn.init.normal_(codes, std=0.02)
        for layer in [*self.lift_a, *self.lift_b, self.projection]:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def validate_structure(self):
        if tuple(self.unified_code_dims.cpu().tolist()) != self.code_dims:
            raise ValueError("Unified RankLift code dimensions disagree with metadata")
        bits = int(self.unified_rms_eps_bits.item())
        if struct.unpack("<d", struct.pack("<q", bits))[0] != self.rms_eps:
            raise ValueError("Unified RankLift rms_eps disagrees with metadata")
        if self.group_ids.min() < 0 or self.group_ids.max() >= self.num_groups:
            raise ValueError("Unified RankLift group_ids out of range")
        sizes = tuple(torch.bincount(self.group_ids, minlength=self.num_groups).tolist())
        if sizes != self.group_sizes:
            raise ValueError("Unified RankLift group populations disagree with parameters")
        order = torch.argsort(self.group_ids, stable=True)
        if not torch.equal(order, self.token_ids_by_group):
            raise ValueError("Unified RankLift grouped token order is inconsistent")
        if not torch.equal(torch.argsort(order), self.inverse_grouped_order):
            raise ValueError("Unified RankLift inverse token order is inconsistent")
        offsets = torch.empty_like(self.group_offsets)
        start = 0
        for group, size in enumerate(sizes):
            offsets[order[start:start + size]] = torch.arange(size, device=order.device)
            if self.token_codes[group].shape != (size, self.code_dims[group]):
                raise ValueError("Unified RankLift private-code shape mismatch")
            start += size
        if not torch.equal(offsets, self.group_offsets):
            raise ValueError("Unified RankLift token offsets are inconsistent")

    @staticmethod
    def structure_from_state(state):
        required = {"group_ids", "unified_code_dims", "unified_rms_eps_bits",
                    "projection.weight", "projection.bias", "token_codes.0"}
        if not required.issubset(state):
            raise ValueError("Unified RankLift checkpoint lacks structural tensors")
        weight = state["projection.weight"]
        groups = state["group_ids"].cpu()
        codes = tuple(state["unified_code_dims"].cpu().tolist())
        if weight.ndim != 2 or groups.ndim != 1 or not codes:
            raise ValueError("invalid Unified RankLift checkpoint structure")
        if groups.dtype != torch.int64 or groups.numel() == 0 \
                or groups.min() < 0 or groups.max() >= len(codes):
            raise ValueError("invalid Unified RankLift checkpoint group_ids")
        bits = int(state["unified_rms_eps_bits"].item())
        return dict(vocab_size=groups.numel(), embed_dim=weight.shape[0],
                    feature_dim=weight.shape[1], code_dims=codes, group_ids=groups,
                    group_sizes=tuple(torch.bincount(groups, minlength=len(codes)).tolist()),
                    rms_eps=struct.unpack("<d", struct.pack("<q", bits))[0])

    def features_from_codes(self, group, codes):
        if self.lift_dims[group] == 0:
            return codes
        u = _parameter_free_rms_norm(codes, self.rms_eps)
        lifted = F.silu(self.lift_a[group](u)) * self.lift_b[group](u)
        return torch.cat((codes, lifted), dim=-1)

    def materialize_features(self):
        grouped = torch.cat([
            self.features_from_codes(g, codes)
            for g, codes in enumerate(self.token_codes)
        ], dim=0)
        return grouped.index_select(0, self.inverse_grouped_order)

    def materialize_effective_table(self):
        """Reference only: the output head must use materialize_features instead."""
        return self.projection(self.materialize_features())

    def forward(self, input_ids, doc_mask=None):
        ids = input_ids.reshape(-1)
        # Embedding lookup rejects negative/out-of-range IDs; tensor indexing
        # would silently wrap negative token IDs to the vocabulary tail.
        groups = F.embedding(ids, self.group_ids[:, None]).squeeze(-1)
        offsets = F.embedding(ids, self.group_offsets[:, None]).squeeze(-1)
        features = self.token_codes[0].new_empty((ids.numel(), self.feature_dim))
        for g, codes in enumerate(self.token_codes):
            mask = groups == g
            selected = F.embedding(offsets[mask], codes)
            features[mask] = self.features_from_codes(g, selected)
        # One input projection, too. Empty groups stay in the autograd graph.
        output = self.projection(features)
        return output.view(*input_ids.shape, self.embed_dim), None


def _ints(value):
    return tuple(int(x) for x in (value.split(",") if isinstance(value, str) else value))


def build_unified_ranklift(config, vocab_size, embed_dim, state=None):
    """Shared train/resume/eval constructor; saved membership wins on reload."""
    prefix = "unified_ranklift_"
    if state is not None:
        structure = UnifiedRankLiftEmbed.structure_from_state(state)
        if (structure["vocab_size"], structure["embed_dim"]) != (vocab_size, embed_dim):
            raise ValueError("Unified RankLift checkpoint model dimensions mismatch")
        for key, field, parse in (("code_dims", "code_dims", _ints),
                                  ("feature_dim", "feature_dim", int),
                                  ("populations", "group_sizes", _ints),
                                  ("rms_eps", "rms_eps", float)):
            if prefix + key in config and parse(config[prefix + key]) != structure[field]:
                raise ValueError(f"Unified RankLift config {key} differs from checkpoint structure")
        return UnifiedRankLiftEmbed(
            vocab_size, embed_dim, code_dims=structure["code_dims"],
            feature_dim=structure["feature_dim"], group_ids=structure["group_ids"],
            rms_eps=structure["rms_eps"],
        )
    codes = _ints(config.get(prefix + "code_dims", "384,256,128,112"))
    populations = _ints(config.get(prefix + "populations", "2048,6144,24576,119168"))
    if len(codes) != len(populations):
        raise ValueError("Unified RankLift needs one population per code dimension")
    path = config.get(prefix + "frequency_path", "resources/token_freq_sample10.npz")
    if not path:
        raise ValueError("fresh Unified RankLift requires a frequency artifact")
    counts = load_frequency_counts(path, vocab_size,
                                   key=config.get(prefix + "frequency_key", "counts"),
                                   pseudocount=0.0)
    groups = frequency_group_ids_from_populations(counts, populations)
    logging.getLogger(__name__).info("Unified RankLift frequency artifact: %s (sha256=%s)",
                                     path, file_sha256(path))
    return UnifiedRankLiftEmbed(
        vocab_size, embed_dim, code_dims=codes,
        feature_dim=config.get(prefix + "feature_dim", 460), group_ids=groups,
        rms_eps=config.get(prefix + "rms_eps", 1e-6),
    )
