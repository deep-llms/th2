"""Published compressed vocabulary parameterizations used as baselines.

The modules in this file expose the same small interface as the project's
other embedding implementations::

    embedding, auxiliary = module(input_ids)

``auxiliary`` is always ``None`` because these methods have no learned router.
Each module represents a fixed, context-independent effective table, so an
exactly tied output head can evaluate ``hidden @ effective_table.T`` without
owning a second copy of the parameters.  The corresponding efficient output
algebra lives in :mod:`compositional.tied_head`.

The implementations deliberately distinguish an architecture from the
procedure used to obtain it:

* P-VQ's published procedure starts from a pretrained dense table, repeatedly
  quantizes its shared coordinates during a curriculum, and only then switches
  to :class:`PVQEmbed`.  Random assignments are available for unit tests and
  explicitly-labelled from-scratch adaptations, but are not a reproduction of
  the published training protocol.
* GroupReduce is a post-hoc weighted block-SVD method.  This module stores and
  trains the resulting block factors; the clustering/SVD conversion is kept
  separate from the forward algebra.
* Slim fixes a random, approximately balanced sub-vector mapping before
  training.  The mapping is a persistent buffer because it is part of the
  model even though it is not a learned floating-point parameter.
* TTEmbedding follows the TT-matrix convention from Khrulkov et al. (2020),
  including the variance-corrected core initialization from their Eq. (3).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _as_long_vector(values, *, name, length=None):
    tensor = torch.as_tensor(values, dtype=torch.long).detach().clone()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {tensor.shape}")
    if length is not None and tensor.numel() != length:
        raise ValueError(
            f"{name} must contain {length} entries, got {tensor.numel()}"
        )
    return tensor


def _balanced_random_codes(vocab_size, num_codes, seed):
    """Return fixed random codes whose usage differs by at most one.

    A private CPU generator makes the structural mapping reproducible without
    advancing the global initialization RNG used by learned parameters.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    codes = torch.arange(vocab_size, dtype=torch.long).remainder(num_codes)
    return codes[torch.randperm(vocab_size, generator=generator)]


class PVQEmbed(nn.Module):
    """Compact Partial Vector Quantization (P-VQ) embedding.

    For token ``i`` with fixed code ``c_i`` the effective embedding is

        e_i = concat(codebook[c_i], exclusive[i]).

    This is exactly the compact form in Zhang et al. (AAAI 2021).  It preserves
    one exclusive vector per token, avoiding identical full token embeddings.
    The published method obtains ``assignments`` with balanced k-means after a
    dense pretraining/curriculum phase.  ``assignments=None`` creates a fixed
    balanced-random mapping only so the compact architecture can be tested or
    used as an explicitly labelled from-scratch adaptation.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        shared_dim,
        num_codes,
        assignments=None,
        assignment_seed=42,
    ):
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 1:
            raise ValueError("vocab_size must be positive and embed_dim > 1")
        if not 0 < shared_dim < embed_dim:
            raise ValueError(
                f"shared_dim must be in [1, {embed_dim - 1}], got {shared_dim}"
            )
        if not 0 < num_codes <= vocab_size:
            raise ValueError(
                f"num_codes must be in [1, {vocab_size}], got {num_codes}"
            )

        if assignments is None:
            assignments = _balanced_random_codes(
                vocab_size, num_codes, assignment_seed
            )
        else:
            assignments = _as_long_vector(
                assignments, name="assignments", length=vocab_size
            )
        if assignments.numel() and (
            assignments.min().item() < 0
            or assignments.max().item() >= num_codes
        ):
            raise ValueError(
                f"assignments must be in [0, {num_codes - 1}]"
            )
        used = torch.bincount(assignments, minlength=num_codes)
        if torch.any(used == 0):
            missing = torch.where(used == 0)[0].tolist()
            raise ValueError(f"every P-VQ code must be used; missing {missing}")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.shared_dim = int(shared_dim)
        self.exclusive_dim = int(embed_dim - shared_dim)
        self.num_codes = int(num_codes)

        self.codebook = nn.Parameter(
            torch.randn(num_codes, shared_dim) * 0.02
        )
        self.exclusive = nn.Parameter(
            torch.randn(vocab_size, self.exclusive_dim) * 0.02
        )
        self.register_buffer("assignments", assignments, persistent=True)
        self.register_load_state_dict_post_hook(
            lambda module, incompatible: module.validate_structure()
        )

    def validate_structure(self):
        if self.assignments.shape != (self.vocab_size,):
            raise ValueError(
                f"P-VQ assignments must have shape ({self.vocab_size},)"
            )
        if self.assignments.min().item() < 0 \
                or self.assignments.max().item() >= self.num_codes:
            raise ValueError("P-VQ assignment is outside the codebook")
        if torch.any(torch.bincount(
            self.assignments, minlength=self.num_codes
        ) == 0):
            raise ValueError("P-VQ state contains an unused code")

    def forward(self, input_ids, doc_mask=None):
        shared = self.codebook[self.assignments[input_ids]]
        exclusive = self.exclusive[input_ids]
        return torch.cat((shared, exclusive), dim=-1), None

    @torch.no_grad()
    def initialize_from_dense(self, dense_weight, assignments=None):
        """Initialize compact parameters from a dense effective table.

        For fixed assignments, arithmetic centroids minimize the squared
        reconstruction error of the shared coordinates.  To reproduce the
        paper, pass assignments produced by its balanced-k-means curriculum;
        this method intentionally does not pretend that one-shot clustering is
        equivalent to that curriculum.
        """
        dense_weight = torch.as_tensor(dense_weight)
        expected = (self.vocab_size, self.embed_dim)
        if tuple(dense_weight.shape) != expected:
            raise ValueError(
                f"dense_weight must have shape {expected}, got "
                f"{tuple(dense_weight.shape)}"
            )
        if assignments is not None:
            assignments = _as_long_vector(
                assignments, name="assignments", length=self.vocab_size
            ).to(self.assignments.device)
            if assignments.min().item() < 0 or assignments.max().item() >= self.num_codes:
                raise ValueError(
                    f"assignments must be in [0, {self.num_codes - 1}]"
                )
            counts = torch.bincount(assignments, minlength=self.num_codes)
            if torch.any(counts == 0):
                raise ValueError("every P-VQ code must receive at least one token")
            self.assignments.copy_(assignments)

        source = dense_weight.to(
            device=self.codebook.device, dtype=self.codebook.dtype
        )
        shared = source[:, :self.shared_dim]
        centroids = torch.zeros_like(self.codebook)
        centroids.index_add_(0, self.assignments, shared)
        counts = torch.bincount(
            self.assignments, minlength=self.num_codes
        ).to(device=centroids.device, dtype=centroids.dtype)
        centroids.div_(counts.unsqueeze(1))
        self.codebook.copy_(centroids)
        self.exclusive.copy_(source[:, self.shared_dim:])


class SlimEmbed(nn.Module):
    """Slim Embedding Layers with fixed random sub-vector sharing.

    The hidden dimension is divided into ``num_components`` coordinate blocks.
    Component ``j`` has its own disjoint codebook and every token stores one
    fixed random code for that component.  Concatenating the selected
    sub-vectors gives the token's effective embedding.

    ``num_subvectors`` is the total number of learned sub-vectors across all
    component codebooks (``M`` in Li et al., AAAI 2018).  It must be divisible
    by ``num_components`` so each position-specific set has equal size.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_components=8,
        num_subvectors=None,
        mapping=None,
        mapping_seed=42,
    ):
        super().__init__()
        if vocab_size <= 0 or embed_dim <= 0:
            raise ValueError("vocab_size and embed_dim must be positive")
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        if embed_dim % num_components != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by "
                f"num_components={num_components}"
            )
        if num_subvectors is None:
            num_subvectors = vocab_size
        if num_subvectors < num_components:
            raise ValueError(
                "num_subvectors must be at least num_components"
            )
        if num_subvectors % num_components != 0:
            raise ValueError(
                f"num_subvectors={num_subvectors} must be divisible by "
                f"num_components={num_components}"
            )

        codes_per_component = num_subvectors // num_components
        if codes_per_component > vocab_size:
            raise ValueError(
                "codes per Slim component cannot exceed vocab_size because "
                "the paper's balanced mapping uses every learned subvector"
            )
        if mapping is None:
            columns = [
                _balanced_random_codes(
                    vocab_size, codes_per_component, mapping_seed + component
                )
                for component in range(num_components)
            ]
            mapping = torch.stack(columns, dim=1)
        else:
            mapping = torch.as_tensor(mapping, dtype=torch.long).detach().clone()
            expected = (vocab_size, num_components)
            if tuple(mapping.shape) != expected:
                raise ValueError(
                    f"mapping must have shape {expected}, got {tuple(mapping.shape)}"
                )
        if mapping.numel() and (
            mapping.min().item() < 0
            or mapping.max().item() >= codes_per_component
        ):
            raise ValueError(
                f"mapping entries must be in [0, {codes_per_component - 1}]"
            )
        for component in range(num_components):
            usage = torch.bincount(
                mapping[:, component], minlength=codes_per_component
            )
            if torch.any(usage == 0):
                raise ValueError(
                    f"Slim mapping component {component} leaves learned "
                    "subvectors unused"
                )

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_components = int(num_components)
        self.num_subvectors = int(num_subvectors)
        self.codes_per_component = int(codes_per_component)
        self.component_dim = int(embed_dim // num_components)

        self.subvectors = nn.Parameter(
            torch.randn(
                num_components, codes_per_component, self.component_dim
            ) * 0.02
        )
        self.register_buffer("mapping", mapping, persistent=True)
        self.register_load_state_dict_post_hook(
            lambda module, incompatible: module.validate_structure()
        )

    def validate_structure(self):
        expected = (self.vocab_size, self.num_components)
        if tuple(self.mapping.shape) != expected:
            raise ValueError(f"Slim mapping must have shape {expected}")
        if self.mapping.min().item() < 0 \
                or self.mapping.max().item() >= self.codes_per_component:
            raise ValueError("Slim mapping entry is outside its component codebook")
        for component in range(self.num_components):
            if torch.any(torch.bincount(
                self.mapping[:, component],
                minlength=self.codes_per_component,
            ) == 0):
                raise ValueError(
                    f"Slim state leaves component {component} subvectors unused"
                )

    def forward(self, input_ids, doc_mask=None):
        pieces = [
            self.subvectors[component, self.mapping[input_ids, component]]
            for component in range(self.num_components)
        ]
        return torch.cat(pieces, dim=-1), None


def make_contiguous_groups(vocab_size, num_groups):
    """Create nearly equal contiguous group assignments."""
    if not 0 < num_groups <= vocab_size:
        raise ValueError(
            f"num_groups must be in [1, {vocab_size}], got {num_groups}"
        )
    base, remainder = divmod(vocab_size, num_groups)
    sizes = [base + (group < remainder) for group in range(num_groups)]
    return torch.repeat_interleave(
        torch.arange(num_groups, dtype=torch.long),
        torch.tensor(sizes, dtype=torch.long),
    )


class GroupReduceEmbed(nn.Module):
    """Trainable compact representation produced by GroupReduce.

    Each vocabulary group ``g`` owns independent factors

        E_g = left_factors[g] @ right_factors[g].T.

    Published GroupReduce obtains the group assignment, ranks, and factors by
    weighted block-SVD plus iterative reassignment of a pretrained dense table.
    This class represents that result exactly and permits subsequent retraining.
    Contiguous groups and random factors are only a convenient initialization
    for tests or explicitly-labelled end-to-end adaptations.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        group_ranks,
        group_ids=None,
    ):
        super().__init__()
        ranks = tuple(int(rank) for rank in group_ranks)
        if vocab_size <= 0 or embed_dim <= 0:
            raise ValueError("vocab_size and embed_dim must be positive")
        if not ranks:
            raise ValueError("group_ranks must be non-empty")
        if any(rank <= 0 or rank > embed_dim for rank in ranks):
            raise ValueError(
                f"group ranks must be in [1, {embed_dim}], got {ranks}"
            )
        num_groups = len(ranks)
        if num_groups > vocab_size:
            raise ValueError("number of groups cannot exceed vocabulary size")
        if group_ids is None:
            group_ids = make_contiguous_groups(vocab_size, num_groups)
        else:
            group_ids = _as_long_vector(
                group_ids, name="group_ids", length=vocab_size
            )
        if group_ids.min().item() < 0 or group_ids.max().item() >= num_groups:
            raise ValueError(f"group_ids must be in [0, {num_groups - 1}]")

        group_sizes = torch.bincount(group_ids, minlength=num_groups)
        if torch.any(group_sizes == 0):
            missing = torch.where(group_sizes == 0)[0].tolist()
            raise ValueError(f"every GroupReduce group must be used; missing {missing}")

        token_ids_by_group = torch.cat([
            torch.where(group_ids == group)[0] for group in range(num_groups)
        ])
        inverse_grouped_order = torch.empty(vocab_size, dtype=torch.long)
        inverse_grouped_order[token_ids_by_group] = torch.arange(
            vocab_size, dtype=torch.long
        )
        offsets = torch.empty(vocab_size, dtype=torch.long)
        start = 0
        for group, size in enumerate(group_sizes.tolist()):
            token_ids = token_ids_by_group[start:start + size]
            offsets[token_ids] = torch.arange(size, dtype=torch.long)
            start += size

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.num_groups = int(num_groups)
        self.group_ranks = ranks
        self.group_sizes = tuple(int(size) for size in group_sizes.tolist())

        self.left_factors = nn.ParameterList([
            nn.Parameter(torch.randn(size, rank) * 0.02)
            for size, rank in zip(self.group_sizes, ranks)
        ])
        self.right_factors = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, rank) * 0.02)
            for rank in ranks
        ])
        self.register_buffer("group_ids", group_ids, persistent=True)
        self.register_buffer("group_offsets", offsets, persistent=True)
        self.register_buffer(
            "token_ids_by_group", token_ids_by_group, persistent=True
        )
        self.register_buffer(
            "inverse_grouped_order", inverse_grouped_order, persistent=True
        )
        self.register_load_state_dict_post_hook(
            lambda module, incompatible: module.validate_structure()
        )

    def validate_structure(self):
        expected_ids = torch.cat([
            torch.where(self.group_ids == group)[0]
            for group in range(self.num_groups)
        ])
        if not torch.equal(self.token_ids_by_group, expected_ids):
            raise ValueError("GroupReduce grouped token order is inconsistent")
        expected_inverse = torch.empty_like(self.inverse_grouped_order)
        expected_inverse[expected_ids] = torch.arange(
            self.vocab_size, device=expected_ids.device
        )
        if not torch.equal(self.inverse_grouped_order, expected_inverse):
            raise ValueError("GroupReduce inverse token order is inconsistent")
        expected_offsets = torch.empty_like(self.group_offsets)
        start = 0
        for group, size in enumerate(self.group_sizes):
            ids = expected_ids[start:start + size]
            expected_offsets[ids] = torch.arange(
                size, device=ids.device
            )
            start += size
        if not torch.equal(self.group_offsets, expected_offsets):
            raise ValueError("GroupReduce token offsets are inconsistent")

    def token_ids_for_group(self, group):
        if not 0 <= group < self.num_groups:
            raise IndexError(f"group index out of range: {group}")
        start = sum(self.group_sizes[:group])
        return self.token_ids_by_group[start:start + self.group_sizes[group]]

    def forward(self, input_ids, doc_mask=None):
        flat_ids = input_ids.reshape(-1)
        flat_groups = self.group_ids[flat_ids]
        flat_offsets = self.group_offsets[flat_ids]
        output = torch.empty(
            flat_ids.numel(), self.embed_dim,
            device=flat_ids.device,
            dtype=self.left_factors[0].dtype,
        )
        # Calling every group, including with an empty selection, keeps all
        # factors connected to autograd for DDP configurations that do not use
        # find_unused_parameters.
        for group in range(self.num_groups):
            mask = flat_groups == group
            coefficients = self.left_factors[group][flat_offsets[mask]]
            output[mask] = coefficients @ self.right_factors[group].T
        return output.view(*input_ids.shape, self.embed_dim), None


def _normalize_modes(values, *, name):
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    modes = tuple(int(value) for value in values)
    if not modes or any(mode <= 0 for mode in modes):
        raise ValueError(f"{name} must contain positive integers, got {modes}")
    return modes


def balanced_padded_modes(size, order):
    """Choose near-equal integer modes whose product is at least ``size``."""
    if size <= 0 or order <= 0:
        raise ValueError("size and order must be positive")
    base = max(1, int(size ** (1.0 / order)))
    while (base + 1) ** order <= size:
        base += 1
    while base ** order > size and base > 1:
        base -= 1
    modes = [base] * order
    cursor = 0
    while math.prod(modes) < size:
        # Increment the currently smallest factor, cycling equal factors.  This
        # minimizes padding while keeping factors close enough for compact TT
        # cores, following the paper's shape-selection guidance.
        minimum = min(modes)
        candidates = [i for i, value in enumerate(modes) if value == minimum]
        index = candidates[cursor % len(candidates)]
        modes[index] += 1
        cursor += 1
    return tuple(modes)


def balanced_exact_modes(size, order):
    """Factor ``size`` exactly into reasonably balanced positive modes."""
    if size <= 0 or order <= 0:
        raise ValueError("size and order must be positive")
    factors = []
    remainder = size
    divisor = 2
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor += 1
    if remainder > 1:
        factors.append(remainder)
    modes = [1] * order
    for factor in sorted(factors, reverse=True):
        index = min(range(order), key=modes.__getitem__)
        modes[index] *= factor
    return tuple(sorted(modes))


def _mixed_radix_digits(indices, modes):
    """Convert flat C-order indices to one digit per tensor mode."""
    digits = []
    for position in range(len(modes)):
        stride = math.prod(modes[position + 1:])
        digits.append(torch.div(indices, stride, rounding_mode="floor") % modes[position])
    return digits


class TTEmbedding(nn.Module):
    """Tensor-Train matrix parameterization of a vocabulary table.

    ``vocab_modes`` multiply to a padded vocabulary size >= ``vocab_size`` and
    ``embedding_modes`` multiply exactly to ``embed_dim``.  Core ``k`` has
    shape ``(r_k, I_k, J_k, r_{k+1})``.  Input lookup selects one ``I_k`` slice
    per core and contracts the TT ranks.  The tied head contracts the same
    cores with hidden states in reverse order and truncates padded vocabulary
    rows, so no dense V x d parameter is stored.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        vocab_modes,
        embedding_modes,
        tt_ranks,
        target_std=None,
        implementation="materialize",
        materialize_chunk_size=1024,
    ):
        super().__init__()
        vocab_modes = _normalize_modes(vocab_modes, name="vocab_modes")
        embedding_modes = _normalize_modes(
            embedding_modes, name="embedding_modes"
        )
        if len(vocab_modes) != len(embedding_modes):
            raise ValueError(
                "vocab_modes and embedding_modes must have the same order"
            )
        order = len(vocab_modes)
        if isinstance(tt_ranks, int):
            tt_ranks = (1,) + (int(tt_ranks),) * (order - 1) + (1,)
        else:
            tt_ranks = _normalize_modes(tt_ranks, name="tt_ranks")
            if len(tt_ranks) == order - 1:
                tt_ranks = (1,) + tt_ranks + (1,)
        if len(tt_ranks) != order + 1:
            raise ValueError(
                f"tt_ranks must have {order - 1} internal or {order + 1} "
                f"full entries, got {tt_ranks}"
            )
        if tt_ranks[0] != 1 or tt_ranks[-1] != 1:
            raise ValueError("the first and last TT ranks must equal 1")
        if math.prod(vocab_modes) < vocab_size:
            raise ValueError(
                f"vocab_modes product {math.prod(vocab_modes)} is smaller "
                f"than vocab_size {vocab_size}"
            )
        if math.prod(embedding_modes) != embed_dim:
            raise ValueError(
                f"embedding_modes product {math.prod(embedding_modes)} must "
                f"equal embed_dim {embed_dim}"
            )
        padded_vocab_size = math.prod(vocab_modes)
        if target_std is None:
            # Modified Glorot target used in the TT-Embedding paper.
            target_std = math.sqrt(2.0 / (padded_vocab_size + embed_dim))
        if target_std <= 0:
            raise ValueError("target_std must be positive when provided")
        if implementation not in {"materialize", "direct"}:
            raise ValueError(
                "TT implementation must be 'materialize' or 'direct'"
            )
        if materialize_chunk_size <= 0:
            raise ValueError("materialize_chunk_size must be positive")

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.vocab_modes = vocab_modes
        self.embedding_modes = embedding_modes
        self.tt_ranks = tuple(int(rank) for rank in tt_ranks)
        self.order = order
        self.padded_vocab_size = padded_vocab_size
        self.target_std = float(target_std)
        self.implementation = implementation
        self.materialize_chunk_size = int(materialize_chunk_size)

        # If unit-variance cores are used, one effective table element has
        # variance equal to the number of rank paths (product internal ranks).
        # Scaling every core by the same s multiplies the final std by s^order.
        rank_paths = math.prod(self.tt_ranks[1:-1])
        core_std = (target_std / math.sqrt(rank_paths)) ** (1.0 / order)
        self.cores = nn.ParameterList([
            nn.Parameter(
                torch.randn(
                    self.tt_ranks[k], vocab_modes[k], embedding_modes[k],
                    self.tt_ranks[k + 1],
                ) * core_std
            )
            for k in range(order)
        ])

    def _lookup_from_cores(self, input_ids, cores):
        flat_ids = input_ids.reshape(-1)
        if flat_ids.numel() and flat_ids.min().item() < 0:
            raise IndexError("input token id is negative")
        if flat_ids.numel() and flat_ids.max().item() >= self.padded_vocab_size:
            raise IndexError("input row is outside the padded TT matrix")
        digits = _mixed_radix_digits(flat_ids, self.vocab_modes)

        # First selected core has shape (B, J_0, r_1), since r_0=1.
        state = cores[0][0, digits[0], :, :]
        for position in range(1, self.order):
            # selected: (B, r_k, J_k, r_{k+1})
            selected = cores[position][:, digits[position], :, :].permute(
                1, 0, 2, 3
            )
            # state: (B, prod(J_<k), r_k)
            state = torch.einsum("bpr,brjq->bpjq", state, selected)
            state = state.reshape(flat_ids.numel(), -1, self.tt_ranks[position + 1])
        embeddings = state.squeeze(-1).reshape(
            *input_ids.shape, self.embed_dim
        )
        return embeddings

    def _lookup(self, input_ids):
        return self._lookup_from_cores(input_ids, tuple(self.cores))

    def materialize(self):
        """Construct the padded table without retaining full-row intermediates.

        A single ``_lookup(arange(V))`` expands production TT intermediates to
        roughly 123 GB at the repository's r=219 configuration. Chunking
        limits the live forward workspace. During training, non-reentrant
        activation checkpointing recomputes each chunk in backward instead of
        retaining every chunk's TT contraction graph; the final V x d table is
        still live because the vocabulary GEMM requires it.
        """
        chunks = []
        cores = tuple(self.cores)
        use_checkpoint = torch.is_grad_enabled() and any(
            core.requires_grad for core in cores
        )
        for start in range(
            0, self.padded_vocab_size, self.materialize_chunk_size
        ):
            rows = torch.arange(
                start,
                min(start + self.materialize_chunk_size, self.padded_vocab_size),
                device=self.cores[0].device,
            )
            if use_checkpoint:
                chunk = checkpoint(
                    lambda row_ids, *core_args: self._lookup_from_cores(
                        row_ids, core_args
                    ),
                    rows,
                    *cores,
                    use_reentrant=False,
                )
            else:
                chunk = self._lookup_from_cores(rows, cores)
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def forward(self, input_ids, doc_mask=None):
        if input_ids.numel() and (
            input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size
        ):
            raise IndexError("input token id is outside the configured vocabulary")
        # Selected-row contraction is exact and avoids constructing V x d for
        # the input path. ``implementation`` controls only the tied output
        # algebra, where all vocabulary rows are necessarily involved.
        return self._lookup(input_ids), None

    def project_hidden(self, hidden_states):
        """Compute exact tied full-vocabulary logits by TT contraction."""
        if self.implementation == "materialize":
            return hidden_states @ self.materialize()[:self.vocab_size].T
        flat = hidden_states.reshape(-1, self.embed_dim)
        # Invariant before processing core k in reverse:
        # (B, J_0, ..., J_k, r_{k+1}, I_{k+1}, ..., I_{N-1}).
        state = flat.reshape(flat.size(0), *self.embedding_modes).unsqueeze(-1)
        for position in range(self.order - 1, -1, -1):
            m_axis = 1 + position
            rank_axis = 2 + position
            state = torch.tensordot(
                state,
                self.cores[position],
                dims=([m_axis, rank_axis], [2, 3]),
            )
            # tensordot order is B, J_prefix, I_suffix, r_k, I_k.
            prefix_len = position
            suffix_len = self.order - position - 1
            rank_position = 1 + prefix_len + suffix_len
            mode_position = rank_position + 1
            permutation = (
                [0]
                + list(range(1, 1 + prefix_len))
                + [rank_position, mode_position]
                + list(range(1 + prefix_len, rank_position))
            )
            state = state.permute(permutation)

        # r_0=1; remaining vocabulary modes are already in C-order.
        logits = state.squeeze(1).reshape(flat.size(0), self.padded_vocab_size)
        logits = logits[:, :self.vocab_size]
        return logits.view(*hidden_states.shape[:-1], self.vocab_size)
