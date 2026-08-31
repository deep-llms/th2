"""Product-code tail interface with an exactly tied output head.

The highest-importance ``head_size`` tokens own private dense rows.  Every
other ("tail") token is represented by a fixed code ``(c_1, ..., c_H)`` into
``H`` shared full-dimensional codebooks of ``B`` rows each, combined with a
learned per-token gate vector ``gates = 1 + gate_offsets``::

    head token:  e_w = bias + E_h[row(w)]
    tail token:  e_w = bias + sum_i gates[w, i] * C_i[codes[w, i]]

Gates are stored as zero-initialized *offsets* from 1 because the training
harness keeps compressed embedding parameters in bfloat16, whose spacing at
1.0 (2^-7) exceeds the learning-rate-sized Adam step: a gate stored as 1.0
could never move, while an offset near 0 has full relative precision.

The effective tail table is ``(G * S) @ [C_1; ...; C_H]`` with ``S`` a fixed
sparse binary selection matrix (one entry per codebook per row): full rank,
combinatorial identity, no learned routing and no convex-weight constraint.
Codes and the head/tail partition are persistent buffers, so a checkpoint is
self-describing and independent of the importance file or dense teacher used
to build it.  See ``docs/product_code_design.md``.
"""

from __future__ import annotations

import hashlib
import struct

import torch
import torch.nn as nn

from .compression_init import frequency_rank_order


def product_code_parameter_count(vocab_size, embed_dim, head_size,
                                 num_hashes, num_buckets):
    """Exact trainable parameter count of :class:`ProductCodeEmbed`."""
    vocab_size, embed_dim = int(vocab_size), int(embed_dim)
    head_size, num_hashes = int(head_size), int(num_hashes)
    num_buckets = int(num_buckets)
    tail_size = vocab_size - head_size
    return (
        head_size * embed_dim
        + num_hashes * num_buckets * embed_dim
        + tail_size * num_hashes
        + embed_dim
    )


def head_tail_partition(importance, head_size):
    """Split the vocabulary into (head_ids, tail_ids) by importance.

    The head is the ``head_size`` highest-importance tokens under the stable
    ``(-importance, token_id)`` order shared with every other arm; both outputs
    are ascending token ids.  This is the single owner of the rule — the
    module, the trainer's artifact checks, and ``scripts/make_pq_codes.py`` all
    call it.
    """
    importance = torch.as_tensor(importance, dtype=torch.float64).cpu()
    if importance.ndim != 1:
        raise ValueError("importance must be a one-dimensional vector")
    vocab_size = importance.numel()
    head_size = int(head_size)
    if not 0 <= head_size < vocab_size:
        raise ValueError("head_size must be in [0, vocab_size)")
    head_ids = torch.sort(frequency_rank_order(importance)[:head_size]).values
    is_tail = torch.ones(vocab_size, dtype=torch.bool)
    is_tail[head_ids] = False
    return head_ids, torch.nonzero(is_tail).flatten()


def _keyed_bucket(token_id, hash_index, salt, seed, num_buckets):
    key = struct.pack("<qqq", int(seed), int(hash_index), int(salt))
    digest = hashlib.blake2b(
        struct.pack("<q", int(token_id)), key=key, digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % int(num_buckets)


def hashed_codes(tail_ids, num_hashes, num_buckets, seed=0):
    """Deterministic keyed-hash codes with unique full signatures.

    Tail tokens are processed in ascending id order.  On a collision the whole
    signature is re-hashed with an incremented salt (MultiHashFormer-style
    iterative rehashing, applied to every coordinate); if that fails within
    ``max_rehash`` tries, a deterministic linear probe over the mixed-radix
    signature space guarantees a unique result.  No RNG state is consumed.
    """
    tail_ids = torch.as_tensor(tail_ids, dtype=torch.long).cpu().tolist()
    num_hashes, num_buckets = int(num_hashes), int(num_buckets)
    if num_hashes <= 0 or num_buckets <= 0:
        raise ValueError("num_hashes and num_buckets must be positive")
    if num_buckets ** num_hashes < len(tail_ids):
        raise ValueError(
            f"{num_buckets}^{num_hashes} signatures cannot cover "
            f"{len(tail_ids)} tail tokens"
        )
    seen = set()
    rows = []
    space = num_buckets ** num_hashes
    max_rehash = 64
    for token in tail_ids:
        signature = tuple(
            _keyed_bucket(token, index, 0, seed, num_buckets)
            for index in range(num_hashes)
        )
        salt = 0
        # Stage 1: salted full rehash (all coordinates), as MultiHashFormer's
        # iterative rehashing but over the whole signature so a saturated
        # prefix cannot trap the search.
        while signature in seen and salt < max_rehash:
            salt += 1
            signature = tuple(
                _keyed_bucket(token, index, salt, seed, num_buckets)
                for index in range(num_hashes)
            )
        # Stage 2 (deterministic, guaranteed): linear probe over the mixed-
        # radix signature space starting at the hashed position.
        if signature in seen:
            start = 0
            for value in signature:
                start = start * num_buckets + value
            for offset in range(1, space):
                probe = (start + offset) % space
                digits = []
                for _ in range(num_hashes):
                    digits.append(probe % num_buckets)
                    probe //= num_buckets
                candidate = tuple(reversed(digits))
                if candidate not in seen:
                    signature = candidate
                    break
            else:
                raise RuntimeError(
                    f"could not find a unique signature for token {token}"
                )
        seen.add(signature)
        rows.append(signature)
    return torch.tensor(rows, dtype=torch.long).view(len(tail_ids), num_hashes)


def _chunked_sq_distances(points, centroids, chunk=8192):
    """Squared distances (n, k) computed in row chunks to bound memory."""
    out = torch.empty(points.size(0), centroids.size(0),
                      dtype=points.dtype, device=points.device)
    c_norm = (centroids * centroids).sum(-1)
    for start in range(0, points.size(0), chunk):
        block = points[start:start + chunk]
        out[start:start + chunk] = (
            (block * block).sum(-1, keepdim=True)
            - 2.0 * block @ centroids.T
            + c_norm.unsqueeze(0)
        )
    return out


def pq_codes(table, num_hashes, num_buckets, *, iters=20, seed=0,
             capacity_factor=2.0):
    """Product-quantization codes for a (tail) table.

    The embedding dimension is split into ``num_hashes`` contiguous
    sub-vectors; k-means with ``num_buckets`` centroids runs on each.  A
    capacity limit (``capacity_factor`` times the mean occupancy) keeps sharing
    density bounded: overfull buckets shed their farthest members to the
    nearest non-full centroid.  Duplicate full signatures are resolved by
    moving the later token's last coordinate to its next-nearest centroid.
    Uses a private generator; the global RNG is untouched.
    """
    table = torch.as_tensor(table, dtype=torch.float32)
    n, d = table.shape
    num_hashes, num_buckets = int(num_hashes), int(num_buckets)
    if d % num_hashes != 0:
        raise ValueError("embedding dimension must be divisible by num_hashes")
    if num_buckets > n:
        raise ValueError("num_buckets cannot exceed the number of tail rows")
    if num_buckets ** num_hashes < n:
        raise ValueError(
            f"{num_buckets}^{num_hashes} signatures cannot cover {n} rows"
        )
    sub = d // num_hashes
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    capacity = max(1, int(capacity_factor * n / num_buckets))
    codes = torch.empty(n, num_hashes, dtype=torch.long)
    ranked_candidates = []
    top_k = min(num_buckets, 256)
    for index in range(num_hashes):
        points = table[:, index * sub:(index + 1) * sub].contiguous()
        choice = torch.randperm(n, generator=generator)[:num_buckets]
        centroids = points[choice].clone()
        assignment = None
        for _ in range(max(1, int(iters))):
            distances = _chunked_sq_distances(points, centroids)
            assignment = distances.argmin(dim=1)
            sums = torch.zeros_like(centroids).index_add_(0, assignment, points)
            counts = torch.bincount(assignment, minlength=num_buckets)
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        distances = _chunked_sq_distances(points, centroids)
        assignment = distances.argmin(dim=1)
        assignment = _enforce_capacity(distances, assignment, capacity)
        codes[:, index] = assignment
        ranked_candidates.append(
            distances.topk(top_k, dim=1, largest=False).indices.cpu()
        )
    codes = _resolve_duplicate_signatures(
        codes.cpu(), ranked_candidates, num_buckets, capacity
    )
    return codes


def _enforce_capacity(distances, assignment, capacity, top_k=64):
    """Greedy repair: overfull buckets shed farthest members to a nearby bucket.

    Candidates are each point's ``top_k`` nearest centroids; if none has room,
    the globally least-full bucket takes the point, so the repair terminates
    with every bucket at or below ``capacity`` whenever capacity * K >= n.
    """
    assignment = assignment.clone()
    num_buckets = distances.size(1)
    counts = torch.bincount(assignment, minlength=num_buckets)
    top_k = min(int(top_k), num_buckets)
    for bucket in torch.nonzero(counts > capacity).flatten().tolist():
        members = torch.nonzero(assignment == bucket).flatten()
        order = distances[members, bucket].argsort(descending=True)
        excess = int(counts[bucket].item() - capacity)
        movers = members[order[:excess]]
        candidates = distances[movers].topk(top_k, dim=1, largest=False).indices
        for point, ranked in zip(movers.tolist(), candidates.tolist()):
            target = next(
                (c for c in ranked if c != bucket and counts[c] < capacity), None
            )
            if target is None:
                target = int(counts.argmin().item())
                if counts[target] >= capacity:
                    raise RuntimeError("capacity repair failed: no free bucket")
            assignment[point] = target
            counts[bucket] -= 1
            counts[target] += 1
    return assignment


def _resolve_duplicate_signatures(codes, ranked_candidates, num_buckets,
                                  capacity=None):
    """Make full signatures unique — guaranteed whenever B^H >= n.

    For a duplicate row, in order: (A) change one coordinate to one of its
    nearest centroids (last coordinate first), preferring buckets below
    ``capacity``; (B) change one coordinate to *any* bucket, same preference;
    (C) a deterministic mixed-radix linear probe over the whole signature
    space, first over signatures whose buckets are all below capacity, then
    unconstrained.  Stage C terminates because the caller checked
    ``num_buckets ** num_hashes >= n``; the capacity bound is therefore
    best-effort (always respected when any capacity-respecting unique
    signature exists) and the realized occupancy is reported by the caller.
    """
    codes = codes.clone()
    n, num_hashes = codes.shape
    B = int(num_buckets)
    counts = [torch.bincount(codes[:, j], minlength=B).tolist() for j in range(num_hashes)]
    seen = {}
    rows = codes.tolist()
    space = B ** num_hashes

    def under_capacity(j, bucket):
        return capacity is None or counts[j][bucket] < capacity

    def single_coordinate(signature, candidates_of, prefer_free):
        for j in reversed(range(num_hashes)):
            for candidate in candidates_of(j):
                if candidate == signature[j]:
                    continue
                if prefer_free and not under_capacity(j, candidate):
                    continue
                trial = signature[:j] + (candidate,) + signature[j + 1:]
                if trial not in seen:
                    return trial
        return None

    def linear_probe(signature, require_free):
        start = 0
        for value in signature:
            start = start * B + value
        for offset in range(1, space):
            probe = (start + offset) % space
            digits = []
            for _ in range(num_hashes):
                digits.append(probe % B)
                probe //= B
            trial = tuple(reversed(digits))
            if trial in seen:
                continue
            if require_free and not all(
                under_capacity(j, trial[j]) for j in range(num_hashes)
            ):
                continue
            return trial
        return None

    for row, signature in enumerate(rows):
        signature = tuple(signature)
        if signature not in seen:
            seen[signature] = row
            continue
        nearest = lambda j: ranked_candidates[j][row].tolist()  # noqa: E731
        everything = lambda j: range(B)  # noqa: E731
        trial = (
            single_coordinate(signature, nearest, True)
            or single_coordinate(signature, everything, True)
            or single_coordinate(signature, nearest, False)
            or single_coordinate(signature, everything, False)
            or linear_probe(signature, True)
            or linear_probe(signature, False)
        )
        if trial is None:
            raise RuntimeError(f"could not make signature unique for row {row}")
        for j in range(num_hashes):
            if trial[j] != signature[j]:
                counts[j][signature[j]] -= 1
                counts[j][trial[j]] += 1
        codes[row] = torch.tensor(trial, dtype=torch.long)
        seen[trial] = row
    return codes


class _GatedCodebookSum(torch.autograd.Function):
    """``sum_i gates[:, i] * C_i[codes[:, i]]`` without saving gathered rows.

    A plain ``gather * gate`` chain saves one gathered ``(M, d)`` tensor per
    codebook for the gate gradient (four fp32 ``(V, d)`` tensors, ~2.3 GB,
    when the tied head materializes the whole table).  This function saves
    only ``codes`` and ``gates`` (the codebooks are parameters, referenced
    not copied) and recomputes the gathers in backward.
    """

    @staticmethod
    def forward(ctx, codes, gates, *codebooks):
        work = gates.dtype
        out = None
        for index, codebook in enumerate(codebooks):
            part = codebook.index_select(0, codes[:, index]).to(work)
            part.mul_(gates[:, index:index + 1])
            out = part if out is None else out.add_(part)
        ctx.save_for_backward(codes, gates, *codebooks)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        codes, gates, *codebooks = ctx.saved_tensors
        work = gates.dtype
        grad = grad_out.to(work)
        grad_gates = torch.empty_like(gates)
        grad_codebooks = []
        for index, codebook in enumerate(codebooks):
            rows = codebook.index_select(0, codes[:, index]).to(work)
            grad_gates[:, index] = (grad * rows).sum(1)
            accum = torch.zeros(codebook.shape, dtype=work, device=codebook.device)
            accum.index_add_(0, codes[:, index], grad * gates[:, index:index + 1])
            grad_codebooks.append(accum.to(codebook.dtype))
        return (None, grad_gates) + tuple(grad_codebooks)


class ProductCodeEmbed(nn.Module):
    """Dense head rows plus product-code tail composition (see module doc)."""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        head_size,
        num_hashes,
        num_buckets,
        *,
        importance=None,
        head_ids=None,
        codes=None,
        assignment="hashed",
        seed=0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.head_size = int(head_size)
        self.num_hashes = int(num_hashes)
        self.num_buckets = int(num_buckets)
        self.assignment = str(assignment)
        self.seed = int(seed)
        self._validate_hyperparameters()
        self.tail_size = self.vocab_size - self.head_size

        head_ids = self._resolve_head(importance, head_ids)
        head_row = torch.full((self.vocab_size,), -1, dtype=torch.long)
        head_row[head_ids] = torch.arange(self.head_size, dtype=torch.long)
        tail_ids = torch.nonzero(head_row < 0).flatten()
        tail_row = torch.full((self.vocab_size,), -1, dtype=torch.long)
        tail_row[tail_ids] = torch.arange(self.tail_size, dtype=torch.long)
        if codes is None:
            if self.assignment == "hashed":
                codes = hashed_codes(
                    tail_ids, self.num_hashes, self.num_buckets, self.seed
                )
            else:
                raise ValueError(
                    "explicit codes are required for assignment="
                    f"{self.assignment!r}"
                )
        codes = torch.as_tensor(codes, dtype=torch.long).cpu().clone()

        self.register_buffer("head_ids", head_ids, persistent=True)
        self.register_buffer("tail_ids", tail_ids, persistent=True)
        self.register_buffer("head_row", head_row, persistent=True)
        self.register_buffer("tail_row", tail_row, persistent=True)
        self.register_buffer("codes", codes, persistent=True)

        self.E_h = nn.Parameter(torch.empty(self.head_size, self.embed_dim))
        self.C = nn.ParameterList([
            nn.Parameter(torch.empty(self.num_buckets, self.embed_dim))
            for _ in range(self.num_hashes)
        ])
        self.gate_offsets = nn.Parameter(torch.empty(self.tail_size, self.num_hashes))
        self.bias = nn.Parameter(torch.zeros(self.embed_dim))
        self.reset_parameters()
        self.validate_structure()
        self.register_load_state_dict_post_hook(self._validate_after_load)
        self._step_metrics = None

    # ----------------------------------------------------------------- setup
    def _validate_hyperparameters(self):
        if self.vocab_size <= 0 or self.embed_dim <= 0:
            raise ValueError("vocab_size and embed_dim must be positive")
        if not 0 <= self.head_size < self.vocab_size:
            raise ValueError("head_size must be in [0, vocab_size)")
        if self.num_hashes <= 0 or self.num_buckets <= 0:
            raise ValueError("num_hashes and num_buckets must be positive")
        if self.assignment not in {"hashed", "pq", "checkpoint"}:
            raise ValueError(f"unknown assignment {self.assignment!r}")

    def _resolve_head(self, importance, head_ids):
        if head_ids is not None:
            ids = torch.as_tensor(head_ids, dtype=torch.long).cpu()
            ids = torch.sort(ids.flatten()).values
            if ids.numel() != self.head_size:
                raise ValueError(
                    f"expected {self.head_size} head ids, got {ids.numel()}"
                )
            return ids
        if self.head_size == 0:
            return torch.empty(0, dtype=torch.long)
        if importance is None:
            raise ValueError(
                "importance is required when head_ids is not supplied"
            )
        importance = torch.as_tensor(importance, dtype=torch.float64).cpu()
        if importance.shape != (self.vocab_size,):
            raise ValueError(
                f"importance must have shape ({self.vocab_size},), got "
                f"{tuple(importance.shape)}"
            )
        return head_tail_partition(importance, self.head_size)[0]

    def reset_parameters(self):
        nn.init.normal_(self.E_h, std=0.02)
        for codebook in self.C:
            # A composed tail row sums num_hashes codebook rows; matching the
            # head-row norm at init keeps head and tail on the same scale.
            nn.init.normal_(codebook, std=0.02 / (self.num_hashes ** 0.5))
        nn.init.zeros_(self.gate_offsets)
        nn.init.zeros_(self.bias)

    @staticmethod
    def no_decay_parameters():
        """Parameter names (relative to this module) to exempt from weight decay."""
        return ("gate_offsets",)

    @property
    def work_dtype(self):
        """Accumulation dtype: fp32 for half-precision parameters, else as-is.

        Adding a small offset to 1.0 in bf16 would round back to 1.0 (spacing
        2^-7), so gates and the tail sum are formed in fp32 when parameters
        are bf16/fp16; fp32/fp64 parameters keep their own precision.
        """
        dtype = self.bias.dtype
        if dtype in (torch.bfloat16, torch.float16):
            return torch.float32
        return dtype

    @property
    def gates(self):
        """Effective per-token mixing weights, ``1 + gate_offsets``."""
        return 1.0 + self.gate_offsets.to(self.work_dtype)

    @property
    def parameter_count(self):
        return product_code_parameter_count(
            self.vocab_size, self.embed_dim, self.head_size,
            self.num_hashes, self.num_buckets,
        )

    # ------------------------------------------------------------ structure
    def validate_structure(self):
        """Validate the partition, inverse maps, and code uniqueness.

        Runs at construction and after every ``load_state_dict`` — including
        on CUDA-resident modules during Trainer resume — so every comparison
        tensor is created on the buffers' own device.
        """
        V, H = self.vocab_size, self.num_hashes
        device = self.head_row.device
        for name in ("head_ids", "tail_ids", "head_row", "tail_row", "codes"):
            if getattr(self, name).dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")
        if self.head_ids.shape != (self.head_size,):
            raise ValueError("head_ids has the wrong size")
        if self.tail_ids.shape != (self.tail_size,):
            raise ValueError("tail_ids has the wrong size")
        for name in ("head_row", "tail_row"):
            if getattr(self, name).shape != (V,):
                raise ValueError(f"{name} has the wrong size")
        if self.head_ids.numel() > 1 and bool(torch.any(
            self.head_ids[1:] <= self.head_ids[:-1]
        )):
            raise ValueError("head_ids must be sorted unique")
        if self.tail_ids.numel() > 1 and bool(torch.any(
            self.tail_ids[1:] <= self.tail_ids[:-1]
        )):
            raise ValueError("tail_ids must be sorted unique")
        union = torch.cat([self.head_ids, self.tail_ids])
        if union.numel() != V or bool(torch.any(union < 0)) or bool(torch.any(union >= V)):
            raise ValueError("head/tail ids are out of range")
        if not torch.equal(torch.sort(union).values, torch.arange(V, device=device)):
            raise ValueError("head and tail ids must partition the vocabulary")
        expected_head = torch.full((V,), -1, dtype=torch.long, device=device)
        expected_head[self.head_ids] = torch.arange(self.head_size, device=device)
        expected_tail = torch.full((V,), -1, dtype=torch.long, device=device)
        expected_tail[self.tail_ids] = torch.arange(self.tail_size, device=device)
        if not torch.equal(self.head_row, expected_head):
            raise ValueError("head_row is not the exact inverse of head_ids")
        if not torch.equal(self.tail_row, expected_tail):
            raise ValueError("tail_row is not the exact inverse of tail_ids")
        if self.codes.shape != (self.tail_size, H):
            raise ValueError(
                f"codes must have shape ({self.tail_size}, {H}), got "
                f"{tuple(self.codes.shape)}"
            )
        if self.tail_size and (
            bool(torch.any(self.codes < 0)) or bool(torch.any(self.codes >= self.num_buckets))
        ):
            raise ValueError("codes contain out-of-range bucket ids")
        if self.tail_size:
            # One mixed-radix int64 key per row (4096^4 = 2^48 fits) makes the
            # uniqueness check a 1-D unique instead of a row-wise one.
            key = torch.zeros(self.tail_size, dtype=torch.long, device=device)
            for index in range(H):
                key = key * self.num_buckets + self.codes[:, index]
            if torch.unique(key).numel() != self.tail_size:
                raise ValueError("tail signatures must be unique")
        return True

    @staticmethod
    def _validate_after_load(module, incompatible_keys):
        module.validate_structure()

    @classmethod
    def structure_from_state(cls, state):
        """Validate and extract checkpoint-authoritative structure."""
        if not isinstance(state, dict):
            raise TypeError("Product Code state must be a dictionary")
        keys = set(state)
        required = {"E_h", "C.0", "gate_offsets", "bias", "codes", "head_ids",
                    "tail_ids", "head_row", "tail_row"}
        missing = required - keys
        if missing:
            raise ValueError(
                f"Product Code state is missing {sorted(missing)}"
            )
        codebooks = []
        for key in keys:
            if key.startswith("C."):
                suffix = key.removeprefix("C.")
                if not suffix.isdigit():
                    raise ValueError(f"invalid codebook key {key!r}")
                codebooks.append((int(suffix), key))
        codebooks.sort()
        if [index for index, _ in codebooks] != list(range(len(codebooks))):
            raise ValueError("Product Code codebooks are not contiguous")
        num_hashes = len(codebooks)
        bias = state["bias"]
        if bias.ndim != 1:
            raise ValueError("bias must be one-dimensional")
        embed_dim = int(bias.numel())
        E_h = state["E_h"]
        if E_h.ndim != 2 or E_h.shape[1] != embed_dim:
            raise ValueError("E_h must be a (head_size, embed_dim) matrix")
        head_size = int(E_h.shape[0])
        buckets = {int(state[key].shape[0]) for _, key in codebooks}
        if len(buckets) != 1 or any(
            state[key].ndim != 2 or state[key].shape[1] != embed_dim
            for _, key in codebooks
        ):
            raise ValueError("codebooks must share one (num_buckets, embed_dim) shape")
        num_buckets = buckets.pop()
        codes = state["codes"]
        vocab_size = int(state["head_row"].numel())
        tail_size = vocab_size - head_size
        if codes.shape != (tail_size, num_hashes):
            raise ValueError("codes shape does not match head/tail sizes")
        if state["gate_offsets"].shape != (tail_size, num_hashes):
            raise ValueError("gate_offsets shape does not match codes")
        if state["head_ids"].shape != (head_size,):
            raise ValueError("head_ids shape does not match E_h")
        return {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "head_size": head_size,
            "num_hashes": num_hashes,
            "num_buckets": num_buckets,
            "head_ids": state["head_ids"],
            "codes": codes,
        }

    # -------------------------------------------------------------- forward
    def materialize(self, token_ids=None):
        """Effective embeddings for ``token_ids`` (default: the whole vocab).

        Sync-free: both branches are gathered for every position with clamped
        row indices and selected by mask, so no ``nonzero`` host round-trip is
        needed and every parameter stays in autograd for head-only or
        tail-only batches (required by DDP with find_unused_parameters=False).
        """
        device = self.bias.device
        if token_ids is None:
            token_ids = torch.arange(self.vocab_size, device=device)
        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
        flat = token_ids.reshape(-1)
        is_head = (self.head_row[flat] >= 0).unsqueeze(1)
        tail_rows = self.tail_row[flat].clamp(min=0)
        codes = self.codes[tail_rows]
        gates = self.gates[tail_rows]                                   # fp32
        work = self.work_dtype
        tail = _GatedCodebookSum.apply(codes, gates, *self.C)          # (M, d) work dtype
        if self.head_size:
            head = self.E_h[self.head_row[flat].clamp(min=0)].to(work)
            mixed = torch.where(is_head, head, tail)
        else:
            mixed = tail
        out = mixed.to(self.bias.dtype) + self.bias
        return out.view(*token_ids.shape, self.embed_dim)

    def forward(self, input_ids, doc_mask=None):
        embeddings = self.materialize(input_ids)
        with torch.no_grad():
            flat = input_ids.reshape(-1)
            is_head = (self.head_row[flat] >= 0)
            norms = embeddings.detach().reshape(-1, self.embed_dim).float().norm(dim=-1)
            head_count = is_head.sum()
            total = torch.full((), flat.numel(), device=norms.device, dtype=torch.long)
            head_sum = (norms * is_head).sum()
            self._step_metrics = {
                "product_code_head_fraction": (head_count.to(norms.dtype), total),
                "product_code_head_embedding_norm": (head_sum, head_count),
                "product_code_tail_embedding_norm": (norms.sum() - head_sum, total - head_count),
            }
        return embeddings, None

    def pop_step_metrics(self):
        """Return and clear device-side ``metric: (sum, count)`` values."""
        metrics = self._step_metrics
        self._step_metrics = None
        return metrics
