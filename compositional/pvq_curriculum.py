"""Dense-table clustering curriculum from the P-VQ training procedure."""

from __future__ import annotations

import json
import os

import torch
from transformers import TrainerCallback

from .compression_init import capacity_constrained_kmeans


PVQ_CURRICULUM_STATE = "pvq_curriculum_state.pt"
PVQ_CURRICULUM_METADATA = "pvq_curriculum_state.json"


class PVQCurriculumCallback(TrainerCallback):
    """Periodically quantize a dense tied table's leading coordinates.

    This callback intentionally operates on the *dense* embedding Parameter.
    Between clustering events each row is independently trainable, matching
    P-VQ Algorithm 1.  Compact codebook/exclusive parameters are created only
    after the curriculum by ``scripts/convert_dense_baseline.py``.

    The original paper cites a balanced-k-means implementation whose n-by-n
    assignment is infeasible at this project's vocabulary size.  We use the
    explicitly recorded capacity-repair approximation implemented in
    :func:`capacity_constrained_kmeans`.
    """

    def __init__(
        self,
        *,
        shared_dim,
        k_begin=1024,
        k_end=128,
        k_decay=128,
        cluster_every=1000,
        curriculum_steps=10000,
        cluster_iters=100,
        cluster_restarts=10,
        cluster_chunk_size=4096,
        cluster_device="cuda",
        seed=42,
        resume_checkpoint=None,
        final_recluster=True,
    ):
        if shared_dim <= 0:
            raise ValueError("shared_dim must be positive")
        if not 0 < k_end <= k_begin:
            raise ValueError("require 0 < k_end <= k_begin")
        if k_decay <= 0 or cluster_every <= 0 or curriculum_steps <= 0:
            raise ValueError("curriculum schedule values must be positive")
        self.shared_dim = int(shared_dim)
        self.k_begin = int(k_begin)
        self.k_end = int(k_end)
        self.k_decay = int(k_decay)
        self.cluster_every = int(cluster_every)
        self.curriculum_steps = int(curriculum_steps)
        self.cluster_iters = int(cluster_iters)
        self.cluster_restarts = int(cluster_restarts)
        self.cluster_chunk_size = int(cluster_chunk_size)
        self.cluster_device = cluster_device
        self.seed = int(seed)
        self.final_recluster = bool(final_recluster)
        self.last_cluster_step = None
        self.assignments = None
        self.num_codes = None

        if resume_checkpoint is not None:
            state_path = os.path.join(
                resume_checkpoint, PVQ_CURRICULUM_STATE
            )
            if not os.path.isfile(state_path):
                raise FileNotFoundError(
                    "P-VQ curriculum resume checkpoint is missing "
                    f"{PVQ_CURRICULUM_STATE}: {resume_checkpoint}"
                )
            saved = torch.load(
                state_path, map_location="cpu", weights_only=True
            )
            self._validate_saved_state(saved)
            self.last_cluster_step = int(saved["last_cluster_step"].item())
            self.num_codes = int(saved["num_codes"].item())
            self.assignments = saved["assignments"].to(torch.long)

    def _validate_saved_state(self, saved):
        required = {"last_cluster_step", "num_codes", "assignments"}
        if not isinstance(saved, dict) or set(saved) != required:
            raise ValueError(
                f"Invalid {PVQ_CURRICULUM_STATE} schema: "
                f"{sorted(saved) if isinstance(saved, dict) else type(saved)}"
            )
        if saved["assignments"].ndim != 1:
            raise ValueError("saved P-VQ assignments must be one-dimensional")

    def _codes_for_step(self, step):
        event_index = step // self.cluster_every
        return max(self.k_begin - event_index * self.k_decay, self.k_end)

    def _dense_weight(self, model):
        while hasattr(model, "module"):
            model = model.module
        input_module = model.get_input_embeddings()
        output_module = model.get_output_embeddings()
        if input_module is None or output_module is None:
            raise ValueError("P-VQ curriculum requires conventional input/output embeddings")
        if not hasattr(input_module, "weight") or not hasattr(output_module, "weight"):
            raise ValueError("P-VQ curriculum requires dense embedding weights")
        if input_module.weight.data_ptr() != output_module.weight.data_ptr():
            raise ValueError("P-VQ curriculum requires exact input-output parameter tying")
        weight = input_module.weight
        if not 0 < self.shared_dim < weight.size(1):
            raise ValueError(
                f"shared_dim={self.shared_dim} must be below hidden size {weight.size(1)}"
            )
        if self.k_begin > weight.size(0):
            raise ValueError("k_begin cannot exceed vocabulary size")
        return weight

    @torch.no_grad()
    def _cluster(self, model, step):
        weight = self._dense_weight(model)
        num_codes = self._codes_for_step(step)
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if distributed else 0

        if rank == 0:
            _, assignments = capacity_constrained_kmeans(
                weight[:, :self.shared_dim].detach().float(),
                num_codes,
                num_iters=self.cluster_iters,
                num_restarts=self.cluster_restarts,
                seed=self.seed + step,
                chunk_size=self.cluster_chunk_size,
                device=self.cluster_device,
            )
            assignments = assignments.to(weight.device)
            shared = weight[:, :self.shared_dim].detach().float()
            centroids = torch.zeros(
                num_codes, self.shared_dim,
                device=weight.device, dtype=torch.float32,
            )
            centroids.index_add_(0, assignments, shared)
            counts = torch.bincount(
                assignments, minlength=num_codes
            ).to(device=weight.device, dtype=torch.float32)
            centroids.div_(counts.unsqueeze(1))
        else:
            assignments = torch.empty(
                weight.size(0), dtype=torch.long, device=weight.device
            )
            centroids = torch.empty(
                num_codes, self.shared_dim,
                dtype=torch.float32, device=weight.device,
            )

        if distributed:
            torch.distributed.broadcast(assignments, src=0)
            torch.distributed.broadcast(centroids, src=0)
        weight[:, :self.shared_dim].copy_(
            centroids[assignments].to(dtype=weight.dtype)
        )
        self.last_cluster_step = int(step)
        self.num_codes = int(num_codes)
        self.assignments = assignments.cpu()

    def _maybe_cluster(self, model, step):
        if step < 0 or step >= self.curriculum_steps:
            return
        if step % self.cluster_every != 0:
            return
        if self.last_cluster_step == step:
            return
        self._cluster(model, step)

    def _state_dict(self):
        if self.assignments is None:
            raise RuntimeError("P-VQ curriculum has not performed a clustering event")
        return {
            "last_cluster_step": torch.tensor(
                self.last_cluster_step, dtype=torch.long
            ),
            "num_codes": torch.tensor(self.num_codes, dtype=torch.long),
            "assignments": self.assignments.to(torch.long),
        }

    def _save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self._state_dict(), os.path.join(directory, PVQ_CURRICULUM_STATE)
        )
        metadata = {
            "format_version": 1,
            "last_cluster_step": self.last_cluster_step,
            "num_codes": self.num_codes,
            "shared_dim": self.shared_dim,
            "assignment_method": (
                "scalable_capacity_repair_kmeans_not_original_hungarian_balanced_kmeans"
            ),
            "schedule": {
                "k_begin": self.k_begin,
                "k_end": self.k_end,
                "k_decay": self.k_decay,
                "cluster_every": self.cluster_every,
                "curriculum_steps": self.curriculum_steps,
            },
            "clustering": {
                "iters": self.cluster_iters,
                "restarts": self.cluster_restarts,
                "chunk_size": self.cluster_chunk_size,
                "seed": self.seed,
            },
        }
        with open(os.path.join(directory, PVQ_CURRICULUM_METADATA), "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._maybe_cluster(model, int(state.global_step))

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._maybe_cluster(model, int(state.global_step))

    def on_save(self, args, state, control, **kwargs):
        if args.should_save and self.assignments is not None:
            checkpoint_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            self._save(checkpoint_dir)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.final_recluster and int(state.global_step) >= self.curriculum_steps:
            # Recluster the final dense table at target K so conversion does not
            # reuse assignments from before the last interval of dense updates.
            self._cluster(model, self.curriculum_steps)
        if args.should_save and self.assignments is not None:
            self._save(args.output_dir)
