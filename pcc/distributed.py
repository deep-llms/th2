"""Small DDP primitives; no process creation or GPU management on import."""
import hashlib
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class JointLoss(nn.Module):
    """Keep the LM head and checkpointed loss inside DDP's forward graph."""
    def __init__(self, model):
        super().__init__()
        self.joint = model

    def forward(self, context):
        return self.joint.losses(self.joint(context), context)


def wrap(model):
    device = next(model.parameters()).device
    return DistributedDataParallel(JointLoss(model),
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=False)


def identical_parameters(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, digest.hexdigest())
    if len(set(hashes)) != 1:
        raise ValueError("DDP model replicas diverged")
    return hashes[0]
