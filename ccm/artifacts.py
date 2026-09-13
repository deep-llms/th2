"""Checked tensor artifacts. Never silently attach a table from another run."""
from pathlib import Path
import time
import torch
from .contracts import require, fresh_dir, write_json, read_json, file_hash, digest_json, VERSION, HOOKS


def tensor_hash(tensor):
    x = tensor.detach().cpu().contiguous()
    import hashlib
    return hashlib.sha256(memoryview(x.reshape(-1).view(torch.uint8).numpy())).hexdigest()


def state_hash(state):
    return digest_json({k: dict(hash=tensor_hash(v), shape=list(v.shape), dtype=str(v.dtype))
                        for k, v in sorted(state.items())})


def save_bundle(path, tensors, metadata):
    out = fresh_dir(path)
    torch.save({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, out/"tensors.pt")
    meta = dict(metadata, version=VERSION, file_sha256=file_hash(out/"tensors.pt"),
                tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_hash(v))
                         for k, v in tensors.items()})
    meta["artifact_hash"] = digest_json(meta)
    write_json(out/"artifact.json", meta)
    return meta


def load_bundle(path):
    p = Path(path)
    meta = read_json(p/"artifact.json")
    d = dict(meta)
    expected = d.pop("artifact_hash")
    require(digest_json(d) == expected, "Artifact metadata hash mismatch")
    require(meta["version"] == VERSION, "Unsupported artifact version")
    require(file_hash(p/"tensors.pt") == meta["file_sha256"], "Artifact tensor file corrupted")
    t = torch.load(p/"tensors.pt", map_location="cpu", weights_only=True)
    require(set(t) == set(meta["tensors"]), "Artifact tensor names mismatch")
    for k, value in t.items():
        spec = meta["tensors"][k]
        require(list(value.shape) == spec["shape"] and str(value.dtype) == spec["dtype"]
                and tensor_hash(value) == spec["sha256"], f"Artifact tensor mismatch: {k}")
    return t, meta


def load_table(path, expected):
    t, meta = load_bundle(path)
    for k, v in expected.items():
        require(meta.get(k) == v, f"Incompatible table metadata: {k}")
    require(meta.get("hooks") == HOOKS, "Incompatible residual hook contract")
    require(t["lookup"].dtype == torch.bfloat16 and not t["lookup"].requires_grad, "Invalid frozen lookup")
    require(bool(torch.isfinite(t["lookup"]).all()), "Nonfinite table values")
    n, d = t["lookup"].shape
    if meta["constructor"] in ("shallow", "contextual", "delta"):
        require(t["counts"].dtype == torch.int64 and t["counts"].shape == (n,)
                and bool((t["counts"] > 0).all()), "Invalid contextual counts")
        require(t["variance"].dtype == torch.float64 and t["variance"].shape == (n,)
                and bool(torch.isfinite(t["variance"]).all()) and bool((t["variance"] >= 0).all()),
                "Invalid contextual variance")
        require(t["master_sum"].dtype == torch.float32 and t["master_sum"].shape == (n, d)
                and bool(torch.isfinite(t["master_sum"]).all()), "Invalid master sums")
    if meta["constructor"] == "shuffled":
        require(t["permutation"].dtype == torch.int64 and t["permutation"].shape == (n,)
                and torch.equal(t["permutation"].sort().values, torch.arange(n)), "Invalid shuffled permutation")
    return t, meta


class Accumulator:
    def __init__(self, slots, width, device="cpu"):
        self.sums = torch.zeros(slots, width, dtype=torch.float32, device=device)
        self.counts = torch.zeros(slots, dtype=torch.int64, device=device)
        self.squares = torch.zeros(slots, dtype=torch.float64, device=device)

    def add(self, slots, values):
        hit = slots >= 0
        s = slots[hit].long().to(self.sums.device)
        z = values[hit].detach().to(self.sums.device, dtype=torch.float32)
        require(bool(torch.isfinite(z).all()), "Nonfinite writer states")
        self.sums.index_add_(0, s, z)
        self.counts.index_add_(0, s, torch.ones_like(s))
        self.squares.index_add_(0, s, z.double().square().sum(-1))

    def finish(self):
        require(bool((self.counts > 0).all()), "Unobserved top-N slot; counting/compiler mismatch")
        mean = self.sums/self.counts[:, None]
        variance = ((self.squares/self.counts-mean.double().square().sum(-1)).clamp_min(0)
                    /self.sums.shape[1])
        return dict(lookup=mean.to(torch.bfloat16), counts=self.counts, variance=variance,
                    master_sum=self.sums, squared_norm_sum=self.squares)
