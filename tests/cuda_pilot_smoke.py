"""Opt-in CUDA/NCCL tests; synthetic inputs, no downloads or process stopping.

Single GPU end-to-end (all offline arms, real trainer/compiler/evaluator):
  PYTHONPATH=.:tests python tests/cuda_pilot_smoke.py --pipeline
Four GPUs, bf16/checkpointing/Adam/save-reload acceptance:
  PYTHONPATH=. torchrun --standalone --nproc_per_node=4 tests/cuda_pilot_smoke.py
Add --full-size for 12 layers / 151936 vocab / 262144 memory slots / length 2048.
Full-size is a synthetic shape test, NOT verified Base-asset provenance or a
production throughput benchmark. Run only on GPUs already verified available.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Qwen3Config

from ccm.artifacts import state_hash
from ccm.model import MemoryLM
from ccm.runtime import MasterAdamW, optimizer_groups, save_checkpoint, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--full-size", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this test must not silently fall back to CPU")
    torch.set_num_threads(1)
    if args.pipeline:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.full_size:
            raise ValueError("Pipeline mode is single GPU and tiny only")
        from test_pilot import EndToEndTests
        EndToEndTests.device = "cuda:0"
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EndToEndTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        print("CUDA OFFLINE PIPELINE PASS", flush=True)
        return

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        world = dist.get_world_size()
        c = Qwen3Config(vocab_size=151936 if args.full_size else 64,
                       hidden_size=1024 if args.full_size else 32,
                       intermediate_size=3072 if args.full_size else 64,
                       num_hidden_layers=12 if args.full_size else 3,
                       num_attention_heads=16 if args.full_size else 4,
                       num_key_value_heads=8 if args.full_size else 2,
                       head_dim=128 if args.full_size else 8,
                       tie_word_embeddings=True, attention_dropout=0., pad_token_id=0)
        c._attn_implementation = "sdpa"
        n = 262144 if args.full_size else 8
        length = 2048 if args.full_size else 16
        for phase, arm in (("common", "base"), ("stage1", "contextual"),
                           ("stage1", "grad"), ("stage2", "contextual"), ("stage2", "grad")):
            torch.manual_seed(17)
            table = torch.randn(n, c.hidden_size, dtype=torch.bfloat16) if arm == "contextual" else None
            model = MemoryLM(c, arm, table=table, slots=n).to(device=device, dtype=torch.bfloat16)
            del table
            model.set_phase(phase)
            model.train()
            assert model.backbone.lm_head.weight is model.backbone.model.embed_tokens.weight
            if args.full_size:
                assert sum(p.numel() for p in model.backbone.parameters()) == 344354816
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            wrapped = DDP(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)
            opt = MasterAdamW(optimizer_groups(model, phase))
            for g in opt.inner.param_groups:
                g["lr"] = 5e-4 * g["multiplier"]
            if arm != "base":
                before = state_hash({"table": model.table})
            ids = (torch.arange(length, device=device)[None] % 29) + 2 + rank
            slots = (torch.arange(length, device=device)[None] % n).to(torch.int32)
            slots[:, ::3] = -1
            slots[:, 0] = slots[:, -1] = -1
            targets = torch.cat((ids[:, 1:], torch.full_like(ids[:, :1], -100)), 1)
            batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                         position_ids=torch.arange(length, device=device)[None], slots=slots, targets=targets)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            losses = []
            # Uneven local accumulation counts exercise no_sync + one reduction.
            local_count = 1 + rank % 2
            global_targets = (length-1) * sum(1 + r % 2 for r in range(world))
            for step in range(3):
                opt.zero_grad()
                loss_sum = torch.zeros((), device=device, dtype=torch.float64)
                for j in range(local_count):
                    with wrapped.no_sync() if j < local_count-1 else nullcontext():
                        out = wrapped(**batch, loss_chunk=128)
                        loss = out["loss_sum"] * world / global_targets
                        loss.backward()
                        loss_sum += out["loss_sum"].detach().double()
                        del out, loss
                assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
                norm = opt.step()
                assert norm >= 0
                dist.all_reduce(loss_sum)
                losses.append(float(loss_sum / global_targets))
            torch.cuda.synchronize()
            elapsed, peak = time.monotonic()-start, torch.cuda.max_memory_allocated()
            if arm != "base":
                after = state_hash({"table": model.table})
                assert (before != after) == (arm == "grad")
                if arm == "contextual":
                    assert model.table.grad is None
            # Exact state agreement, not just similar aggregate losses.
            hashes = [None] * world
            dist.all_gather_object(hashes, state_hash(model.state_dict()))
            assert len(set(hashes)) == 1, "DDP replicas diverged"
            if rank == 0:
                model.eval()
                with torch.no_grad():
                    expected = model(**batch)["hidden"].clone()
                with tempfile.TemporaryDirectory(prefix="ccm-cuda-checkpoint-") as td:
                    ck = Path(td)/"checkpoint"
                    save_checkpoint(ck, model, dict(arm=arm, engineering=True), opt if not args.full_size else None)
                    restored, _ = load_model(ck, device)
                    restored.eval()
                    with torch.no_grad():
                        actual = restored(**batch)["hidden"]
                    assert torch.equal(actual, expected), "Save/reload changed forward output"
                    del restored, actual, expected
                print(json.dumps(dict(result="PASS", phase=phase, arm=arm, world=world,
                                      full_size=args.full_size, losses=losses, seconds=elapsed,
                                      rank0_peak_GiB=peak/2**30)), flush=True)
            dist.barrier()
            del wrapped, model, opt, batch
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            print("CUDA NCCL BF16 ACCEPTANCE PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
