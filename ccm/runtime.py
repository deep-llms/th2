"""Local training/checkpoint runtime; optional torchrun DDP, no job management."""
from contextlib import nullcontext
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import Qwen3Config
import transformers
from .contracts import require, write_json, read_json, fresh_dir, file_hash, digest_json, seed_bundle, schedule, VERSION
from .artifacts import state_hash, load_table
from .model import MemoryLM, pilot_config
from .data import microbatches, collate


def device_context(requested):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(requested)
    if world > 1:
        require(device.type == "cuda", "Production DDP requires CUDA/NCCL")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    return device, rank, world


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def model_inputs(batch):
    return {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids", "slots", "targets")}


def checkpoint_meta(path):
    p = Path(path)
    meta = read_json(p/"checkpoint.json")
    payload = dict(meta)
    expected = payload.pop("metadata_hash", None)
    require(digest_json(payload) == expected, "Checkpoint metadata checksum mismatch")
    require(file_hash(p/"model.pt") == meta["model_sha256"], "Checkpoint model checksum mismatch")
    require(file_hash(p/"config.json") == meta["config_sha256"], "Checkpoint config checksum mismatch")
    return meta


def save_checkpoint(path, model, meta, optimizer=None):
    p = fresh_dir(path)
    torch.save(model.state_dict(), p/"model.pt")
    model.backbone.config.to_json_file(p/"config.json")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), p/"optimizer.pt")
    record = dict(meta, model_sha256=file_hash(p/"model.pt"), config_sha256=file_hash(p/"config.json"),
                  backbone_contract=model.backbone_contract(),
                  reader_hash=state_hash(model.reader.state_dict()) if model.reader else None,
                  table_hash=state_hash({"table": model.table}) if model.reader else None)
    if optimizer is not None:
        record["optimizer_sha256"] = file_hash(p/"optimizer.pt")
    record["metadata_hash"] = digest_json(record)
    write_json(p/"checkpoint.json", record)
    return record


def load_model(path, device="cpu"):
    meta = checkpoint_meta(path)
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    config._attn_implementation = "sdpa"
    state = torch.load(Path(path)/"model.pt", map_location="cpu", weights_only=True)
    table = state.get("table")
    model = MemoryLM(config, meta["arm"], table=table,
                     slots=len(table) if table is not None else 0)
    dtype = next(v.dtype for k, v in state.items() if k.endswith("embed_tokens.weight"))
    model.to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    if model.reader is not None and model.arm != "grad":
        model.table = model.table.to(torch.bfloat16)
    require(model.backbone_contract() == meta["backbone_contract"], "Checkpoint architecture contract mismatch")
    return model.to(device), meta


class MasterAdamW:
    """bf16 model/gradients with fp32 Adam moments AND persistent fp32 weights.

    A plain AdamW on bf16 Parameters would keep low-precision optimizer state.
    Save the fp32 masters explicitly; a rounded model state cannot restore them.
    """
    def __init__(self, groups):
        self.pairs = []
        master_groups = []
        for group in groups:
            pairs = [(p, torch.nn.Parameter(p.detach().float().clone())) for p in group["params"]]
            self.pairs.extend(pairs)
            master_groups.append(dict(group, params=[m for _, m in pairs]))
        self.inner = torch.optim.AdamW(master_groups, betas=(0.9, 0.95), eps=1e-8)

    def zero_grad(self):
        self.inner.zero_grad(set_to_none=True)
        for p, _ in self.pairs:
            p.grad = None

    def step(self):
        for p, m in self.pairs:
            m.grad = None if p.grad is None else p.grad.detach().float()
        norm = torch.nn.utils.clip_grad_norm_([m for _, m in self.pairs], 1.0, error_if_nonfinite=True)
        self.inner.step()
        with torch.no_grad():
            for p, m in self.pairs:
                p.copy_(m)
        return float(norm)

    def state_dict(self):
        return dict(optimizer=self.inner.state_dict(), masters=[m.detach().cpu() for _, m in self.pairs])


def optimizer_groups(model, phase):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "table":
            mult, decay = 5.0, 0.0
        elif name.startswith("reader."):
            mult = 1.0 if phase == "stage1" else 5.0
            decay = 0.01 if p.ndim >= 2 else 0.0
        else:
            mult, decay = 1.0, 0.1 if p.ndim >= 2 else 0.0
        groups.setdefault((mult, decay), []).append(p)
    result = [dict(params=ps, multiplier=m, weight_decay=d, lr=0.0) for (m, d), ps in groups.items()]
    require(result, "No trainable parameters")
    return result


def require_coverage(path, corpus, vocab):
    c = read_json(path)
    require(c["passed"] and c["eligible_hit_rate"] >= 0.2, "Coverage guard failed")
    require(c["corpus_hash"] == corpus.meta["manifest_hash"] and c["vocabulary_hash"] == vocab.hash,
            "Coverage artifact belongs to different data/vocabulary")
    return c


def delta_policy_matches(decision, seed, common_hash):
    """Legacy screen remains supported; per-seed decisions bind their own writer."""
    if decision.get("replication_policy") == "seed17_then_all":
        return decision.get("decision_seed") == 17
    if decision.get("replication_policy") == "per_seed":
        r = decision.get("results", {})
        names = ("hit_vs_contextual", "hit_vs_shuffled", "miss_vs_contextual")
        return (decision.get("decision_seed") == seed
                and decision.get("source_checkpoint_hash") == common_hash
                and decision.get("overall_safeguard") is True
                and all(n in r and r[n].get("replicates") == 10000
                        and r[n].get("bootstrap_seed") == 20260913 for n in names)
                and r["hit_vs_contextual"]["upper95"] < 0
                and r["hit_vs_shuffled"]["upper95"] < 0
                and r["miss_vs_contextual"]["upper95"] <= .002)
    return False


def train(args, corpus, vocab=None):
    device, rank, world = device_context(args.device)
    require(args.engineering or device.type == "cuda", "Pilot training requires bf16 CUDA; use --engineering for toy CPU")
    seeds = seed_bundle(args.seed)
    torch.manual_seed(seeds["backbone"])
    phase, arm = args.phase, args.arm
    require(not args.online, "Online self-writing is not implemented in pilot v1.")
    require(phase != "stage1" or arm != "base", "Stage-1 Base is evaluated without adaptation")
    require(phase != "stage2" or arm != "shallow", "Shallow is Stage-1 diagnostic only")
    require(corpus.meta["engineering"] == args.engineering, "Engineering/pilot corpus mode mismatch")
    common_meta = None
    artifact = None
    if phase == "common":
        require(arm == "base" and not args.checkpoint, "Common run must initialize a memory-free model from scratch")
        if args.engineering:
            c = Qwen3Config.from_pretrained(args.model_config, local_files_only=True)
            c._attn_implementation = "sdpa"
        else:
            c = pilot_config(args.model_config)
        model = MemoryLM(c)
    else:
        require(args.checkpoint is not None, "Stage 1/2 requires theta_4B")
        base, common_meta = load_model(args.checkpoint)
        require(common_meta["phase"] == "common" and common_meta["arm"] == "base" and
                common_meta["step"] == corpus.budget.common_steps, "Must start from the exact completed common checkpoint")
        require(common_meta["corpus_hash"] == corpus.meta["manifest_hash"] and common_meta["seed"] == args.seed,
                "Common checkpoint data/seed mismatch")
        c = base.backbone.config
        table = None
        if arm != "base":
            require(vocab is not None and args.coverage is not None, "Memory arms require vocabulary and pre-reader coverage")
            require_coverage(args.coverage, corpus, vocab)
            require(vocab.metadata["corpus_hash"] == corpus.meta["manifest_hash"], "Vocabulary corpus mismatch")
            if arm != "grad":
                require(args.table is not None, "Compiled arm requires its checked table")
                tensors, artifact = load_table(args.table, dict(constructor=arm, vocabulary_hash=vocab.hash,
                    corpus_hash=corpus.meta["manifest_hash"], source_checkpoint_hash=common_meta["model_sha256"],
                    tokenizer=corpus.meta["provenance"]["tokenizer"], backbone_contract=base.backbone_contract()))
                table = tensors["lookup"]
                require(table.shape == (len(vocab.keys), c.hidden_size), "Table capacity/width differs from vocabulary/model")
            if phase == "stage2" and arm == "delta":
                require(args.delta_decision is not None, "Delta requires a pre-registered Stage-1 decision")
                decision = read_json(args.delta_decision)
                require(decision.get("include_delta") is True and decision.get("corpus_hash") == corpus.meta["manifest_hash"]
                        and decision.get("vocabulary_hash") == vocab.hash
                        and delta_policy_matches(decision, args.seed, common_meta["model_sha256"]),
                        "Invalid Delta inclusion decision")
        model = MemoryLM(c, arm, table=table, slots=len(vocab.keys) if vocab else 0,
                         reader_seed=seeds["reader"], table_seed=seeds["grad_table"])
        model.backbone.load_state_dict(base.backbone.state_dict(), strict=True)
        del base
    dtype = torch.float32 if args.engineering and device.type == "cpu" else torch.bfloat16
    model.to(device=device, dtype=dtype)
    # Frozen lookup storage stays bf16 even in fp32 CPU tests.
    if model.reader is not None and arm != "grad":
        model.table = model.table.to(torch.bfloat16)
    model.set_phase(phase)
    model.train()
    if args.activation_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                      find_unused_parameters=False) if world > 1 else model
    opt = MasterAdamW(optimizer_groups(model, phase))
    out = Path(args.output)
    if rank == 0:
        fresh_dir(out)
    if world > 1:
        dist.barrier()
    total = dict(common=corpus.budget.common_steps, stage1=corpus.budget.adapt_steps,
                 stage2=corpus.budget.continue_steps)[phase]
    peak = dict(common=3e-4, stage1=5e-4, stage2=1.5e-4)[phase]
    warm = 0.05 if phase == "stage1" else 0.02
    meta = dict(version=VERSION, phase=phase, arm=arm, seed=args.seed, seeds=seeds,
                engineering=args.engineering, corpus_hash=corpus.meta["manifest_hash"],
                tokenizer=corpus.meta["provenance"]["tokenizer"], vocabulary_hash=vocab.hash if vocab else None,
                source_checkpoint_hash=common_meta["model_sha256"] if common_meta else None,
                table_artifact_hash=artifact["artifact_hash"] if artifact else None,
                paired_initial_reader_hash=state_hash(model.reader.state_dict()) if model.reader else None,
                initial_grad_table_hash=state_hash({"table": model.table}) if arm == "grad" else None,
                config=vars(args), torch=torch.__version__, transformers=transformers.__version__,
                optimizer="AdamW fp32 masters/moments, reset per phase", model_dtype=str(dtype),
                world_size=world, peak_lr=peak, total_steps=total)
    if rank == 0:
        write_json(out/"run.json", meta)
    sync(device)
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last = 0
    log = (out/"train.jsonl").open("x") if rank == 0 else None
    try:
        for step, rows in enumerate(corpus.optimizer_batches(phase, seeds["data"]), 1):
            require(step <= total, "Too many optimizer batches")
            require(len(rows) >= world, "Too few segments for distributed batch")
            targets_total = sum(len(r["tokens"])-1 for r in rows)
            require(targets_total > 0, "Batch has no targets")
            local = list(microbatches(rows[rank::world], args.microbatch_segments))
            opt.zero_grad()
            step_loss = torch.zeros((), device=device, dtype=torch.float64)
            for j, small in enumerate(local):
                batch = to_device(collate(small, corpus.meta["special_ids"], vocab), device)
                ctx = wrapped.no_sync() if world > 1 and j < len(local)-1 else nullcontext()
                with ctx:
                    result = wrapped(**model_inputs(batch), loss_chunk=args.loss_chunk)
                    loss = result["loss_sum"]*world/targets_total
                    loss.backward()
                    step_loss += result["loss_sum"].detach().double()
                del result, batch, loss
            lr = schedule(step, total, peak, warm)
            for g in opt.inner.param_groups:
                g["lr"] = lr*g["multiplier"]
            grad_norm = opt.step()
            if world > 1:
                dist.all_reduce(step_loss)
            sync(device)
            elapsed = time.monotonic()-start
            peak_bytes = torch.tensor(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                                      device=device, dtype=torch.int64)
            if world > 1:
                dist.all_reduce(peak_bytes, op=dist.ReduceOp.MAX)
            record = dict(step=step, input_tokens=step*corpus.budget.batch_tokens,
                          nll=float(step_loss)/targets_total, lr=lr, grad_norm=grad_norm,
                          elapsed_seconds=elapsed, tokens_per_second=step*corpus.budget.batch_tokens/elapsed,
                          peak_cuda_bytes=int(peak_bytes))
            if rank == 0:
                import json
                log.write(json.dumps(record)+"\n")
                log.flush()
                if step == 1 or step % args.log_every == 0:
                    print(json.dumps(record), flush=True)
                if step % args.save_every == 0 or step == total:
                    save_checkpoint(out/f"checkpoint-{step}", model, dict(meta, step=step, metrics=record), opt)
            if world > 1:
                dist.barrier()
            last = step
        require(last == total, "Training ended before exact step budget")
        if rank == 0:
            write_json(out/"complete.json", dict(success=True, phase=phase, arm=arm, step=last,
                       input_tokens=last*corpus.budget.batch_tokens, checkpoint=f"checkpoint-{last}"))
    finally:
        if log:
            log.close()
        if world > 1:
            dist.destroy_process_group()
