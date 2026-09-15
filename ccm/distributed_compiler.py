"""Deterministic batch-sharded offline compiler and mandatory equivalence gate.

Preserve reference forward batch shapes; merge raw FP32 sums, int64 counts and
FP64 squared-norm sums, NEVER rank-local means/variances. Masters remain on CPU;
bounded collective chunks avoid a second full set of GPU accumulators.
"""
from itertools import islice
import os
from pathlib import Path
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from .artifacts import Accumulator, save_bundle
from .compiler import batches
from .contracts import require, fresh_dir, write_json, read_json, digest_json, HOOKS
from .data import collate
from .model import MemoryLM
from .runtime import load_model, to_device, sync, model_inputs
from .studies import SCALEUP, study_of, require_vocabulary

# Fixed before scientific compiler results. Do not tune to make a failed gate pass.
TOLERANCE = dict(mean_atol=1e-5, mean_rtol=1e-5, variance_atol=1e-5,
                 variance_rtol=1e-5, nll_atol=1e-5)


def merge_accumulators(accum, device, chunk_elements=1048576):
    if not dist.is_initialized():
        return
    for a in accum.values():
        for t in (a.sums, a.counts, a.squares):
            flat = t.view(-1)
            for i in range(0, flat.numel(), chunk_elements):
                chunk = flat[i:i+chunk_elements].to(device).clone()
                dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
                flat[i:i+len(chunk)].copy_(chunk.cpu())


def accumulate(model, stream, corpus, vocab, size, kinds, rank=0, world=1):
    device = next(model.parameters()).device
    acc = {kind: Accumulator(len(vocab.keys), model.backbone.config.hidden_size) for kind in kinds}
    tokens = 0
    for i, rows in enumerate(batches(stream, size)):
        if i % world != rank:
            continue
        b = to_device(collate(rows, corpus.meta["special_ids"], vocab), device)
        r = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                  position_ids=b["position_ids"], capture=True)
        shallow, deep = r["r2_pre_memory"].float(), r["deep_pre_final_norm"].float()
        for kind, a in acc.items():
            a.add(b["slots"], shallow if kind == "shallow" else deep if kind == "contextual" else deep-shallow)
        tokens += int(b["attention_mask"].sum())
    sync(device)
    return acc, tokens


def observed_tables(acc):
    # Validation subsets need not observe every full-vocabulary key. Unseen
    # rows are zero for BOTH paths; production finish() still requires all keys.
    n = acc.counts.clamp_min(1)
    mean = acc.sums/n[:, None]
    var = ((acc.squares/n-mean.double().square().sum(-1)).clamp_min(0)/mean.shape[1])
    return mean, var


def compare_accumulators(reference, candidate):
    result = {}
    for kind, a in reference.items():
        b = candidate[kind]
        require(torch.equal(a.counts, b.counts), "Distributed compiler counts differ")
        am, av = observed_tables(a)
        bm, bv = observed_tables(b)
        require(torch.allclose(am, bm, atol=TOLERANCE["mean_atol"], rtol=TOLERANCE["mean_rtol"]),
                "Distributed compiler means exceed frozen tolerance")
        require(torch.allclose(av, bv, atol=TOLERANCE["variance_atol"], rtol=TOLERANCE["variance_rtol"]),
                "Distributed compiler variance exceeds frozen tolerance")
        result[kind] = dict(counts_exact=True, mean_max_abs=float((am-bm).abs().max()),
                            variance_max_abs=float((av-bv).abs().max()), observed_slots=int((a.counts > 0).sum()))
    return result


def check_lookup_nll(writer, reference, candidate, corpus, vocab, rows):
    require(rows, "Empty compiler lookup validation set")
    device = next(writer.parameters()).device
    losses = []
    observed = reference["contextual"].counts > 0
    active_hits = 0
    for row in rows:
        slots, _ = vocab.route(row["tokens"], corpus.meta["special_ids"])
        active_hits += int(observed[torch.from_numpy(slots[slots >= 0]).long()].sum())
    require(active_hits > 0, "Compiler lookup validation must exercise observed memory rows")
    for acc in (reference, candidate):
        lookup = observed_tables(acc["contextual"])[0].bfloat16()
        require(bool(lookup[observed].abs().any()), "Lookup validation cannot use an all-zero table")
        m = MemoryLM(writer.backbone.config, "contextual", table=lookup)
        m.backbone.load_state_dict(writer.backbone.state_dict())
        # Zero W_v would make a broken table pass trivially. Use the same fixed
        # NONZERO reader in both checks; this is not fitted on dev data.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(710017)
            torch.nn.init.normal_(m.reader.wv.weight, std=.02)
        m.to(device=device, dtype=next(writer.parameters()).dtype).set_phase("eval")
        m.table = m.table.bfloat16()
        total, count = 0., 0
        for small in batches(rows, 4):
            b = to_device(collate(small, corpus.meta["special_ids"], vocab), device)
            r = m(**model_inputs(b), loss_chunk=128)
            total += float(r["losses"].double().sum())
            count += int(r["target_count"])
        require(count > 0, "No lookup validation targets")
        losses.append(total/count)
        del m
    require(abs(losses[0]-losses[1]) <= TOLERANCE["nll_atol"], "Distributed lookup NLL exceeds frozen tolerance")
    return dict(reference_nll=losses[0], distributed_nll=losses[1], abs_difference=abs(losses[0]-losses[1]),
                observed_memory_hits=active_hits)


def binding(args, corpus, vocab, ck, world, device):
    return dict(corpus_hash=corpus.meta["manifest_hash"], vocabulary_hash=vocab.hash,
                ordered_mapping_hash=vocab.mapping_hash, source_checkpoint_hash=ck["model_sha256"],
                source_code_hash=args.source_code_hash, world_size=world,
                microbatch_segments=args.microbatch_segments, device_type=device.type,
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                torch=torch.__version__, cuda=torch.version.cuda, tolerance=TOLERANCE)


@torch.no_grad()
def distributed_compile(args, corpus, vocab):
    require_vocabulary(corpus, vocab)
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    else:
        require(corpus.meta["engineering"], "CPU distributed compilation is engineering-only")
    require(args.reserved_gpus >= (world if device.type == "cuda" else 0), "Reserved GPUs below compiler world size")
    require(args.validation_batches > 0 and args.validation_eval_batches > 0, "Positive compiler validation sizes required")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=1))
    try:
        model, ck = load_model(args.checkpoint, device)
        require(ck["phase"] == "common" and ck["arm"] == "base" and ck["step"] == corpus.budget.common_steps
                and ck["corpus_hash"] == corpus.meta["manifest_hash"], "Need the exact completed common writer")
        require(ck.get("study", "pilot12") == study_of(corpus), "Distributed writer study mismatch")
        require(corpus.meta["engineering"] or model.backbone.config.num_hidden_layers == (28 if study_of(corpus) == SCALEUP else 12),
                "Distributed writer depth mismatch")
        model.set_phase("compile")
        spec = binding(args, corpus, vocab, ck, world, device)
        if not args.validate_distributed:
            require(args.compiler_validation is not None, "Distributed compilation requires a passing equivalence gate")
            proof = read_json(args.compiler_validation)
            require(proof.get("success") is True and proof.get("binding") == spec,
                    "Compiler validation must match writer/data/vocabulary/world/batch/code/device")
            require(digest_json({k: v for k, v in proof.items() if k != "validation_hash"}) == proof.get("validation_hash"),
                    "Compiler validation checksum mismatch")
        out = Path(args.output)
        if rank == 0:
            fresh_dir(out)
        if world > 1:
            dist.barrier()
        start, cpu = time.monotonic(), time.process_time()
        kinds = ("shallow", "contextual") if study_of(corpus) == SCALEUP else ("shallow", "contextual", "delta")
        reference = None
        if args.validate_distributed:
            rows = list(islice(corpus.segments("compile"), args.validation_batches*args.microbatch_segments))
            require(len(rows) >= world*args.microbatch_segments, "Validation needs at least one full batch per rank")
            if rank == 0:
                reference, _ = accumulate(model, rows, corpus, vocab, args.microbatch_segments, kinds)
            if world > 1:
                dist.barrier()
            stream = rows
        else:
            stream = corpus.segments("compile")
        active_start = time.monotonic()
        acc, tokens = accumulate(model, stream, corpus, vocab, args.microbatch_segments, kinds, rank, world)
        merge_accumulators(acc, device)
        t = torch.tensor(tokens, dtype=torch.int64, device=device)
        if world > 1:
            dist.all_reduce(t)
        active = time.monotonic()-active_start
        if rank == 0:
            if args.validate_distributed:
                statistics = compare_accumulators(reference, acc)
                eval_rows = list(islice(corpus.segments("dev"), args.validation_eval_batches*4))
                nll = check_lookup_nll(model, reference, acc, corpus, vocab, eval_rows)
                report = dict(success=True, binding=spec, statistics=statistics, lookup=nll,
                              tokens=int(t), validation_batches=args.validation_batches,
                              validation_eval_batches=args.validation_eval_batches)
                report["validation_hash"] = digest_json(report)
                write_json(out/"validation.json", report)
            else:
                require(int(t) == corpus.budget.compile_tokens, "Distributed compiler token count mismatch")
                for a in acc.values():
                    require(np.array_equal(a.counts.numpy(), vocab.counts), "Distributed count/write alignment mismatch")
                base = dict(source_checkpoint_path=str(Path(args.checkpoint).resolve()),
                            source_checkpoint_hash=ck["model_sha256"], backbone_contract=model.backbone_contract(),
                            tokenizer=corpus.meta["provenance"]["tokenizer"], corpus_hash=corpus.meta["manifest_hash"],
                            vocabulary_hash=vocab.hash, compile_segment_hash=corpus.meta["files"]["compile.segments.jsonl"],
                            hooks=HOOKS, source_seed=ck["seed"], config=vars(args), source_code_hash=args.source_code_hash)
                control_start = time.monotonic()
                for kind, a in acc.items():
                    save_bundle(out/kind, a.finish(), dict(base, constructor=kind, online_capable=False))
                n, width = len(vocab.keys), model.backbone.config.hidden_size
                isolated = torch.empty(n, width, dtype=torch.bfloat16)
                for i in range(0, n, args.isolated_batch):
                    keys = vocab.keys[i:i+args.isolated_batch]
                    ids = torch.tensor(np.stack([keys >> 32, keys & 0xffffffff], axis=1), device=device)
                    r = model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                              position_ids=torch.tensor([0, 1], device=device).expand(len(ids), -1), capture=True)
                    isolated[i:i+len(ids)] = r["deep_pre_final_norm"][:, 1].cpu().bfloat16()
                save_bundle(out/"isolated", dict(lookup=isolated), dict(base, constructor="isolated", online_capable=False))
                permutation = torch.randperm(n, generator=torch.Generator().manual_seed(200000+ck["seed"]))
                lookup = acc["contextual"].finish()["lookup"]
                save_bundle(out/"shuffled", dict(lookup=lookup[permutation], permutation=permutation),
                            dict(base, constructor="shuffled", permutation_seed=200000+ck["seed"], online_capable=False))
                active += time.monotonic()-control_start
        accounting = torch.tensor([active, time.process_time()-cpu], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(accounting)
            dist.barrier()
        wall = time.monotonic()-start
        if rank == 0 and not args.validate_distributed:
            report = dict(success=True, tokens=int(t), world_size=world, wall_seconds=wall,
                          active_gpu_hours=float(accounting[0])/3600 if device.type == "cuda" else 0,
                          reserved_node_gpu_hours=wall*args.reserved_gpus/3600 if device.type == "cuda" else 0,
                          process_cpu_seconds=float(accounting[1]), input_token_bytes=int(t)*4,
                          output_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                          note="Active task time includes forward/transfer/merge waits, not GPU-kernel-only time; token bytes exclude index/checkpoint reads")
            write_json(out/"complete.json", report)
        return report if rank == 0 else None
    finally:
        if world > 1:
            dist.destroy_process_group()
