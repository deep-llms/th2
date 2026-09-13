"""Memory-off offline value constructors and the pre-reader coverage guard."""
from pathlib import Path
import time
import numpy as np
import torch
from .contracts import require, fresh_dir, write_json, HOOKS
from .data import collate
from .runtime import load_model, to_device, sync
from .artifacts import Accumulator, save_bundle


def batches(stream, size):
    pending = []
    for row in stream:
        pending.append(row)
        if len(pending) == size:
            yield pending
            pending = []
    if pending:
        yield pending


def coverage(corpus, vocab, output):
    require(vocab.metadata["corpus_hash"] == corpus.meta["manifest_hash"], "Coverage vocabulary/corpus mismatch")
    hits = eligible = targets = 0
    for row in corpus.segments("dev"):
        slots, e = vocab.route(row["tokens"], corpus.meta["special_ids"])
        hits += int((slots >= 0).sum())
        eligible += int(e.sum())
        targets += len(slots)-1
    require(eligible > 0 and targets > 0, "No eligible development positions")
    report = dict(passed=hits/eligible >= 0.2, eligible_hit_rate=hits/eligible,
                  overall_memory_active_rate=hits/targets, hits=hits, eligible=eligible, targets=targets,
                  corpus_hash=corpus.meta["manifest_hash"], vocabulary_hash=vocab.hash,
                  role="dev", before_reader_training=True)
    write_json(output, report)
    return report


@torch.no_grad()
def compile_tables(args, corpus, vocab):
    require(vocab.metadata["corpus_hash"] == corpus.meta["manifest_hash"], "Compiler vocabulary/corpus mismatch")
    model, ck = load_model(args.checkpoint, args.device)
    require(ck["phase"] == "common" and ck["arm"] == "base" and ck["step"] == corpus.budget.common_steps,
            "Writer must be the completed memory-free theta_4B")
    require(ck["corpus_hash"] == corpus.meta["manifest_hash"], "Writer/data mismatch")
    model.set_phase("compile")
    device = torch.device(args.device)
    out = fresh_dir(args.output)
    n, width = len(vocab.keys), model.backbone.config.hidden_size
    # CPU masters avoid competing with model accelerator memory. Per-microbatch
    # transfer/accumulation cost is measured, not hidden in a speed claim.
    accum = {kind: Accumulator(n, width) for kind in ("shallow", "contextual", "delta")}
    sync(device)
    start, cpu_start = time.monotonic(), time.process_time()
    tokens = 0
    for rows in batches(corpus.segments("compile"), args.microbatch_segments):
        batch = to_device(collate(rows, corpus.meta["special_ids"], vocab), device)
        r = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                  position_ids=batch["position_ids"], capture=True)
        shallow, deep = r["r2_pre_memory"].float(), r["r12_pre_final_norm"].float()
        for kind, values in (("shallow", shallow), ("contextual", deep), ("delta", deep-shallow)):
            accum[kind].add(batch["slots"], values)
        tokens += int(batch["attention_mask"].sum())
        del batch, r, shallow, deep
    require(tokens == corpus.budget.compile_tokens, "Wrong compiler input-token budget")
    for a in accum.values():
        require(np.array_equal(a.counts.numpy(), vocab.counts), "Count/write target alignment mismatch")
    sync(device)
    contextual_wall = time.monotonic()-start
    base = dict(source_checkpoint_path=str(Path(args.checkpoint).resolve()),
                source_checkpoint_hash=ck["model_sha256"], backbone_contract=model.backbone_contract(),
                tokenizer=corpus.meta["provenance"]["tokenizer"],
                corpus_hash=corpus.meta["manifest_hash"], vocabulary_hash=vocab.hash,
                compile_segment_hash=corpus.meta["files"]["compile.segments.jsonl"],
                hooks=HOOKS, source_seed=ck["seed"], config=vars(args),
                source_code_hash=args.source_code_hash)
    for kind, a in accum.items():
        save_bundle(out/kind, a.finish(), dict(base, constructor=kind, online_capable=kind in ("contextual", "delta")))
    # Isolated control is intentionally different: [a,b] has no teacher-forced
    # target at b, but b's final hidden state IS the isolated constructor.
    isolated = torch.empty(n, width, dtype=torch.bfloat16)
    iso_start = time.monotonic()
    for i in range(0, n, args.isolated_batch):
        keys = vocab.keys[i:i+args.isolated_batch]
        ids = torch.tensor(np.stack([keys >> 32, keys & 0xffffffff], axis=1), device=device)
        r = model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                  position_ids=torch.tensor([0, 1], device=device).expand(len(ids), -1), capture=True)
        isolated[i:i+len(ids)] = r["r12_pre_final_norm"][:, 1].to("cpu", dtype=torch.bfloat16)
    save_bundle(out/"isolated", dict(lookup=isolated), dict(base, constructor="isolated", online_capable=False))
    g = torch.Generator().manual_seed(200000+ck["seed"])
    permutation = torch.randperm(n, generator=g)
    contextual = accum["contextual"].finish()["lookup"]
    save_bundle(out/"shuffled", dict(lookup=contextual[permutation], permutation=permutation),
                dict(base, constructor="shuffled", permutation_seed=200000+ck["seed"], online_capable=False))
    sync(device)
    wall = time.monotonic()-start
    report = dict(success=True, tokens=tokens, wall_seconds=wall,
                  contextual_pass_seconds=contextual_wall, isolated_and_control_seconds=time.monotonic()-iso_start,
                  process_cpu_seconds=time.process_time()-cpu_start,
                  process_cpu_hours=(time.process_time()-cpu_start)/3600,
                  accelerator_hours=wall/3600 if device.type == "cuda" else 0,
                  input_token_bytes=corpus.budget.compile_tokens*4,
                  output_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                  notes="CPU time is process CPU time; accelerator-hours are allocated single-device wall time")
    write_json(out/"complete.json", report)
    return report
