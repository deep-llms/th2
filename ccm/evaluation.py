"""Teacher-forced NLL, causal hit/miss diagnostics, and paired segment records."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contracts import require, read_json, write_json, fresh_dir, digest_json, file_hash
from .data import collate
from .runtime import load_model, model_inputs, to_device, sync
from .compiler import batches
from .artifacts import load_bundle
from .studies import require_vocabulary


def bins(values):
    return np.searchsorted(np.quantile(values, np.arange(1, 10)/10), values, side="right")


def validate_final_access(role, final, lock, corpus, metadata):
    require(role in ("dev", "val"), "Only held-out roles may enter the evaluator")
    require(not final or role == "val", "Final-evaluation flag is only for locked D_val")
    if role == "val":
        require(final and lock is not None, "Locked D_val requires --final-evaluation and a frozen decision lock")
        d = read_json(lock)
        payload = dict(d)
        expected_hash = payload.pop("lock_hash", None)
        require(digest_json(payload) == expected_hash, "Final lock checksum mismatch")
        require(d.get("choices_locked") is True and d.get("corpus_hash") == corpus.meta["manifest_hash"],
                "Invalid final-evaluation lock")
        require(metadata["phase"] == "stage2" and metadata["arm"] in d["required_arms"],
                "Final evaluation must cover predeclared Stage-2 comparison arms")
        require(metadata["model_sha256"] in d["checkpoint_hashes"], "Checkpoint not named in final lock")
        require(metadata["metadata_hash"] in d["checkpoint_metadata_hashes"], "Checkpoint run contract not named in final lock")


@torch.no_grad()
def evaluate(args, corpus, vocab):
    model, meta = load_model(args.checkpoint, args.device)
    require(meta["corpus_hash"] == corpus.meta["manifest_hash"], "Evaluation corpus mismatch")
    require_vocabulary(corpus, vocab)
    if model.arm != "base":
        require(meta["vocabulary_hash"] == vocab.hash, "Model/vocabulary mismatch")
    validate_final_access(args.role, args.final_evaluation, args.final_lock, corpus, meta)
    model.set_phase("eval")
    if model.arm in ("shallow", "contextual", "delta"):
        require(args.diagnostic_table is not None, "Contextual constructors require variance diagnostics")
    out = fresh_dir(args.output)
    frequency = bins(vocab.counts)
    variance = None
    if args.diagnostic_table:
        tensors, table_meta = load_bundle(args.diagnostic_table)
        require(table_meta["vocabulary_hash"] == vocab.hash and table_meta["corpus_hash"] == corpus.meta["manifest_hash"],
                "Diagnostic table mismatch")
        expected_writer = meta["model_sha256"] if meta["phase"] == "common" else meta["source_checkpoint_hash"]
        require(table_meta["source_checkpoint_hash"] == expected_writer, "Diagnostic table uses a different writer checkpoint")
        require("variance" in tensors, "Diagnostic constructor has no variance")
        variance = bins(tensors["variance"].numpy())
    if args.role == "val":
        claims = Path(str(args.final_lock)+".evaluations")
        claims.mkdir(exist_ok=True)
        # Reserve only after input checks; failure during evaluation requires review.
        # Different legitimate controls can have byte-identical model weights
        # (e.g. an identity permutation in a tiny shuffle test). Identify the
        # named seed/arm/run contract, not weights alone.
        write_json(claims/f"{meta['metadata_hash']}.json", dict(checkpoint_hash=meta["model_sha256"],
                   checkpoint_metadata_hash=meta["metadata_hash"],
                   output=str(out.resolve()), status="claimed"))
    totals = {k: [0.0, 0] for k in ("overall", "hit", "miss", "eligible_miss")}
    freq_totals = [[0.0, 0] for _ in range(10)]
    var_totals = [[0.0, 0] for _ in range(10)]
    gate_values = []
    hits = eligible = input_tokens = 0
    device = torch.device(args.device)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    with (out/"segments.jsonl").open("x") as f:
        for rows in batches(corpus.segments(args.role), args.microbatch_segments):
            b = to_device(collate(rows, corpus.meta["special_ids"], vocab), device)
            r = model(**model_inputs(b), loss_chunk=args.loss_chunk)
            loss = r["losses"].double().cpu().numpy()
            target = b["targets"].cpu().numpy() != -100
            slot = b["slots"].cpu().numpy()
            hit = (slot >= 0) & target
            e = b["eligible"].cpu().numpy() & target
            masks = dict(overall=target, hit=hit, miss=target & ~hit, eligible_miss=e & ~hit)
            if model.gates is not None:
                gate_values.append(model.gates.cpu().numpy()[hit])
            input_tokens += int(b["attention_mask"].sum())
            hits += int(hit.sum())
            eligible += int(e.sum())
            for i, row in enumerate(rows):
                record = {k: row[k] for k in ("doc_id", "content_hash", "segment_id")}
                for name, mask in masks.items():
                    s, n = float(loss[i][mask[i]].sum()), int(mask[i].sum())
                    record[name] = [s, n]
                    totals[name][0] += s
                    totals[name][1] += n
                for name, indexes, aggregate in (("frequency", frequency, freq_totals), ("variance", variance, var_totals)):
                    record[name] = []
                    for j in range(10):
                        mask = hit[i] & (indexes[slot[i].clip(min=0)] == j) if indexes is not None else np.zeros_like(hit[i])
                        pair = [float(loss[i][mask].sum()), int(mask.sum())]
                        record[name].append(pair)
                        aggregate[j][0] += pair[0]
                        aggregate[j][1] += pair[1]
                f.write(json.dumps(record, allow_nan=False)+"\n")
            del r, b
    sync(device)
    wall = time.monotonic()-start
    gate = np.concatenate(gate_values) if gate_values else np.array([])
    report = dict(study=meta.get("study", "pilot12"), arm=meta["arm"], phase=meta["phase"], seed=meta["seed"], role=args.role,
                  step=meta["step"], total_steps=meta["total_steps"], engineering=meta["engineering"],
                  source_checkpoint_hash=meta["source_checkpoint_hash"],
                  paired_initial_reader_hash=meta["paired_initial_reader_hash"],
                  checkpoint_hash=meta["model_sha256"], corpus_hash=meta["corpus_hash"], vocabulary_hash=vocab.hash,
                  metrics={k: dict(loss_sum=s, count=n, nll=s/n if n else None) for k, (s, n) in totals.items()},
                  eligible_hit_rate=hits/eligible if eligible else None,
                  overall_memory_active_rate=hits/totals["overall"][1], frequency=freq_totals,
                  variance=var_totals, diagnostic_table=args.diagnostic_table,
                  diagnostic_table_hash=table_meta["artifact_hash"] if args.diagnostic_table else None,
                  gate=dict(mean=float(gate.mean()), p10=float(np.quantile(gate, .1)),
                            median=float(np.median(gate)), p90=float(np.quantile(gate, .9))) if len(gate) else None,
                  input_tokens=input_tokens, wall_seconds=wall, tokens_per_second=input_tokens/wall,
                  peak_accelerator_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  table_bytes=model.table.numel()*model.table.element_size() if model.reader else 0,
                  final_evaluation=args.final_evaluation)
    lock = read_json(args.final_lock) if args.role == "val" else None
    report["final_lock_hash"] = lock["lock_hash"] if lock else None
    report["statistical_policy"] = lock["statistical_policy"] if lock else None
    report["segments_sha256"] = file_hash(out/"segments.jsonl")
    write_json(out/"metrics.json", report)
    return report
