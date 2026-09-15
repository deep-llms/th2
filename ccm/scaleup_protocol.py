"""Fail-closed stability provenance and preregistered 28L replication decision."""
from pathlib import Path
import os
import torch
import transformers
from .contracts import require, read_json, file_hash, digest_json, write_json
from .studies import SCALEUP


def stability_binding(args, corpus):
    return dict(study=SCALEUP, seed=args.seed, corpus_hash=corpus.meta["manifest_hash"],
                config_sha256=file_hash(Path(args.model_config)/"config.json"),
                source_code_hash=args.source_code_hash, engineering=args.engineering,
                microbatch_segments=args.microbatch_segments, loss_chunk=args.loss_chunk,
                activation_checkpointing=args.activation_checkpointing,
                world_size=int(os.environ.get("WORLD_SIZE", "1")), torch=torch.__version__,
                cuda=torch.version.cuda, transformers=transformers.__version__,
                common_lr=getattr(args, "common_lr", 3e-4))


def stability_record(args, corpus, success, steps, error=None):
    record = dict(stability_binding(args, corpus), success=success, steps=steps,
                  total_schedule_steps=corpus.budget.common_steps,
                  objective_numerical_failure=not success, error=error,
                  note="Short full-schedule prefix only; not a proof of full-run stability")
    record["record_hash"] = digest_json(record)
    return record


def checked_record(path):
    r = read_json(path)
    require(digest_json({k: v for k, v in r.items() if k != "record_hash"}) == r.get("record_hash"),
            "Stability evidence checksum mismatch")
    return r


def validate_stability_inputs(args, corpus):
    if args.phase != "common" or args.engineering:
        return
    expected = stability_binding(args, corpus)
    if expected["common_lr"] == 2e-4:
        path = getattr(args, "lr_failure_report", None)
        require(path is not None, "Fallback LR requires recorded objective failure at 3e-4")
        failure = checked_record(path)
        require(all(failure.get(k) == v for k, v in dict(expected, common_lr=3e-4).items())
                and failure.get("success") is False and failure.get("objective_numerical_failure") is True,
                "Fallback must follow objective numerical failure, not early loss preference")
    if getattr(args, "stability_steps", None) is None:
        path = getattr(args, "stability_report", None)
        require(path is not None, "Run and verify the short 28L stability check before full common training")
        r = checked_record(path)
        require(r.get("success") is True and r.get("steps", 0) > 0
                and all(r.get(k) == v for k, v in expected.items())
                and r.get("total_schedule_steps") == corpus.budget.common_steps,
                "Stability evidence does not match the fresh full common run")


def replication_decision(isolated_report, shuffled_report, output):
    reports = [read_json(p) for p in (isolated_report, shuffled_report)]
    evals = []
    for r, control in zip(reports, ("isolated", "shuffled")):
        require(r.get("cluster") == "doc_id" and r.get("replicates") == 10000
                and r.get("bootstrap_seed") == 20260913 and r.get("population") == "overall"
                and r.get("cross_seed_coupling") == "independent"
                and r.get("backbone_seeds") == [17] and not r.get("final_report"),
                "Decision requires the locked single-seed development comparisons")
        require(len(r["left"]) == len(r["right"]) == 1, "Wrong first-backbone panel")
        a, b = [read_json(Path(p)/"metrics.json") for p in (r["left"][0], r["right"][0])]
        require(a.get("study") == b.get("study") == SCALEUP and a["arm"] == "contextual"
                and b["arm"] == control and a["role"] == b["role"] == "dev"
                and a.get("engineering") is False and b.get("engineering") is False
                and a.get("seed") == b.get("seed") == 17
                and not a.get("final_evaluation") and not b.get("final_evaluation")
                and a["phase"] == b["phase"] == "stage2"
                and a["step"] == b["step"] == a["total_steps"] == b["total_steps"] == 7629,
                "Decision must use completed 28L Stage-2 results, not pilot or heldout results")
        # Recompute the report from its checksummed per-document records.
        from .statistics import paired_arrays, bootstrap
        expected = bootstrap([paired_arrays(r["left"][0], r["right"][0], "overall", "doc_id")],
                             coupling="independent")
        require(all(r[k] == expected[k] for k in ("mean_difference", "lower95", "upper95")),
                "Decision report differs from paired source records")
        evals.append(a)
    for key in ("checkpoint_hash", "corpus_hash", "vocabulary_hash", "source_checkpoint_hash",
                "paired_initial_reader_hash", "segments_sha256"):
        require(evals[0][key] == evals[1][key],
                f"Primary reports use different Contextual evidence: {key}")
    passed = [r["upper95"] < 0 for r in reports]
    result = dict(study=SCALEUP, seed=17, primary_passes=passed,
                  decision="replicate" if all(passed) else "ambiguous" if any(passed) else "diagnose",
                  automatic_launch=False, reports_sha256=[file_hash(p) for p in (isolated_report, shuffled_report)])
    write_json(output, result)
    return result
