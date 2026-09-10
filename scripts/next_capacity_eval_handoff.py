"""Prepare and strictly verify the new-arm evaluation/fine-tuning handoff.

This module never launches GPU work or mutates checkpoints.  It freezes the
98-checkpoint comparison grid, validates local inputs, and checks the queue
completion records before a subsequent stage may begin.
"""
import argparse
import json
import math
from pathlib import Path

from eval.runtime import offline

offline()

from capacity_allocation.data import load_text_data, preprocess_text, sha256, write_json
from eval.benchmarks import load_tasks, task_plan
from eval.diagnostic_data import load_bundle, tokenizer_identity
from eval.diagnostics_checkpoint import training_provenance
from eval.runtime import checkpoint_identity


ARMS = (
    "T768", "P512-128-384", "FixedResidual", "WNW",
    "A640", "A768", "A768-Direct",
    "D-1024", "O1024-I232", "O1024-I256",
    "C-Direct", "D-Direct", "T512", "O1280",
)
EVAL_STEPS = (250, 500, 1000, 2000, 3000, 4000, 5000)
FINETUNE_TASKS = ("hellaswag", "arc_easy", "xnli")
FINETUNE_SEEDS = (42, 123, 456)
EXPECTED_BENCHMARK_TASKS = 78
EXPECTED_SCORED_TARGETS = 9_829_694
EFFECTIVE_TRAIN_BATCH = 512
BLOCK_SIZE = 2048


def checkpoint_grid(training_root, steps=EVAL_STEPS):
    return [(arm, step, training_root / arm / f"checkpoint-{step}")
            for arm in ARMS for step in steps]


def _read(path):
    return json.loads(Path(path).read_text())


def _finite_numbers(value):
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def prepare(args):
    from transformers import AutoTokenizer

    manifest, counts, _, datasets = load_bundle(args.bundle)
    assert manifest["languages"] == ["en"] and set(datasets) == {"en"}
    assert manifest["validation"]["en"]["scored_targets"] == EXPECTED_SCORED_TARGETS
    assert len(counts) == 151936

    plan, unavailable = task_plan("en")
    assert len(plan) == EXPECTED_BENCHMARK_TASKS and not unavailable
    tasks = load_tasks(plan, args.benchmarks)
    assert set(tasks) == {row["task"] for row in plan}
    coverage = {}
    for name, task in tasks.items():
        docs = task.eval_docs
        assert len(docs) > 0
        coverage[name] = dict(eval_examples=len(docs), fingerprint=docs._fingerprint)
    for name in ("hellaswag", "arc_easy", "xnli_en"):
        assert tasks[name].has_training_docs()
        docs = tasks[name].training_docs()
        assert len(docs) > 0
        coverage[name].update(training_examples=len(docs),
                              training_fingerprint=docs._fingerprint)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    tokenizer.model_max_length = 10**30
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    assert tokenizer_identity(tokenizer) == manifest["tokenizer_sha256"]
    packed = preprocess_text(load_text_data(args.eval_data, ("en",)), tokenizer,
        block_size=BLOCK_SIZE, num_proc=160, batch_size=1000, cache_dir=args.cache)
    assert len(packed) == 4802 and len(packed) * (BLOCK_SIZE - 1) == EXPECTED_SCORED_TARGETS
    assert packed._fingerprint == manifest["validation"]["en"]["packed_fingerprint"]

    checkpoints = {}
    for arm, step, checkpoint in checkpoint_grid(args.training_root):
        state = _read(checkpoint / "trainer_state.json")
        assert state["global_step"] == step
        provenance = training_provenance(checkpoint, manifest)
        assert provenance["status"] == "verified" and provenance["global_step"] == step
        assert provenance["effective_batch_size"] == EFFECTIVE_TRAIN_BATCH
        assert provenance["consumed_input_tokens"] == step * EFFECTIVE_TRAIN_BATCH * BLOCK_SIZE
        name = f"{arm}_step{step}"
        checkpoints[name] = dict(arm=arm, step=step,
            identity=checkpoint_identity(checkpoint), training=provenance)
    assert len(checkpoints) == len(ARMS) * len(EVAL_STEPS)
    write_json(args.output, dict(success=True, schema="next_capacity_eval_handoff_v1",
        arms=list(ARMS), eval_steps=list(EVAL_STEPS), checkpoints=checkpoints,
        benchmark_coverage=coverage, benchmark_tasks=[row["task"] for row in plan],
        ppl_fingerprint=packed._fingerprint, scored_targets=EXPECTED_SCORED_TARGETS,
        diagnostic_manifest_sha256=sha256(args.bundle / "manifest.json"),
        diagnostic_probe_ids=manifest["probe_ids"]))
    print("ALL_98_CHECKPOINTS_AND_EVALUATION_INPUTS_VERIFIED", flush=True)


def _stage_records(output, expected):
    complete = _read(output / "complete.json")
    records = complete["completed"]
    assert complete["success"] is True and len(records) == expected
    assert len({record["name"] for record in records}) == expected
    assert all(record["exit_code"] == 0 and record["error"] is None for record in records)
    for record in records:
        result = Path(record["result"]).resolve()
        assert result.is_relative_to(output.resolve()) and result.is_file()
    return records


def verify_eval(args):
    reference = _read(args.reference)
    assert reference["success"] is True and reference["schema"] == "next_capacity_eval_handoff_v1"
    identities = {row["identity"]["path"]: (name, row)
                  for name, row in reference["checkpoints"].items()}
    expected = len(identities) * 5
    records = _stage_records(args.output, expected)
    seen = set()
    counts = {"ppl": 0, "benchmarks": 0, "frequency": 0, "spectra": 0, "gradients": 0}
    for record in records:
        item = _read(record["result"])
        assert item["success"] is True and item["languages"] == ["en"]
        checkpoint = item["checkpoint"]["path"]
        name, checkpoint_row = identities[checkpoint]
        assert item["checkpoint"] == checkpoint_row["identity"]
        if "ppl" in item:
            kind = "ppl"
            row = item["ppl"]["by_language"]["en"]
            assert row["scored_targets"] == reference["scored_targets"]
            assert row["data_fingerprint"] == reference["ppl_fingerprint"]
            assert math.isfinite(row["nll"]) and math.isfinite(row["ppl"])
        elif "diagnostics" in item:
            assert item["diagnostic_manifest_sha256"] == reference["diagnostic_manifest_sha256"]
            assert item["training"] == checkpoint_row["training"]
            assert item["settings"]["precision"] == "bf16"
            assert len(item["diagnostics"]) == 1
            kind, result = next(iter(item["diagnostics"].items()))
            assert kind in ("frequency", "spectra", "gradients")
            if kind == "frequency":
                row = result["by_language"]["en"]["overall"]
                buckets = result["by_language"]["en"]["buckets"]
                assert row["scored_targets"] == reference["scored_targets"]
                assert sum(bucket["scored_targets"] for bucket in buckets.values()) == row["scored_targets"]
                assert math.isfinite(row["nll"]) and math.isfinite(row["ppl"])
            elif kind == "spectra":
                assert set(result["interfaces"]) == {"input", "output"}
                for interface in result["interfaces"].values():
                    for view in ("raw", "effective"):
                        assert set(interface[view]) == {"seen", "all"}
                        assert all(row["status"] == "ok" for row in interface[view].values())
                assert _finite_numbers(result)
            else:
                assert result["optimizer_steps"] == 0 and result["precision"] == "fp32"
                assert result["master_weights"] == "torch.float32"
                assert result["probe_ids"] == reference["diagnostic_probe_ids"]
                assert result["scored_targets"] == 8 * (BLOCK_SIZE - 1)
                assert _finite_numbers(result)
        else:
            kind = "benchmarks"
            assert set(item["benchmarks"]) == set(reference["benchmark_tasks"])
            assert item["lm_eval_version"] == "0.4.10"
            for task, result in item["benchmarks"].items():
                assert result["samples"]["effective"] == reference["benchmark_coverage"][task]["eval_examples"]
                assert _finite_numbers(result)
        key = (name, kind)
        assert key not in seen
        seen.add(key)
        counts[kind] += 1
    assert all(value == len(identities) for value in counts.values())
    write_json(args.summary, dict(success=True, checkpoints=len(identities), jobs=expected,
                                  jobs_by_kind=counts))
    print("ALL_490_EVALUATION_AND_DIAGNOSTIC_JOBS_VERIFIED", flush=True)


def verify_finetune(args):
    reference = _read(args.reference)
    final = {row["identity"]["path"]: (name, row)
             for name, row in reference["checkpoints"].items() if row["step"] == 5000}
    expected = len(final) * len(FINETUNE_TASKS) * len(FINETUNE_SEEDS)
    records = _stage_records(args.output, expected)
    seen = set()
    for record in records:
        item = _read(record["result"])
        assert item["success"] is True and item["languages"] == ["en"]
        checkpoint = item["checkpoint"]["path"]
        name, checkpoint_row = final[checkpoint]
        assert item["checkpoint"] == checkpoint_row["identity"]
        assert item["task"] in FINETUNE_TASKS and item["seed"] in FINETUNE_SEEDS
        assert item["train_language"] == "en" and item["precision"] == "bf16"
        assert item["master_weights"] == "fp32"
        config = item["config"]
        assert config["epochs"] == 3 and config["lr"] == 2e-5 and config["max_length"] == 256
        assert config["batch_size"] == (16 if item["task"] == "hellaswag" else 32)
        expected_steps = ((item["used_examples"] + config["batch_size"] - 1) // config["batch_size"]) * 3
        assert item["training"]["steps"] == expected_steps
        assert item["used_examples"] + item["skipped_examples"] == item["source_examples"]
        plan, _ = task_plan("en", [item["task"]])
        assert set(item["benchmarks"]) == {row["task"] for row in plan}
        for task, result in item["benchmarks"].items():
            assert result["samples"]["effective"] == reference["benchmark_coverage"][task]["eval_examples"]
            assert _finite_numbers(result)
        key = (name, item["task"], item["seed"])
        assert key not in seen
        seen.add(key)
    assert len(seen) == expected
    write_json(args.summary, dict(success=True, checkpoints=len(final), jobs=expected,
        tasks=list(FINETUNE_TASKS), seeds=list(FINETUNE_SEEDS)))
    print("ALL_126_FINAL_CHECKPOINT_FINETUNE_JOBS_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--training-root", type=Path, required=True)
    prep.add_argument("--benchmarks", type=Path, required=True)
    prep.add_argument("--eval-data", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--bundle", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for mode in ("verify-eval", "verify-finetune"):
        command = subparsers.add_parser(mode)
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "verify-eval":
        verify_eval(args)
    else:
        verify_finetune(args)


if __name__ == "__main__":
    main()
