"""Fresh full PCC probe, conditional control, and development-locked test.

Consumes a completed eligible screen, never its trained adapter weights.
There is no resume, gate override, or automatic pretraining launch.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import PreparedContexts, validate_matched_data
from .diagnostics import fingerprint, paired_adapters, run_preflight
from .evaluation import evaluate
from .objectives import optimizer
from .protocol import (MODEL_ID, REVISION, PAIRS, CONTEXT, TOKENS_PER_UPDATE,
                       SCREEN_UPDATES, SCREEN_DEV_TOKENS, FULL_SEED,
                       FULL_UPDATES, FULL_WARMUP, PROBE_EVAL_TOKENS, CALIBRATION_TOKENS)
from .screen import write_json, train_update
from .statistics import select_pair, probe_decision
from .training import calibrate_scale, student_update


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def screen_evidence(directory, backbone):
    directory = Path(directory)
    if (directory / "failure.json").exists() or read_json(directory / "complete.json")["status"] != "ok":
        raise ValueError("Full probe requires a successfully completed screen")
    provenance = read_json(directory / "provenance.json")
    if provenance.get("model_id") != MODEL_ID or provenance.get("model_revision") != REVISION:
        raise ValueError("Screen must identify the pinned pretrained model and revision")
    preflight = read_json(directory / "preflight.json")
    if (preflight.get("status") != "ok" or not preflight.get("checks") or
            not all(x.get("passed") is True for x in preflight["checks"].values()) or
            preflight.get("backbone_sha256_before_and_after") != fingerprint(backbone.model)):
        raise ValueError("Screen preflight/model identity does not match this backbone")
    losses, counts = {}, None
    names = ["Base"] + [f"{arm}-{s}-{d}" for s, d in PAIRS for arm in ("deep", "shallow")]
    for name in names:
        with np.load(directory / f"eval-{name}.npz", allow_pickle=False) as record:
            current = record["target_counts"]
            if not np.array_equal(record["sequence_indices"], np.arange(len(current))):
                raise ValueError("Screen sequence order is invalid")
            if counts is not None and not np.array_equal(current, counts):
                raise ValueError("Screen arm target counts differ")
            counts = current.copy()
            losses[name] = record["loss_sums"].copy()
    decision = select_pair(losses, counts)
    saved = read_json(directory / "decision.json")
    # Recompute all scientific fields; elapsed wall time is the only extra field.
    if any(saved.get(key) != value for key, value in decision.items()):
        raise ValueError("Saved screen decision disagrees with paired loss records")
    return decision, read_json(directory / "data.json")


def verify_slice(smaller, larger, start=0):
    """Screen contexts must be the recorded fixed slice of the full stream."""
    if type(start) is not int or start < 0 or start + len(smaller) > len(larger):
        raise ValueError("Invalid screen slice offset")
    for key in ("source", "preprocessing", "packing", "tokenizer_revision", "data_order_seed"):
        if smaller.metadata[key] != larger.metadata[key]:
            raise ValueError(f"Screen/full data differ: {key}")
    valid = smaller.valid
    other = larger.valid[start:start + len(smaller)]
    if not np.all(other[valid]) or not valid[:-1].all():
        raise ValueError("Screen slice must preserve complete contexts except its final row")
    for left, right in ((smaller.ids, larger.ids), (smaller.positions, larger.positions),
                        (smaller.segments, larger.segments)):
        if left is not None and not np.array_equal(left[valid], right[start:start + len(smaller)][valid]):
            raise ValueError("Screen is not the recorded token-identical full-stream slice")


def freeze(adapter):
    adapter.zero_grad(set_to_none=True)
    return adapter.eval().requires_grad_(False)


def checkpoint(output, name, adapter, s, d, scale=None):
    torch.save({"state_dict": adapter.state_dict(), "arm": name, "s": s, "d": d,
                "init_seed": FULL_SEED, "updates": FULL_UPDATES,
                "input_tokens": FULL_UPDATES * TOKENS_PER_UPDATE,
                "sigma_delta": scale, "model_id": MODEL_ID, "model_revision": REVISION},
               output / f"{name}.pt")


def probe(backbone, screen_dir, train_path, dev_path, test_path, output, *, microbatch=1, provenance=None, load_data=None):
    load_data = load_data or PreparedContexts
    per_update = TOKENS_PER_UPDATE // CONTEXT
    if microbatch < 1 or per_update % microbatch:
        raise ValueError("Microbatch must divide the global context batch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = backbone.model
    device = next(model.parameters()).device
    try:
        write_json(output / "provenance.json", provenance if provenance is not None else {
            "model_config": model.config.to_dict(), "torch": torch.__version__,
            "model_revision": None, "invoked_via_python_api": True})
        if model.dtype != torch.bfloat16:
            raise ValueError("Full probe requires a bf16 frozen backbone")
        selected, screen_data = screen_evidence(screen_dir, backbone)
        write_json(output / "screen-decision.json", selected)
        if selected["selected_pair"] is None:
            result = {"status": "ok", "decision": "stop_negative_screen", "test_unlocked": False,
                      "pretraining_authorized": False}
            write_json(output / "complete.json", result)
            return result
        s, d = selected["selected_pair"]
        train = load_data(train_path, "train", FULL_UPDATES * TOKENS_PER_UPDATE)
        dev = load_data(dev_path, "dev", PROBE_EVAL_TOKENS)
        validate_matched_data(train, dev)
        screen_train = load_data(screen_data["train_path"], "train", SCREEN_UPDATES * TOKENS_PER_UPDATE)
        screen_dev = load_data(screen_data["dev_path"], "dev", SCREEN_DEV_TOKENS)
        if screen_train.metadata != screen_data["train"] or screen_dev.metadata != screen_data["dev"]:
            raise ValueError("Screen context metadata changed since screening")
        verify_slice(screen_train, train)
        verify_slice(screen_dev, dev, screen_dev.metadata.get("probe_dev_start_context", 0))
        if max(train.ids.max(), dev.ids.max()) >= model.config.vocab_size:
            raise ValueError("Prepared input IDs exceed the pinned vocabulary")
        del screen_train, screen_dev
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        preflight = run_preflight(backbone)
        write_json(output / "preflight.json", preflight)
        write_json(output / "data.json", {"train": train.metadata, "dev": dev.metadata,
            "train_path": str(train.path), "dev_path": str(dev.path)})
        # Do not inspect, resolve, or open test_path before the development lock opens.
        config = {"selected_pair": [s, d], "full_adapter_init_seed": FULL_SEED,
            "updates": FULL_UPDATES, "warmup": FULL_WARMUP, "tokens_per_update": TOKENS_PER_UPDATE,
            "calibration_input_tokens": CALIBRATION_TOKENS, "eval_input_tokens": PROBE_EVAL_TOKENS,
            "microbatch": microbatch, "lambda_corr": 1., "smooth_l1_beta": 1.,
            "adapter": {"hidden_size": model.config.hidden_size, "heads": 2, "head_dim": 128,
                        "rms_norm_eps": model.config.rms_norm_eps, "dropout": 0.},
            "optimizer": {"name": "AdamW", "betas": [.9, .95], "eps": 1e-8,
                          "peak_lr": 3e-4, "final_lr": 3e-5, "matrix_weight_decay": .01,
                          "norm_bias_gate_vector_weight_decay": 0., "grad_clip": 1.},
            "precision": {"backbone": "bf16", "autocast": "bf16", "adapter_master": "fp32",
                          "loss_reductions": "fp32", "optimizer_moments": "fp32", "calibration_sum": "fp64"},
            "permutation": "global update; position/norm deciles; SeedSequence([20260922, update]); singleton fails",
            "test_consistency": "all five primary point NLL contrasts strictly negative; report CIs",
            "pretraining_authorized": False}
        write_json(output / "config.json", config)
        teacher, initial = paired_adapters(backbone, seed=FULL_SEED)
        initial_hash = fingerprint(initial)
        if fingerprint(teacher) != initial_hash:
            raise RuntimeError("Full-run paired initialization differs")
        torch.save({"state_dict": initial.state_dict(), "init_seed": FULL_SEED}, output / "initial.pt")
        write_json(output / "initialization.json", {"seed": FULL_SEED, "sha256": initial_hash,
            "screen_checkpoint_loaded": False, "shared_by_all_full_arms": True})
        teacher_opts = {"Privileged-Deep": optimizer(teacher)}
        phase_start = time.monotonic()
        with (output / "train-teacher.jsonl").open("x") as log:
            for update in range(1, FULL_UPDATES + 1):
                records = train_update(backbone, train, {"Privileged-Deep": (teacher, d)}, teacher_opts,
                    s, d, (update - 1) * per_update, per_update, microbatch, update,
                    total_updates=FULL_UPDATES, warmup=FULL_WARMUP)
                for record in records:
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(f"teacher update={update}/{FULL_UPDATES}", flush=True)
        teacher_seconds = time.monotonic() - phase_start
        freeze(teacher)
        del teacher_opts
        checkpoint(output, "Privileged-Deep", teacher, s, d)
        teacher_hash = fingerprint(teacher)
        calibration = calibrate_scale(backbone, teacher, train, s, d, CALIBRATION_TOKENS, microbatch)
        write_json(output / "calibration.json", calibration)
        scale = calibration["sigma_delta"]

        def train_students(names, permuted=False):
            arms = {name: copy.deepcopy(initial) for name in names}
            if any(fingerprint(arm) != initial_hash for arm in arms.values()):
                raise RuntimeError("Student initialization differs from fresh teacher initialization")
            opts = {name: optimizer(arm) for name, arm in arms.items()}
            phase_start = time.monotonic()
            if permuted:
                (output / "permutations").mkdir()
            with (output / ("train-permuted.jsonl" if permuted else "train-students.jsonl")).open("x") as log:
                for update in range(1, FULL_UPDATES + 1):
                    records, audit = student_update(backbone, teacher, train, arms, opts, s, d,
                        (update - 1) * per_update, per_update, microbatch, update, scale,
                        total_updates=FULL_UPDATES, warmup=FULL_WARMUP, permuted=permuted)
                    if audit is not None:
                        np.savez_compressed(output / "permutations" / f"update-{update:04d}.npz", **audit)
                    for record in records:
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    print(f"{'permuted' if permuted else 'students'} update={update}/{FULL_UPDATES}", flush=True)
            seconds = time.monotonic() - phase_start
            for name, arm in arms.items():
                freeze(arm)
                checkpoint(output, name, arm, s, d, scale)
            return arms, seconds

        students, student_seconds = train_students(("Shallow-ExtraAttn", "Student-PCC"))
        losses, counts = evaluate(backbone, teacher, students, dev, s, d, output / "dev", microbatch=microbatch)
        decision = probe_decision(losses, counts)
        write_json(output / "dev-initial-decision.json", decision)
        permuted_seconds = None
        if decision["conditional_control_required"]:
            control, permuted_seconds = train_students(("Target-Permuted",), permuted=True)
            extra, extra_counts = evaluate(backbone, teacher, control, dev, s, d,
                output / "dev-permuted", microbatch=microbatch, include_reference=False)
            if counts != extra_counts:
                raise ValueError("Conditional control used different evaluation target counts")
            losses.update(extra)
            students.update(control)
            decision = probe_decision(losses, counts)
        write_json(output / "dev-decision.json", decision)
        if fingerprint(teacher) != teacher_hash or fingerprint(model) != preflight["backbone_sha256_before_and_after"]:
            raise RuntimeError("Frozen teacher/backbone changed during the full probe")
        write_json(output / "frozen-decisions.json", {"config": config, "dev_decision": decision,
            "sigma_delta": scale, "teacher_sha256": teacher_hash,
            "student_sha256": {name: fingerprint(arm) for name, arm in students.items()},
            "backbone_sha256": preflight["backbone_sha256_before_and_after"],
            "interpretation": "All required dev gates passed; check locked-test directions once."
                if decision["all_dev_gates_pass"] else "Do not scale; failed gates: " + ", ".join(
                    name for name, passed in decision["gates"].items() if not passed)})
        result = {"status": "ok", "decision": "stop_dev_gates", "test_unlocked": False,
                  "pretraining_authorized": False, "pilot_specification_recommended": False}
        if decision["all_dev_gates_pass"] and test_path is None:
            result["decision"] = "validation_complete_test_not_supplied"
        elif decision["all_dev_gates_pass"]:
            write_json(output / "test-started.json", {"one_attempt": True, "dev_decisions_frozen": True})
            test = load_data(test_path, "test", PROBE_EVAL_TOKENS)
            validate_matched_data(train, test)
            validate_matched_data(dev, test)
            if test.ids.max() >= model.config.vocab_size:
                raise ValueError("Test input IDs exceed the pinned vocabulary")
            write_json(output / "test-data.json", {"metadata": test.metadata, "path": str(test.path)})
            test_losses, test_counts = evaluate(backbone, teacher, students, test, s, d,
                output / "test", microbatch=microbatch)
            confirm = probe_decision(test_losses, test_counts, split="test")
            write_json(output / "test-decision.json", confirm)
            result.update(test_unlocked=True, pilot_specification_recommended=confirm["directionally_consistent"],
                decision="draft_separate_pilot_specification" if confirm["directionally_consistent"] else "stop_test_inconsistent")
        write_json(output / "timing.json", {"total_seconds": time.monotonic() - started,
            "teacher_train_seconds": teacher_seconds, "paired_student_train_seconds": student_seconds,
            "permuted_train_seconds": permuted_seconds,
            "teacher_train_input_tokens_per_second": FULL_UPDATES * TOKENS_PER_UPDATE / teacher_seconds,
            "paired_student_input_tokens_per_second": 2 * FULL_UPDATES * TOKENS_PER_UPDATE / student_seconds,
            "permuted_input_tokens_per_second": FULL_UPDATES * TOKENS_PER_UPDATE / permuted_seconds if permuted_seconds else None,
            "all_clean_teacher_pass_costs_included": True,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        write_json(output / "complete.json", result)
        return result
    except BaseException as error:
        try:
            write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        except Exception as report_error:
            error.add_note(f"Could not save failure.json: {report_error}")
        raise
