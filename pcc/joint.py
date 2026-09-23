"""Offline joint-training CLI. Separate from the frozen PCC screen/probe."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .joint_config import ARMS, SEEDS, JointSettings, load_config, plan


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    modes = result.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "train", "capacity", "report", "manifest"):
        command = modes.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name in ("train", "capacity", "report"):
            command.add_argument("--data-dir", required=True, type=Path)
        if name in ("train", "capacity"):
            command.add_argument("--arm", required=True, choices=ARMS)
            command.add_argument("--seed-index", type=int, choices=(0, 1), default=0)
            command.add_argument("--physical-gpu", required=True, type=int)
        if name == "train":
            command.add_argument("--resume", type=Path)
        if name == "report":
            command.add_argument("--runs-dir", required=True, type=Path)
        if name == "manifest":
            command.add_argument("--physical-gpu", required=True, type=int)
            command.add_argument("--data-dir", type=Path,
                                 help="Reuse already prepared inputs, verified by each run")
    return result


def manifest(config_path, physical_gpu, output, data_dir=None):
    if physical_gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    jobs = [{"name": "inputs", "argv": ["{python}", "-u", "-m", "pcc.joint", "prepare",
             "--config", str(config_path.resolve()), "--output", "{run_dir}/inputs"],
             "required_outputs": [{"path": "inputs/complete.json", "json_equals": {"status": "ok"}}]}]
    inputs = "{run_dir}/inputs" if data_dir is None else str(Path(data_dir).resolve())
    if data_dir is not None:
        jobs = []
    for seed_index in range(2):
        for arm in ARMS:
            name = f"seed-{seed_index}-{arm}"
            jobs.append({"name": name, "gpus": [physical_gpu], "argv": ["{python}", "-u", "-m", "pcc.joint", "train",
                "--config", str(config_path.resolve()), "--data-dir", inputs, "--arm", arm,
                "--seed-index", str(seed_index), "--physical-gpu", str(physical_gpu),
                "--output", "{run_dir}/" + name],
                "required_outputs": [{"path": f"{name}/complete.json", "json_equals": {
                    "status": "ok", "updates": 1536, "input_tokens": 50331648, "arm": arm}}]})
    jobs.append({"name": "report", "argv": ["{python}", "-u", "-m", "pcc.joint", "report",
        "--config", str(config_path.resolve()), "--data-dir", inputs,
        "--runs-dir", "{run_dir}", "--output", "{run_dir}/report"],
        "required_outputs": [{"path": "report/complete.json", "json_equals": {"status": "ok"}}]})
    with Path(output).open("x") as handle:
        json.dump({"jobs": jobs}, handle, indent=2)


def prepare(config, output):
    from transformers import AutoTokenizer
    from .data import SampledDataLoader, save_contexts, validate_matched_data
    from .input_check import describe_contexts
    from .screen import write_json
    from .protocol import REVISION
    settings = JointSettings()
    output.mkdir(parents=True, exist_ok=False)
    try:
        model_path = Path(config["model_path"])
        if model_path.name != REVISION and not model_path.name.endswith("-" + REVISION):
            raise ValueError("Model path must identify the locked revision")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        loader = SampledDataLoader(tokenizer, output / "data-cache",
                                   train_documents=config.get("train_documents"))
        train = loader(config["train_data"], "train", settings.updates * settings.tokens_per_update)
        dev = loader(config["val_data"], "dev", settings.dev_tokens)
        validate_matched_data(train, dev)
        from transformers import AutoConfig
        from .protocol import validate_config
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        validate_config(model_config)
        if max(train.ids.max(), dev.ids.max()) >= model_config.vocab_size:
            raise ValueError("Input IDs exceed the pinned model vocabulary")
        streams = {}
        for name, data in (("train", train), ("dev", dev)):
            save_contexts(output / f"{name}.npz", data.ids, data.valid, data.positions, data.metadata, data.segments)
            streams[name] = describe_contexts(data)
        write_json(output / "complete.json", {"status": "ok", "plan": plan(config), "streams": streams})
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def load_inputs(config, directory, seed_index):
    import copy
    import numpy as np
    from .data import PreparedContexts, validate_matched_data
    from .input_check import context_fingerprint
    from .joint_training import require
    settings = JointSettings()
    record = json.loads((directory / "complete.json").read_text())
    # JSON normalization makes tuples in the in-memory plan compare as arrays.
    require(record["status"] == "ok" and record["plan"] == json.loads(json.dumps(plan(config))), "Prepared inputs/config mismatch")
    train = PreparedContexts(directory / "train.npz", "train", settings.updates * settings.tokens_per_update)
    dev = PreparedContexts(directory / "dev.npz", "dev", settings.dev_tokens)
    validate_matched_data(train, dev)
    for name, data in (("train", train), ("dev", dev)):
        require(context_fingerprint(data) == record["streams"][name]["context_sha256"], "Prepared inputs changed")
    # The first pool order already uses 20260922. The second seed permutes the
    # exact same pool, not a different prefix of the full corpus.
    if seed_index == 1:
        indices = np.random.default_rng(SEEDS[1][1]).permutation(len(train))
        train = copy.copy(train)
        train.ids, train.valid, train.positions = (a[indices] for a in (train.ids, train.valid, train.positions))
        if train.segments is not None:
            train.segments = train.segments[indices]
    return train, dev, {"train": context_fingerprint(train), "dev": context_fingerprint(dev)}


def run_arm(args, config):
    import hashlib
    import random
    import socket
    import time
    import numpy as np
    import torch
    import transformers
    from .joint_model import JointQwen
    from .joint_training import code_identity, train, require
    from .model import load_local
    from .screen import write_json
    from .protocol import REVISION
    torch.set_num_threads(4)
    data, dev, fingerprints = load_inputs(config, args.data_dir, args.seed_index)
    seed, order_seed = SEEDS[args.seed_index]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from scripts.gpu_status import require_free
    require_free([args.physical_gpu])
    backbone, _ = load_local(config["model_path"], torch.device("cuda:0"))
    model = JointQwen(backbone.model, args.arm, seed=seed)
    # Verify the actual pretrained 2048-token path, not only tiny fixtures.
    context = data.batch(0, 1, "cuda:0")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model.model.model(input_ids=context.input_ids,
            attention_mask=context.additive_mask(torch.float32),
            position_ids=context.position_ids, use_cache=False).last_hidden_state
        actual = model(context)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    del context, reference, actual
    watch = {"early": model.model.model.layers[0].mlp.down_proj.weight,
             "late": model.model.model.layers[-1].mlp.down_proj.weight}
    if model.adapter is not None:
        watch["branch"] = model.adapter.out.weight
    original = {name: value.detach().cpu().clone() for name, value in watch.items()} if args.mode == "capacity" else {}
    # Authentic input identity and exact software evidence travel with checkpoints.
    weights = Path(config["model_path"]) / "model.safetensors"
    with weights.open("rb") as handle:
        weights_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    identity = {"settings": asdict(JointSettings()), "arm": args.arm,
                "purpose": "capacity_check" if args.mode == "capacity" else "scientific_training",
                "adapter_seed": seed, "data_order_seed": order_seed, "config": config,
                "inputs": fingerprints, "model_revision": REVISION, "weights_sha256": weights_hash,
                "code": code_identity(), "mixed_precision": True,
                "torch": str(torch.__version__), "transformers": str(transformers.__version__)}
    started = time.monotonic()
    report = train(model, data, dev, args.output, identity, microbatch=config["microbatch"],
                   resume=getattr(args, "resume", None), stop_after=2 if args.mode == "capacity" else None)
    if args.mode == "capacity":
        # Bounded test weights are retained as diagnostics, never used to start a run.
        import json as json_module
        records = [json_module.loads(line) for line in (args.output / "train.jsonl").read_text().splitlines()]
        require(len(records) == 2 and report["status"] == "stopped_at_step", "Incomplete capacity check")
        changed = {name: not torch.equal(original[name], value.detach().cpu()) for name, value in watch.items()}
        require(all(changed.values()), "Capacity check did not update expected parameters")
        saved = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=True)
        require(saved["identity"] == identity and saved["update"] == 2, "Checkpoint did not roundtrip")
        require(saved["next_context"] == 32, "Capacity checkpoint cursor is wrong")
        write_json(args.output / "capacity.json", {"status": "ok", "arm": args.arm,
            "updates": 2, "input_tokens": 65536, "microbatch": config["microbatch"],
            "host": socket.gethostname(), "physical_gpu": args.physical_gpu,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "initial_native_equivalence": True, "parameters_changed": changed,
            "checkpoint_roundtrip": True,
            "step_seconds": [r["seconds"] for r in records], "elapsed_seconds": time.monotonic() - started,
            "scientific_run": False})
    return report


def report_results(args, config):
    import csv
    import numpy as np
    from .joint_training import require, audit_final_checkpoint
    from .statistics import paired_bootstrap
    from .screen import write_json
    args.output.mkdir(parents=True, exist_ok=False)
    reports, curves, training_curves, timings = [], [], [], []
    try:
        common_code = common_weights = None
        settings = JointSettings()
        for seed_index in range(2):
            _, dev, fingerprints = load_inputs(config, args.data_dir, seed_index)
            eligible = dev.valid[:, :-1] & dev.valid[:, 1:]
            if dev.segments is not None:
                eligible &= dev.segments[:, :-1] == dev.segments[:, 1:]
            expected_counts = eligible.sum(-1)
            losses, counts = {}, None
            for arm in ARMS:
                directory = args.runs_dir / f"seed-{seed_index}-{arm}"
                completed = json.loads((directory / "complete.json").read_text())
                identity = json.loads((directory / "identity.json").read_text())
                require(not (directory / "failure.json").exists(), "Run has a failure record")
                require(completed["status"] == "ok" and completed["updates"] == settings.updates
                        and completed["input_tokens"] == settings.updates * settings.tokens_per_update
                        and completed["arm"] == arm, "Incomplete arm")
                require(identity["inputs"] == fingerprints and identity["config"] == config
                        and identity["purpose"] == "scientific_training"
                        and identity["settings"] == asdict(JointSettings()) and identity["arm"] == arm
                        and (identity["adapter_seed"], identity["data_order_seed"]) == SEEDS[seed_index], "Unmatched arm identity")
                if common_code is None:
                    common_code, common_weights = identity["code"], identity["weights_sha256"]
                require(identity["code"] == common_code and identity["weights_sha256"] == common_weights, "Mixed code/model versions")
                require((directory / "final.pt").is_file(), "Missing final checkpoint")
                audit = audit_final_checkpoint(directory / "final.pt", identity)
                with np.load(directory / "eval.npz", allow_pickle=False) as values:
                    require(np.array_equal(values["sequence_indices"], np.arange(len(values["target_counts"]))), "Wrong evaluation order")
                    current = values["target_counts"]
                    require(np.array_equal(current, expected_counts), "Wrong validation target counts")
                    if counts is not None:
                        require(np.array_equal(counts, current), "Unmatched evaluation targets")
                    counts = current.copy()
                    losses[arm] = values["loss_sums"].copy()
                    require(abs(float(losses[arm].sum() / counts.sum()) - completed["nll"]) < 1e-12, "Evaluation summary mismatch")
                for line in (directory / "validation.jsonl").read_text().splitlines():
                    curves.append({"seed_index": seed_index, "arm": arm, **json.loads(line)})
                train_log = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
                require(train_log == audit["history"]["train"], "Checkpoint and training log differ")
                training_curves.extend({"seed_index": seed_index, "arm": arm, **r} for r in train_log)
                timings.append({"seed_index": seed_index, "arm": arm,
                    "training_seconds": sum(r.get("seconds", 0.) for r in train_log),
                    "elapsed_seconds_this_invocation": completed.get("elapsed_seconds_this_invocation"),
                    "cuda_peak_allocated_bytes": completed.get("cuda_peak_allocated_bytes")})
                require([r["update"] for r in train_log] == list(range(1, settings.updates + 1)), "Wrong completed update history")
                require(all(r["input_tokens"] == r["update"] * settings.tokens_per_update for r in train_log), "Wrong completed token history")
            stats = paired_bootstrap(losses, counts, resamples=2000, seed=20260922)
            contrasts = {}
            for control in ("Base", "Shallow"):
                contrasts[control] = {"deep_minus_control": stats["nll"]["Deep"] - stats["nll"][control],
                    "ci95": np.quantile(stats["samples"]["Deep"] - stats["samples"][control], [.025, .975]).tolist()}
            passed = all(c["ci95"][1] < 0 for c in contrasts.values()) and contrasts["Shallow"]["deep_minus_control"] <= -.0005
            reports.append({"seed_index": seed_index, "nll": stats["nll"], "contrasts": contrasts, "passed": passed})
        with (args.output / "validation-curves.csv").open("x") as handle:
            writer = csv.DictWriter(handle, fieldnames=["seed_index", "arm", "update", "nll", "target_tokens"])
            writer.writeheader(); writer.writerows(curves)
        with (args.output / "training-curves.csv").open("x") as handle:
            writer = csv.DictWriter(handle, fieldnames=["seed_index", "arm", "update", "input_tokens", "nll", "grad_norm", "seconds"], extrasaction="ignore")
            writer.writeheader(); writer.writerows(training_curves)
        # Standalone plot for inspection after unattended execution.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for seed_index, axis in enumerate(axes):
            for arm in ARMS:
                values = [r for r in curves if r["seed_index"] == seed_index and r["arm"] == arm]
                axis.plot([r["update"] for r in values], [r["nll"] for r in values], label=arm)
            axis.set(title=f"Paired seed {seed_index + 1}", xlabel="Optimizer update", ylabel="Fixed validation NLL")
            axis.legend()
        figure.tight_layout()
        figure.savefig(args.output / "validation-curves.png", dpi=160)
        plt.close(figure)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for seed_index, axis in enumerate(axes):
            by_arm = {arm: {r["update"]: r["nll"] for r in curves if r["seed_index"] == seed_index and r["arm"] == arm} for arm in ARMS}
            steps = sorted(by_arm["Deep"])
            require(all(set(by_arm[arm]) == set(steps) for arm in ARMS), "Unmatched monitoring updates")
            for control in ("Base", "Shallow"):
                axis.plot(steps, [by_arm["Deep"][s] - by_arm[control][s] for s in steps], label=f"Deep − {control}")
            axis.axhline(0, color="grey", linestyle="--")
            axis.set(title=f"Paired seed {seed_index + 1}", xlabel="Optimizer update", ylabel="Validation NLL difference")
            axis.legend()
        figure.tight_layout()
        figure.savefig(args.output / "validation-contrasts.png", dpi=160)
        plt.close(figure)
        result = {"status": "ok", "stage": "joint-v1", "seeds": reports, "exploratory": True,
                  "decision": "recommend_distillation_design" if all(r["passed"] for r in reports) else "no_consistent_joint_advantage",
                  "bootstrap_resamples": 2000, "bootstrap_seed": 20260922,
                  "pretraining_authorized": False, "remote_launch": False, "timings": timings}
        text = ["# Joint-training result", "", result["decision"], "", "Exploratory local validation; no remote launch.", "",
                "| Seed | Base NLL | Shallow NLL | Deep NLL | Deep − shallow, 95% CI | Pass |",
                "|---|---:|---:|---:|---|---|"]
        for row in reports:
            nll, contrast = row["nll"], row["contrasts"]["Shallow"]
            text.append(f"| {row['seed_index'] + 1} | {nll['Base']:.8f} | {nll['Shallow']:.8f} | {nll['Deep']:.8f} | "
                        f"{contrast['deep_minus_control']:+.8f}, {contrast['ci95']} | {row['passed']} |")
        (args.output / "RESULTS.md").write_text("\n".join(text) + "\n")
        write_json(args.output / "complete.json", result)
        return result
    except BaseException as error:
        write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main():
    args = parser().parse_args()
    config = load_config(args.config)
    if args.mode == "manifest":
        manifest(args.config, args.physical_gpu, args.output, args.data_dir)
        return
    if args.output.exists():
        raise ValueError("Output exists; use a fresh directory, including for explicit resume")
    from .__main__ import configure_offline
    gpu = getattr(args, "physical_gpu", None)
    if gpu is not None and gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    configure_offline(gpu)
    if gpu is not None:
        from scripts.gpu_status import require_free
        require_free([gpu])
    if args.mode == "prepare":
        prepare(config, args.output)
    elif args.mode == "report":
        print(json.dumps(report_results(args, config), indent=2))
    else:
        print(json.dumps(run_arm(args, config), indent=2))


if __name__ == "__main__":
    main()
