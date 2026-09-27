"""Offline data preparation, CPU smoke runs, and eight-GPU pilot training."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from .config import ARMS, Recipe, load_config, plan


def offline():
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="offline")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def jobs(config_path, data_dir=None, stop_after=None):
    config_path = str(Path(config_path).resolve())
    load_config(config_path)
    recipe = Recipe().validate()
    end = recipe.end_update(stop_after)
    status = "complete" if end == recipe.updates else "stopped"
    cutoff_args = [] if stop_after is None else ["--stop-after", str(stop_after)]
    items = []
    if data_dir is None:
        data_dir = "{run_dir}/prepared"
        items.append({"name": "prepare", "argv": ["{python}", "-m", "deep_kv", "prepare",
                      "--config", config_path, "--output", data_dir],
                      "required_outputs": [{"path": "prepared/complete.json",
                                            "json_equals": {"format": "deep-kv-data-v2"}}]})
    else:
        data_dir = str(Path(data_dir).resolve())
    for arm in ARMS:
        items.append({"name": f"arm-{arm}", "gpus": list(range(8)),
                      "argv": ["{python}", "-m", "accelerate.commands.launch", "--multi_gpu",
                               "--num_machines", "1", "--num_processes", "8", "--mixed_precision", "bf16",
                               "--dynamo_backend", "no", "--module", "deep_kv", "train", "--config", config_path,
                               "--data-dir", data_dir, "--output", "{run_dir}/" + arm, "--arm", arm] + cutoff_args,
                      "required_outputs": [{"path": f"{arm}/{status}.json", "json_equals": {
                          "status": status, "arm": arm, "update": end,
                          "input_tokens": end * recipe.tokens_per_update}}]})
    items.append({"name": "compare", "argv": ["{python}", "-m", "deep_kv", "report", "--run-dir", "{run_dir}"] + cutoff_args,
                  "required_outputs": [{"path": "comparison.json", "json_equals": {
                      "status": "complete", "compared_update": end,
                      "training_complete": end == recipe.updates}}]})
    return {"jobs": items}


def setup_distributed(cpu=False):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not cpu:
        if world != 8 or rank != local or not 0 <= local < 8 or torch.cuda.device_count() != 8:
            raise ValueError("Pilot requires a single-node launch with all eight visible GPUs")
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("gloo" if cpu else "nccl", timeout=timedelta(seconds=21600))
    return torch.device("cpu" if cpu else f"cuda:{local}")


def build_identity(config, recipe, model, data_manifest, *, mixed_precision):
    import accelerate
    import transformers
    import torch
    from .data import sha256
    from .training import parameter_hash, topology
    root = Path(__file__).resolve().parent.parent
    sources = sorted((root / "deep_kv").glob("*.py")) + [root / p for p in ("pcc/model.py", "pcc/packing.py")]
    identity = {"format": "deep-kv-run-hf-v1", "arm": model.arm, "recipe": asdict(recipe),
            "config": config, "model_config": model.backbone.config.to_dict(),
            "data": data_manifest, "world_size": topology()[0], "mixed_precision": mixed_precision,
            "initial_backbone_sha256": parameter_hash(model.backbone),
            "initial_aux_sha256": parameter_hash(model.aux) if model.aux is not None else None,
            "code": {str(p.relative_to(root)): sha256(p) for p in sources},
            "software": {"torch": str(torch.__version__), "transformers": transformers.__version__,
                         "accelerate": accelerate.__version__}}
    # Config dictionaries can contain integer keys (e.g. id2label). Canonicalize
    # once so JSON receipts and torch checkpoints compare identically on resume.
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


def run_pilot(args):
    import accelerate
    import torch
    import torch.distributed as dist
    import transformers
    from transformers import Qwen3Config
    from .data import TokenStream, verify_sources
    from .model import DeepKV
    from .training import train, topology
    if transformers.__version__ != "5.9.0" or accelerate.__version__ != "1.13.0":
        raise ValueError("Use the B200-matched environment: transformers==5.9.0, accelerate==1.13.0")
    config = load_config(args.config)
    recipe = Recipe().validate()
    device = setup_distributed()
    try:
        torch.manual_seed(recipe.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_config = Qwen3Config.from_json_file(config["model_config"])
        for key, expected in dict(num_hidden_layers=28, hidden_size=1024, num_attention_heads=16,
                                  num_key_value_heads=8, head_dim=128, vocab_size=151936,
                                  rms_norm_eps=1e-6).items():
            if getattr(model_config, key) != expected:
                raise ValueError(f"Pilot backbone configuration mismatch: {key}")
        model_config._attn_implementation = "sdpa"
        model_config.use_cache = False
        train_data = TokenStream(args.data_dir, "train", recipe, verify=topology()[1] == 0)
        eval_data = TokenStream(args.data_dir, "eval", recipe, verify=topology()[1] == 0)
        verify_sources(train_data.manifest, config)
        if train_data.manifest["train"]["preprocessing"]["document_end_token_id"] != 151643:
            raise ValueError("Pilot requires Qwen <|endoftext|> document boundaries")
        model = DeepKV.from_scratch(model_config, args.arm, seed=recipe.seed,
                                    consumer=recipe.consumer, deep_target=recipe.deep_target,
                                    lm_chunk=recipe.lm_chunk)
        identity = build_identity(config, recipe, model, train_data.manifest, mixed_precision=True)
        if dist.is_initialized():
            identities = [None] * dist.get_world_size()
            dist.all_gather_object(identities, identity)
            if any(item != identity for item in identities):
                raise ValueError("DDP initialization or data identity mismatch")
        model.to(device)
        train(model, train_data, eval_data, recipe, args.output, identity, microbatch=config["microbatch"],
              resume=args.resume, stop_after=args.stop_after)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def smoke(args):
    # Intentionally small, synthetic, CPU-only; cannot produce a pilot result.
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import Qwen3Config
    from .data import TokenStream, sha256
    from .model import DeepKV
    from .training import train, topology, write_json
    from pcc.packing import preprocessing_policy
    device = setup_distributed(cpu=True)
    torch.set_num_threads(1)
    recipe = Recipe(updates=3, context=8, tokens_per_update=4096, warmup=1, eval_every=2,
                    monitor_rows=2, eval_rows=5, checkpoint_every=2, consumer=2, deep_target=4, lm_chunk=4,
                    logging_every=1, dataloader_workers=0)
    root = Path(args.output)
    try:
        if topology()[1] == 0:
            root.mkdir(parents=True, exist_ok=False)
            data_dir = root / "prepared"
            data_dir.mkdir()
            meta = {"format": "deep-kv-data-v2", "recipe": asdict(recipe)}
            for split, rows in (("train", recipe.train_rows), ("eval", recipe.eval_rows)):
                np.random.default_rng(42 + (split == "eval")).integers(0, 31, (rows, recipe.context), dtype=np.uint32).tofile(data_dir / f"{split}.bin")
                np.save(data_dir / f"{split}.order.npy", np.arange(rows))
                meta[split] = {"rows": rows, "context": recipe.context, "dtype": "uint32",
                               "tokens_sha256": sha256(data_dir / f"{split}.bin"),
                               "order_sha256": sha256(data_dir / f"{split}.order.npy"),
                               "preprocessing": preprocessing_policy(recipe.context, 30)}
            write_json(data_dir / "complete.json", meta)
        if dist.is_initialized():
            dist.barrier()
        train_data, eval_data = (TokenStream(root / "prepared", split, recipe) for split in ("train", "eval"))
        for arm in ARMS:
            config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48,
                                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
                                tie_word_embeddings=True)
            config._attn_implementation = "sdpa"
            model = DeepKV.from_scratch(config, arm, consumer=2, deep_target=4, lm_chunk=4).to(device)
            identity = build_identity({"synthetic_smoke": True}, recipe, model, train_data.manifest, mixed_precision=args.bf16)
            train(model, train_data, eval_data, recipe, root / arm, identity, microbatch=16,
                  mixed_precision=args.bf16, stop_after=args.stop_after)
        if topology()[1] == 0:
            from .report import report
            report(root, recipe=recipe, stop_after=args.stop_after)
            write_json(root / "smoke_complete.json", {"status": "ok", "arms": list(ARMS), "cpu_only": True,
                                                      "world_size": topology()[0], "pilot_result": False})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    offline()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare", "make-jobs", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name in ("plan", "make-jobs", "train"):
            p.add_argument("--stop-after", type=int, help="Stop all requested arms at this update; retain full LR schedule")
        if name != "plan":
            p.add_argument("--output", type=Path, required=True)
        if name in ("make-jobs", "train"):
            p.add_argument("--data-dir", type=Path, required=name == "train")
        if name == "train":
            p.add_argument("--arm", choices=ARMS, required=True)
            p.add_argument("--resume", action="store_true")
    p = sub.add_parser("smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--bf16", action="store_true", help="Exercise Trainer's bfloat16 path on CPU")
    p = sub.add_parser("report")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--stop-after", type=int, help="Require all arms to have stopped at this exact update")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(load_config(args.config), args.stop_after), indent=2))
    elif args.command == "make-jobs":
        with args.output.open("x") as handle:
            json.dump(jobs(args.config, args.data_dir, args.stop_after), handle, indent=2)
    elif args.command == "prepare":
        from .data import prepare
        prepare(load_config(args.config), args.output, Recipe().validate())
    elif args.command == "train":
        run_pilot(args)
    elif args.command == "report":
        from .report import report
        print(json.dumps(report(args.run_dir, stop_after=args.stop_after), indent=2))
    else:
        smoke(args)


if __name__ == "__main__":
    main()
