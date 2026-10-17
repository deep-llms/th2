"""Offline, token-weighted held-out PPL for this template's packed checkpoints.

Run directly or via Accelerate/torchrun. Each rank scores disjoint whole rows;
there is no distributed sampler padding, DDP forward collective, or logits gather.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import dataclass
import tempfile

os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="offline")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments

from train import ModelArguments, DataArguments, build_model
from packed_training import PackedCollator
from packed_evaluation import evaluate_packed, summed_nll
from training_data import POLICY, prepare_training_data, tokenizer_identity, file_record, digest, require_held_out


@dataclass
class EvalArguments:
    checkpoint: str
    eval_data_dir: str
    output_file: str
    attention_backend: str | None = None  # Default to the training receipt.
    tokenizer_name: str | None = None
    precision: str = "fp32"
    use_cpu: bool = False
    batch_size: int = 1
    max_sequences: int | None = None
    data_cache_dir: str | None = None
    overwrite_cache: bool = False


def data_arguments(recipe, path, cache_dir=None, overwrite=False):
    """Use the checkpoint's actual block size and packing policy, not defaults."""
    supported = {
        "policy": {POLICY},
        "attention": {"causal_within_document", "causal_across_documents"},
        "positions": {"reset_per_document_fragment", "continuous_within_block"},
        "append_boundary": {"one_after_every_document_including_empty"},
        "remainder": {"drop_per_map_batch_within_worker_shard"},
    }
    for key, values in supported.items():
        if recipe.get(key) not in values:
            raise ValueError(f"Unsupported checkpoint data recipe: {key}={recipe.get(key)!r}")
    isolated = recipe["attention"] == "causal_within_document"
    if recipe["labels"] != ("ignore_first_and_document_starts" if isolated else "ignore_block_start"):
        raise ValueError("Checkpoint label policy is inconsistent with attention")
    if recipe["add_special_tokens"] is not False:
        raise ValueError("Unsupported special-token insertion policy")
    return DataArguments(data_dir=path, block_size=recipe["block_size"],
        document_end_token=recipe["document_end"]["token"],
        map_batch_size=recipe["map_batch_size"], preprocessing_num_workers=recipe["requested_workers"],
        isolate_documents=isolated,
        reset_document_positions=recipe["positions"] == "reset_per_document_fragment",
        data_cache_dir=cache_dir, overwrite_cache=overwrite)



def write_result(path, result):
    """Publish complete JSON without overwriting an earlier evaluation."""
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.link(temporary, path)  # Atomic, and refuses an existing destination.
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = HfArgumentParser(EvalArguments)
    args, = (parser.parse_json_file(os.path.abspath(sys.argv[1]))
             if len(sys.argv) == 2 and sys.argv[1].endswith(".json")
             else parser.parse_args_into_dataclasses())
    if args.precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16 or fp16")
    if args.batch_size < 1 or (args.max_sequences is not None and args.max_sequences < 1):
        raise ValueError("batch_size and optional max_sequences must be positive")
    checkpoint, output = Path(args.checkpoint).resolve(), Path(args.output_file).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a fresh output_file")
    receipt_path = checkpoint / "data_recipe.json"
    if not receipt_path.is_file():
        raise ValueError("Packed evaluation requires this template's checkpoint data_recipe.json")
    train_recipe = json.loads(receipt_path.read_text())
    data = data_arguments(train_recipe, args.eval_data_dir, args.data_cache_dir, args.overwrite_cache)
    backend = args.attention_backend or train_recipe["attention_backend"]
    # TrainingArguments sets up/reset HF's shared device state. Construct it
    # before Accelerator, just as Trainer does, so the latter stays initialized.
    prep_args = TrainingArguments(output_dir=str(output.parent), use_cpu=args.use_cpu,
        report_to="none", seed=0, data_seed=0, bf16=args.precision == "bf16", fp16=args.precision == "fp16")
    accelerator = Accelerator(cpu=args.use_cpu, mixed_precision="no" if args.precision == "fp32" else args.precision)
    if backend == "fa4" and (accelerator.device.type != "cuda" or args.precision == "fp32"):
        raise ValueError("FA4 evaluation requires CUDA and bf16/fp16 precision")
    if backend == "fa4":
        from packed_fa4 import load_kernel
        load_kernel()
    tokenizer_dir = args.tokenizer_name or str(checkpoint)
    if not Path(tokenizer_dir).is_dir():
        raise ValueError("Tokenizer must be an existing local directory")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_identity(tokenizer) != train_recipe["tokenizer"]:
        raise ValueError("Evaluation tokenizer differs from the training receipt")
    # Reuse the two HF maps unchanged. A fixed evaluation shuffle is independent
    # of training seed; all checkpoints get the same optional prefix selection.
    dataset, recipe = prepare_training_data(data, prep_args, tokenizer, tokenizer_dir,
        data.block_size, attention_backend=backend)
    require_held_out(train_recipe, recipe)
    if recipe["document_end"]["id"] != train_recipe["document_end"]["id"]:
        raise ValueError("Evaluation document-end token ID differs from training")
    full_size = len(dataset)
    if args.max_sequences is not None:
        dataset = dataset.select(range(min(args.max_sequences, full_size)))
    # Each process owns an independent evaluation replica; autocast is explicit
    # to avoid converting the full vocabulary-sized output to FP32 in Accelerate.
    model = build_model(ModelArguments(model_name_or_path=str(checkpoint), attention_backend=backend),
                        data.isolate_documents).to(accelerator.device)
    metrics = evaluate_packed(model, dataset,
        PackedCollator(data.isolate_documents, data.reset_document_positions, backend), accelerator, args.batch_size)
    if len(dataset) == full_size and metrics["num_tokens"] != recipe["scored_target_tokens"]:
        raise RuntimeError("Evaluation token count differs from the packed-data receipt")
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    result = dict(status="ok", **metrics, checkpoint=str(checkpoint), global_step=state.get("global_step"),
        attention_backend=backend, training_attention_backend=train_recipe["attention_backend"],
        precision=args.precision, world_size=accelerator.num_processes, batch_size_per_rank=args.batch_size,
        selection=dict(full_sequences=full_size, max_sequences=args.max_sequences, shuffle_seed=0),
        training_recipe_sha256=digest(train_recipe), data_recipe=recipe,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scoring_sha256=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("packed_evaluation.py").read_bytes()).hexdigest(),
        weights=[file_record(p) for p in sorted({*checkpoint.glob("*.safetensors"),
                                               *checkpoint.glob("pytorch_model*.bin")})],
        config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        torch_version=torch.__version__)
    if accelerator.is_main_process:
        write_result(output, result)
        print(json.dumps(metrics, indent=2, allow_nan=False))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
