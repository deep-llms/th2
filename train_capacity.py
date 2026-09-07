"""Train one registered arm, from scratch, on frozen local GPT-2 packs.

Launch with python (CPU/single GPU) or torchrun/accelerate (DDP). No downloads,
GPU-process control, pretrained initialization, or implicit experiment queue.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import SequentialSampler, Subset
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator, set_seed

from capacity_allocation.data import PackedTokens, sha256, write_json
from capacity_allocation.modeling import ARMS, activation_report, build_model, experiment_config, parameter_report


class OrderedTrainer(Trainer):
    def _get_num_items_in_batch(self, batch_samples, device):
        # HF 5.9 counts all unmasked labels, but causal LM position zero is
        # never a target after shifting. Count only actually scored labels;
        # retain HF's cross-rank and accumulation normalization machinery.
        shifted = [{"labels": batch["labels"][:, 1:]} for batch in batch_samples]
        return super()._get_num_items_in_batch(shifted, device)

    def _get_train_sampler(self, train_dataset=None):
        # The subset was shuffled ONCE with data_seed, independently of model RNG.
        # Accelerate shards this global sequence across processes.
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Gather per-example sums/counts, NOT a repeated per-batch mean. On an
        # uneven final DDP batch Accelerate may duplicate examples; it can trim
        # those exactly only if the gathered metrics remain per-example.
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        with torch.no_grad(), self.compute_loss_context_manager():
            logits = model(**inputs, use_cache=False).logits
            targets = labels[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().transpose(1, 2), targets,
                ignore_index=-100, reduction="none")
            scores = torch.stack((losses.sum(dim=1, dtype=torch.float64),
                                  (targets != -100).sum(dim=1).double()), dim=1)
        return None, scores, torch.zeros_like(scores[:, :1])


def packed_nll_metrics(prediction):
    scores = np.asarray(prediction.predictions, dtype=np.float64)
    count = scores[:, 1].sum()
    if count <= 0:
        raise ValueError("Validation has no scored targets")
    return dict(loss=float(scores[:, 0].sum()/count), scored_targets=int(count))


class StopAtStep(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step is not None and state.global_step >= self.step:
            control.should_training_stop = True
            control.should_save = True
        return control


def schedule_budget(tokens, batch_per_device, accumulation, world_size, sequence_length):
    if any(type(x) is not int or x <= 0 for x in
           (tokens, batch_per_device, accumulation, world_size, sequence_length)):
        raise ValueError("Token budget and batch dimensions must be positive integers")
    per_step = batch_per_device * accumulation * world_size * sequence_length
    steps = (tokens + per_step - 1) // per_step
    return dict(requested_tokens=tokens, tokens_per_step=per_step, steps=steps,
                actual_tokens=steps*per_step, rounding_extra_tokens=steps*per_step-tokens,
                scored_targets_per_step=per_step//sequence_length*(sequence_length-1))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-tokens", type=int, required=True, help="Full LR schedule horizon, not a stop override")
    p.add_argument("--stop-at-step", type=int)
    p.add_argument("--resume", help="Explicit checkpoint path, within this output directory")
    p.add_argument("--depth", type=int, choices=(6, 12), default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--batch-per-device", type=int, required=True)
    p.add_argument("--gradient-accumulation", type=int, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--warmup-ratio", type=float, required=True)
    p.add_argument("--weight-decay", type=float, required=True)
    p.add_argument("--beta1", type=float, default=.9)
    p.add_argument("--beta2", type=float, required=True)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    p.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tiny", action="store_true", help="Synthetic correctness tests only; not a research run")
    return p


def main():
    cli = parser()
    args = cli.parse_args()
    if not (args.learning_rate > 0 and 0 <= args.warmup_ratio < 1 and args.weight_decay >= 0
            and 0 <= args.beta1 < 1 and 0 <= args.beta2 < 1 and args.adam_epsilon > 0
            and args.max_grad_norm > 0):
        cli.error("Invalid optimizer/scheduler setting")
    if any(x <= 0 for x in (args.eval_batch_size, args.eval_steps, args.save_steps, args.logging_steps)):
        cli.error("Batch and logging/evaluation/save intervals must be positive")
    data = PackedTokens(args.data, "train")
    validation = PackedTokens(args.data, "validation")  # Test is never used for architecture selection.
    seq = data.length
    if not args.tiny and (seq != 2048 or data.manifest["vocab_size"] != 50257
                          or data.manifest["eos_token_id"] != 50256):
        cli.error("Production requires the shared 2048-token GPT-2 packs")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    budget = schedule_budget(args.train_tokens, args.batch_per_device, args.gradient_accumulation, world, seq)
    if args.stop_at_step is not None and not 0 < args.stop_at_step <= budget["steps"]:
        cli.error("stop-at-step must be inside the unchanged schedule horizon")
    nblocks = budget["actual_tokens"] // seq
    if len(data) < nblocks:
        cli.error(f"Need {nblocks} distinct train packs, have {len(data)}; implicit repetition is forbidden")
    config = experiment_config(args.arm, depth=args.depth, tiny=args.tiny, attention=args.attention)
    if data.manifest["vocab_size"] != config.vocab_size:
        cli.error("Data vocabulary does not match model")
    training_args = TrainingArguments(
        output_dir=args.output, use_cpu=args.cpu, seed=args.seed, data_seed=args.data_seed,
        per_device_train_batch_size=args.batch_per_device,
        gradient_accumulation_steps=args.gradient_accumulation,
        per_device_eval_batch_size=args.eval_batch_size, max_steps=budget["steps"],
        learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        warmup_steps=math.ceil(args.warmup_ratio * budget["steps"]),
        weight_decay=args.weight_decay, adam_beta1=args.beta1, adam_beta2=args.beta2,
        adam_epsilon=args.adam_epsilon, max_grad_norm=args.max_grad_norm, optim="adamw_torch",
        bf16=args.precision == "bf16", fp16=False, tf32=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,  # Tested all parameters receive gradients, for every arm.
        eval_strategy="steps", eval_steps=args.eval_steps, prediction_loss_only=False,
        save_strategy="steps", save_steps=args.save_steps, logging_steps=args.logging_steps,
        report_to="none", dataloader_num_workers=0, dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    if training_args.world_size != world or training_args.n_gpu > 1:
        cli.error("Launch one process per GPU with torchrun/accelerate, not single-process DataParallel")
    root = Path(args.output).resolve()
    fingerprint_args = {k: v for k, v in vars(args).items() if k not in ("resume", "stop_at_step")}
    specification = dict(arguments=fingerprint_args, budget=budget,
                         data_manifest_sha256=sha256(Path(args.data) / "manifest.json"),
                         world_size=world,
                         software={name: importlib.metadata.version(name)
                                   for name in ("torch", "transformers", "accelerate", "numpy")},
                         code_sha256={str(p.relative_to(Path(__file__).parent)): sha256(p)
                                      for p in [Path(__file__)] + sorted((Path(__file__).parent / "capacity_allocation").glob("*.py"))})
    # Rank zero checks large hashes once. Other ranks wait; no redundant 8-way disk scan.
    with training_args.main_process_first(desc="verify data and register run"):
        if training_args.process_index == 0:
            PackedTokens(args.data, "train", verify_hash=True)
            PackedTokens(args.data, "validation", verify_hash=True)
            if args.resume:
                checkpoint = Path(args.resume).resolve()
                if checkpoint.parent != root or not (checkpoint / "trainer_state.json").is_file():
                    raise ValueError("Resume requires a Trainer checkpoint inside this run")
                if json.loads((root / "run_specification.json").read_text()) != specification:
                    raise ValueError("Resume specification differs: use identical data/config/software/code")
                prior_step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
                if prior_step >= (args.stop_at_step or budget["steps"]):
                    raise ValueError("Requested stop must be after resumed checkpoint")
                if (root / "result.json").exists():
                    raise ValueError("Completed output already has result.json; resume only interrupted runs")
            else:
                if root.exists() and any(root.iterdir()):
                    raise ValueError("Refusing to overwrite a nonempty run directory")
                root.mkdir(parents=True, exist_ok=True)
                write_json(root / "run_specification.json", specification)
    # Same global pack permutation independent of architecture/parameter RNG consumption.
    generator = torch.Generator().manual_seed(args.data_seed)
    indices = torch.randperm(len(data), generator=generator)[:nblocks].tolist()
    train = Subset(data, indices)
    set_seed(args.seed)
    model = build_model(config)
    counts = parameter_report(model)
    trainer = OrderedTrainer(model=model, args=training_args, train_dataset=train,
                             eval_dataset=validation, data_collator=default_data_collator,
                             compute_metrics=packed_nll_metrics,
                             callbacks=[StopAtStep(args.stop_at_step)])
    if training_args.process_index == 0 and not args.resume:
        fixed_batch = data_collate_diagnostic(validation, training_args.device)
        write_json(root / "initial_scales.json", activation_report(model, fixed_batch))
        write_json(root / "parameters.json", counts)
    trainer.accelerator.wait_for_everyone()
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.reset_peak_memory_stats(training_args.device)
        torch.cuda.synchronize(training_args.device)
    started = time.perf_counter()
    trained = trainer.train(resume_from_checkpoint=args.resume)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.synchronize(training_args.device)
    elapsed = time.perf_counter() - started
    expected_step = args.stop_at_step or budget["steps"]
    if trainer.state.global_step != expected_step:
        raise RuntimeError(f"Training ended at {trainer.state.global_step}, expected {expected_step}")
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", float("nan"))
    if not math.isfinite(loss):
        raise RuntimeError("Non-finite validation loss; refusing a success result")
    if metrics["eval_scored_targets"] != len(validation) * (seq - 1):
        raise RuntimeError("Validation target count mismatch; a pack was omitted or duplicated")
    trainer.save_model(str(root / "final"))
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if training_args.process_index == 0:
        write_json(root / "result.json", dict(
            success=True, status="horizon_complete" if expected_step == budget["steps"] else "stopped_at_step",
            arm=args.arm, seed=args.seed, global_step=trainer.state.global_step,
            processed_tokens=trainer.state.global_step*budget["tokens_per_step"],
            scored_targets=trainer.state.global_step*budget["scored_targets_per_step"],
            validation_nll=loss, validation_ppl=math.exp(loss) if loss < 700 else None,
            validation_scored_targets=metrics["eval_scored_targets"],
            parameters=counts, budget=budget, training_metrics=trained.metrics,
            train_call_wall_seconds=elapsed, timing_includes_periodic_eval_and_checkpoints=True,
            measured_processed_tokens_this_invocation=(
                (trainer.state.global_step - (prior_step if args.resume else 0)) * budget["tokens_per_step"]),
            peak_allocated_bytes_rank0=(torch.cuda.max_memory_allocated(training_args.device)
                                        if torch.cuda.is_available() and not args.cpu else None),
            final_checkpoint="final", tiny_test=args.tiny))


def data_collate_diagnostic(validation, device):
    # Fixed validation context, cropped only for this initialization diagnostic.
    return validation[0]["input_ids"][:128].unsqueeze(0).to(device)


if __name__ == "__main__":
    main()
