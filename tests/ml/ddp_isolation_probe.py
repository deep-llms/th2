"""Run with torchrun --standalone --nproc_per_node=2; CPU-only integration probe."""
import copy
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import TrainingArguments
from packed_training import PackedCollator, PackedTrainer
from test_document_isolation import model, row, objective

torch.set_num_threads(1)
output = Path(sys.argv[1])
rows = [row([3, 4, 31, 8, 9, 10, 31], [0, 0, 0, 1, 1, 1, 1]),
        row([31] * 7, list(range(7))),
        row([3, 4, 5, 6, 7, 8, 31], [0] * 7),
        row([3, 4, 5, 31, 8, 9, 31], [0, 0, 0, 0, 1, 1, 1])]
collator = PackedCollator()
m = model()
reference = copy.deepcopy(m)
ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
full_batch = collator(rows)
_, total = objective(reference, full_batch)
count = full_batch["labels"].ne(-100).sum()
(total / count).backward()
ref_optimizer.step()
backend = "sdpa"
if "--fa4-reference" in sys.argv[2:]:
    # CPU adapter/DDP integration test only, never a CUDA-kernel measurement.
    from unittest.mock import patch
    from test_fa4 import reference_kernel
    import packed_fa4
    patch.object(packed_fa4, "load_kernel", return_value=reference_kernel).start()
    patch.object(packed_fa4, "validate_kernel_inputs").start()
    packed_fa4.register_backend()
    m.set_attn_implementation(packed_fa4.BACKEND)
    collator = PackedCollator(attention_backend="fa4")
    backend = "fa4_adapter_with_cpu_reference"
optimizer = torch.optim.SGD(m.parameters(), lr=.03)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
args = TrainingArguments(output_dir=str(output), use_cpu=True, report_to="none",
    remove_unused_columns=False, per_device_train_batch_size=1, gradient_accumulation_steps=2,
    max_steps=1, save_strategy="no", max_grad_norm=0., seed=42, data_seed=42,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False, ddp_timeout=60, disable_tqdm=True)
callbacks = []
if "--cutoff" in sys.argv[2:]:
    from train import BaselineTrainingArguments, StopAfterCallback
    args = BaselineTrainingArguments(output_dir=str(output), use_cpu=True, report_to="none",
        remove_unused_columns=False, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        max_steps=4, stop_after=1, save_strategy="no", max_grad_norm=0., seed=42, data_seed=42,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False, ddp_timeout=60, disable_tqdm=True)
    callbacks = [StopAfterCallback()]
trainer = PackedTrainer(model=m, args=args, train_dataset=rows, data_collator=collator,
                        optimizers=(optimizer, scheduler), callbacks=callbacks)
if trainer.accelerator.num_processes != 2:
    raise RuntimeError("This probe requires exactly two CPU ranks")
trainer.train()
max_error = 0.
for (name, actual), (_, expected) in zip(m.named_parameters(), reference.named_parameters()):
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5, msg=name)
    max_error = max(max_error, (actual - expected).abs().max().item())
trainer.accelerator.wait_for_everyone()  # All ranks must pass before publishing.
if trainer.is_world_process_zero():
    if callbacks:
        assert trainer.state.global_step == 1 and trainer.state.max_steps == 4
        for name in ("optimizer.pt", "scheduler.pt", "rng_state_0.pth", "rng_state_1.pth"):
            assert (output / "checkpoint-1" / name).is_file(), name
    (output / "verification.json").write_text(json.dumps(dict(
        status="passed", world_size=2, gradient_accumulation=2,
        scored_tokens=count.item(), max_parameter_error=max_error,
        reference="one full-batch SGD update with token-weighted CE", device="cpu", backend=backend,
        global_step=trainer.state.global_step, schedule_steps=trainer.state.max_steps), indent=2))
trainer.accelerator.wait_for_everyone()
torch.distributed.destroy_process_group()
