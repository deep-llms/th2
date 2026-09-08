"""Opt-in future Trainer instrumentation. Not imported by the active trainer.

HF 5.9 on_pre_optimizer_step runs AFTER clipping. Observe _clip_grad_norm
instead, and use on_optimizer_step only for actual parameter deltas.
Supports unsharded FP32/BF16 ordinary Trainer/DDP, not scaled FP16/FSDP/ZeRO.
"""
from pathlib import Path
import math
import torch
from transformers import TrainerCallback
from capacity_allocation.data import write_json
from eval.diagnostic_gradients import selected_parameters, tensor_statistics, row_statistics
from eval.diagnostic_metrics import interfaces


class TrainingRecorder(TrainerCallback):
    def __init__(self, trainer, output_dir, *, every=100, update_every=1000, mapping=None):
        if every < 1 or update_every < 0:
            raise ValueError('Positive gradient interval and nonnegative update interval required')
        args = trainer.args
        if (args.fp16 or args.deepspeed or args.fsdp or args.max_grad_norm <= 0 or
                getattr(trainer.accelerator, 'scaler', None) is not None):
            raise ValueError('Requires unscaled FP32/BF16 ordinary DDP and positive clipping threshold')
        if getattr(trainer.accelerator, 'distributed_type', None) is not None:
            from accelerate.utils import DistributedType
            if trainer.accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU,
                                                            DistributedType.MULTI_CPU):
                raise ValueError('Unsupported/sharded gradient statistics')
        self.trainer, self.root = trainer, Path(output_dir)
        self.every, self.update_every, self.mapping = every, update_every, mapping
        self.before, self.record = {}, None
        if trainer.is_world_process_zero():
            self.root.mkdir(parents=True, exist_ok=False)

    def before_clip(self):
        step = self.trainer.state.global_step+1
        do_updates = bool(self.update_every and step % self.update_every == 0)
        if not self.trainer.is_world_process_zero() or (step % self.every and not do_updates):
            return
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        parameters = selected_parameters(model)
        self.record = dict(global_step=step, gradient_stage='post_DDP_all_reduce_pre_clipping_pre_optimizer',
            gradient_scaling=False, precision='bf16' if self.trainer.args.bf16 else 'fp32',
            world_size=self.trainer.args.world_size,
            parameters={name: tensor_statistics(p, p.grad) for name, p in parameters.items()})
        if self.mapping is not None:
            self.record['rows'] = {side: row_statistics(p.grad, self.mapping)
                                   for side, (p, _) in interfaces(model).items() if p.grad is not None}
        if do_updates:
            self.before = {name: p.detach().float().cpu().clone() for name, p in parameters.items()}

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.record is None:
            return control
        self.record['optimizer_step_skipped'] = bool(self.trainer.accelerator.optimizer_step_was_skipped)
        if self.before:
            model = self.trainer.accelerator.unwrap_model(self.trainer.model)
            params, updates = dict(model.named_parameters()), {}
            for name, before in self.before.items():
                after = params[name].detach().float().cpu()
                if not torch.isfinite(after).all():
                    raise ValueError('Nonfinite post-step weights')
                delta_rms = torch.linalg.vector_norm(after-before).item()/math.sqrt(before.numel())
                weight_rms = torch.linalg.vector_norm(before).item()/math.sqrt(before.numel())
                updates[name] = dict(update_rms=delta_rms, update_to_weight=delta_rms/(weight_rms+1e-12))
            self.record['updates'] = updates
        write_json(self.root/f'step-{self.record["global_step"]}.json', self.record)
        self.before, self.record = {}, None
        return control


def attach_training_diagnostics(trainer, output_dir, **kwargs):
    """Call once after constructing a future CausalTrainer, before train().

    Only observes tensors: no backward, gradient modification or optimizer step.
    Default training is completely unchanged unless explicitly attached.
    """
    if hasattr(trainer, '_capacity_diagnostic_recorder'):
        raise ValueError('Training diagnostics already attached')
    recorder = TrainingRecorder(trainer, output_dir, **kwargs)
    original_clip = trainer._clip_grad_norm
    def observed_clip(model):
        recorder.before_clip()
        return original_clip(model)
    trainer._clip_grad_norm = observed_clip
    trainer.add_callback(recorder)
    trainer._capacity_diagnostic_recorder = recorder
    return recorder
