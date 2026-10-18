"""Small native-Trainer extension for B-DOT's optimizer-step routing state."""
import json
from pathlib import Path
import time
import torch
from transformers import TrainerCallback

from packed_training import PackedTrainer


class RoutingUpdateCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if int(model.completed_steps) != state.global_step:
            raise ValueError('Routing state and Trainer optimizer step disagree on resume')
        model.progress = state.global_step / state.max_steps
        self.trainer._reset_timing()

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        model.progress = state.global_step / state.max_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        model.finish_update()
        model.progress = state.global_step / state.max_steps
        return control

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        reach, _, _ = model.options.phase(model.progress, True)
        if reach == 0 or model.options.S == 1:
            # Also robust to preexisting momentum/zero grad tensors after resume.
            for layer in model.routed_steps:
                if layer.router is not None:
                    for parameter in layer.router.parameters():
                        parameter.grad = None


class BDOTTrainer(PackedTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args') or (args[1] if len(args) > 1 else None)
        if training_args is None:
            raise ValueError('BDOTTrainer requires explicit TrainingArguments')
        if training_args.ddp_find_unused_parameters is False:
            raise ValueError('B-DOT needs unused-parameter handling for dynamic expert choices')
        if training_args.ddp_broadcast_buffers is True:
            raise ValueError('B-DOT collects rank-local counts; DDP buffer broadcasting must be disabled')
        training_args.ddp_find_unused_parameters = True
        training_args.ddp_broadcast_buffers = False
        if training_args.gradient_checkpointing:
            options = training_args.gradient_checkpointing_kwargs or {}
            if options.get('use_reentrant', False):
                raise ValueError('B-DOT supports nonreentrant checkpointing only')
            training_args.gradient_checkpointing_kwargs = dict(options, use_reentrant=False)
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled or self.is_deepspeed_enabled or self.args.n_gpu > 1:
            raise ValueError('B-DOT Phase A supports ordinary single-device or DDP only')
        if self.args.torch_compile:
            raise ValueError('Dynamic dispatch with torch_compile has not been validated')
        self._include_curriculum_eval = True
        self._interval_started = None
        self.add_callback(RoutingUpdateCallback(self))

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, lm_tokens = super().get_batch_samples(epoch_iterator, num_batches, device)
        # Phase A has S active decisions for every input token, including EOS and
        # masked document starts. LM labels and router decisions have different
        # denominators. Forced choices contribute zero z-loss but remain in the
        # denominator. Both sums cover the full global accumulation window.
        count = sum(batch['input_ids'].numel() for batch in batches) * self.model.options.S
        count = self.accelerator.reduce(torch.tensor(count, device=device, dtype=torch.long), reduction='sum')
        for batch in batches:
            batch['bdot_num_decisions'] = count
        return batches, lm_tokens

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs['objective_loss'] = logs.pop('loss')
            logs.update(self.model.last_update_metrics)
            interval = (self.model.training_totals - self.model.logged_training_totals).tolist()
            logs.update(lm_loss=interval[0]/max(interval[1], 1),
                        router_z_loss=interval[2]/max(interval[3], 1),
                        lm_tokens=int(interval[1]), input_tokens=int(interval[4]))
            logs.update(self._interval_performance())
            self.model.logged_training_totals.copy_(self.model.training_totals)
        if 'train_loss' in logs:
            logs['train_objective_loss'] = logs.pop('train_loss')  # HF's native aggregate.
            totals = self.model.training_totals.tolist()
            logs.update(train_lm_loss=totals[0]/max(totals[1], 1),
                        train_router_z_loss=totals[2]/max(totals[3], 1),
                        train_lm_tokens=int(totals[1]), train_input_tokens=int(totals[4]))
        if self.args.should_save:
            path = Path(self.args.output_dir) / 'metrics.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as handle:
                handle.write(json.dumps(dict(step=self.state.global_step, **logs), default=str) + '\n')
        return super().log(logs, start_time)

    def _reset_timing(self):
        self._interval_started = time.perf_counter()
        self._timed_input_tokens = float(self.model.training_totals[4])
        if self.args.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.args.device)

    def _interval_performance(self):
        if self._interval_started is None:
            return {}
        elapsed = time.perf_counter() - self._interval_started
        allocated = reserved = 0
        if self.args.device.type == 'cuda':
            allocated = torch.cuda.max_memory_allocated(self.args.device)
            reserved = torch.cuda.max_memory_reserved(self.args.device)
        values = torch.tensor([elapsed, allocated, reserved], dtype=torch.float64, device=self.args.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
        elapsed, allocated, reserved = values.tolist()
        tokens = float(self.model.training_totals[4]) - self._timed_input_tokens
        result = dict(input_tokens_per_second_interval_wall=tokens/max(elapsed, 1e-12),
                      interval_wall_seconds=elapsed,
                      peak_allocated_bytes=int(allocated) if self.args.device.type == 'cuda' else None,
                      peak_reserved_bytes=int(reserved) if self.args.device.type == 'cuda' else None)
        self._reset_timing()
        return result

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval', *, include_curriculum=True):
        previous = self._include_curriculum_eval
        self._include_curriculum_eval = include_curriculum
        try:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        finally:
            self._include_curriculum_eval = previous

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix='eval'):
        from packed_evaluation import score_packed_loader
        model = self.accelerator.unwrap_model(self.model_wrapped)
        original = (model.evaluation_policy, model.eval_counter, model.last_routes)
        try:
            model.evaluation_policy = 'target'
            result = super().evaluation_loop(dataloader, description, prediction_loss_only,
                                             ignore_keys, metric_key_prefix)
            if self._include_curriculum_eval:
                reach, soft, _ = model.options.phase(model.progress, False, evaluation_policy='curriculum')
                if (reach, soft) == (model.options.r, False):
                    current = {k[len(metric_key_prefix)+1:]: v for k, v in result.metrics.items()}
                else:
                    model.evaluation_policy = 'curriculum'
                    current = score_packed_loader(model, dataloader, self.accelerator)
                result.metrics.update({f'{metric_key_prefix}_curriculum_{k}': v for k, v in current.items()})
                result.metrics.update({f'{metric_key_prefix}_curriculum_reach': reach,
                                       f'{metric_key_prefix}_curriculum_soft': soft,
                                       f'{metric_key_prefix}_spike_loss': current['loss']})
            return result
        finally:
            model.evaluation_policy, model.eval_counter, model.last_routes = original

    def get_decay_parameter_names(self, model):
        names = super().get_decay_parameter_names(model)
        return [name for name in names if not any(part in name for part in ('norm', 'bias', 'film_'))]
