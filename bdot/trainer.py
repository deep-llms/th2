"""Small native-Trainer extension for B-DOT's optimizer-step routing state."""
import json
from pathlib import Path
import torch
from transformers import TrainerCallback

from packed_training import PackedTrainer


class RoutingUpdateCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if int(model.completed_steps) != state.global_step:
            raise ValueError('Routing state and Trainer optimizer step disagree on resume')

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        model.progress = state.global_step / state.max_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        model.finish_update()
        model.progress = state.global_step / state.max_steps
        return control


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
        self.add_callback(RoutingUpdateCallback())

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, lm_tokens = super().get_batch_samples(epoch_iterator, num_batches, device)
        # Phase A has S active decisions for every input token, including EOS and
        # masked document starts. LM labels and router decisions have different
        # denominators. Both sums are over the full global accumulation window.
        count = sum(batch['input_ids'].numel() for batch in batches) * self.model.options.S
        count = self.accelerator.reduce(torch.tensor(count, device=device, dtype=torch.long), reduction='sum')
        for batch in batches:
            batch['bdot_num_decisions'] = count
        return batches, lm_tokens

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs.update(self.model.last_update_metrics)
        if self.args.should_save:
            path = Path(self.args.output_dir) / 'metrics.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as handle:
                handle.write(json.dumps(dict(step=self.state.global_step, **logs), default=str) + '\n')
        return super().log(logs, start_time)

    def get_decay_parameter_names(self, model):
        names = super().get_decay_parameter_names(model)
        return [name for name in names if not any(part in name for part in ('norm', 'bias', 'film_'))]
