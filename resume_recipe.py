"""Strict resume preflight; HF still owns checkpoint loading and data skipping.

Compare all TrainingArguments by default, excluding a small operational allowlist.
The receipt describes the model built for training, not a path to initial weights.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

# These do not change the train batches, objective, optimizer or schedule.
# Unknown/new HF options are compared by default rather than silently ignored.
OPERATIONAL = frozenset('''
output_dir resume_from_checkpoint stop_after local_rank do_train
logging_strategy logging_steps logging_first_step logging_nan_inf_filter
logging_dir log_level log_level_replica log_on_each_node disable_tqdm
report_to run_name project trackio_space_id trackio_bucket_id trackio_static_space_id
include_num_input_tokens_seen skip_memory_metrics
save_strategy save_steps save_total_limit save_on_each_node keep_checkpoint_steps
per_device_eval_batch_size prediction_loss_only eval_steps eval_delay
bf16_full_eval fp16_full_eval eval_do_concat_batches eval_use_gather_object
include_for_metrics eval_accumulation_steps batch_eval_metrics
ddp_timeout
'''.split())
SCHEMA = 1
FILENAME = 'training_recipe.json'


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def fp32_precision_settings(torch):
    """Read precision policy without changing it or mixing precision APIs.

    Modern HF sets cuDNN convolution precision independently of RNN precision.
    In that valid state PyTorch's legacy aggregate cudnn.allow_tf32 getter
    raises, so record the public per-operator settings on newer runtimes.
    """
    result = {'float32_matmul_precision': torch.get_float32_matmul_precision()}
    if hasattr(torch.backends, 'fp32_precision'):
        result['fp32_precision'] = {
            'global': torch.backends.fp32_precision,
            'cuda_matmul': torch.backends.cuda.matmul.fp32_precision,
            'cudnn': torch.backends.cudnn.fp32_precision,
            'cudnn_conv': torch.backends.cudnn.conv.fp32_precision,
            'cudnn_rnn': torch.backends.cudnn.rnn.fp32_precision,
        }
    else:
        result.update(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    return result


def build_training_recipe(model, args, attention_backend, accelerator=None):
    import torch
    options = {k: v for k, v in args.to_dict().items()
               if k not in OPERATIONAL and not k.startswith('hub_')}
    # This is forced by the packed entry point regardless of its CLI default.
    options['remove_unused_columns'] = False
    options['data_seed'] = args.seed if args.data_seed is None else args.data_seed
    config = model.config.to_dict()
    for key in ('_name_or_path', 'transformers_version', 'architectures', 'dtype', 'torch_dtype'):
        config.pop(key, None)
    # Config dtype can be serialization metadata. Actual parameter/buffer dtypes
    # and autocast/TF32 settings below determine the constructed training model.
    tensors = {'parameters': {n: [list(p.shape), str(p.dtype), p.requires_grad]
                              for n, p in model.named_parameters()},
               'buffers': {n: [list(b.shape), str(b.dtype)] for n, b in model.named_buffers()}}
    source = Path(__file__).parent
    code_files = ['train.py', 'resume_recipe.py', 'packed_training.py', 'packed_evaluation.py']
    if attention_backend in ('fa4', 'fa2'):
        code_files += ['packed_fa4.py']
        if attention_backend == 'fa2':
            code_files += ['packed_fa2.py']
    if hasattr(model.config, 'bdot'):
        code_files += ['bdot/model.py', 'bdot/routing.py', 'bdot/trainer.py', 'bdot/data.py', 'bdot/grouped.py', 'bdot/stopping.py', 'bdot/exit_training.py']
        if model.config.bdot.get('route_scope') == 'mlp':
            code_files += ['bdot/partial_mlp.py']
    return canonical({
        'schema': SCHEMA,
        'model': {'class': type(model).__module__ + '.' + type(model).__qualname__,
                  'config': config, 'attention_backend': attention_backend,
                  'tensor_schema_sha256': hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest()},
        'training': options,
        'execution': {'world_size': args.world_size, 'device_type': args.device.type,
                      'mixed_precision': (accelerator.mixed_precision if accelerator is not None
                                          else 'bf16' if args.bf16 else 'fp16' if args.fp16 else 'no'),
                      'distributed_type': str(args.distributed_state.distributed_type),
                      'train_batch_size_per_process': args.train_batch_size,
                      'global_batch_size': args.train_batch_size * args.world_size * args.gradient_accumulation_steps,
                      **fp32_precision_settings(torch),
                      'cudnn_benchmark': torch.backends.cudnn.benchmark,
                      'cudnn_deterministic': torch.backends.cudnn.deterministic,
                      'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
        'runtime': {**{name: version(name) for name in ('torch', 'transformers', 'accelerate')},
                    'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
        'code_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                       for name in code_files},
    })


def differences(old, new, prefix=''):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            path = f'{prefix}.{key}' if prefix else key
            if key not in old or key not in new:
                result.append(path)
            else:
                result.extend(differences(old[key], new[key], path))
        return result
    return [] if old == new else [prefix]


def require_resume_training(checkpoint, recipe, args):
    path = Path(checkpoint)
    receipt = path / FILENAME
    if not receipt.is_file():
        raise ValueError('Resume checkpoint lacks training_recipe.json; cannot verify the training recipe. '
                         'Keep the original implementation or start a new weight-initialized run.')
    saved = json.loads(receipt.read_text())
    if saved.get('schema') != SCHEMA:
        raise ValueError('Unsupported training recipe schema; refusing unverifiable resume')
    changed = differences(saved, recipe)
    if changed:
        raise ValueError('Resume training recipe differs: ' + ', '.join(changed[:30]) +
                         '. Keep the original recipe or start a new weight-initialized run.')
    if args.ignore_data_skip:
        raise ValueError('Resume requires ignore_data_skip=False to preserve data order')
    if args.auto_find_batch_size:
        raise ValueError('Guarded resume requires a fixed batch size; auto_find_batch_size is not validated')
    if args.deepspeed or args.fsdp or 'DEEPSPEED' in recipe['execution']['distributed_type'] or 'FSDP' in recipe['execution']['distributed_type']:
        raise ValueError('Guarded checkpoint completeness is validated for ordinary Trainer/DDP, not DeepSpeed/FSDP')
    state_path = path / 'trainer_state.json'
    if not state_path.is_file():
        raise ValueError('Incomplete resume checkpoint: trainer_state.json missing')
    state = json.loads(state_path.read_text())
    if state.get('train_batch_size') != recipe['execution']['train_batch_size_per_process']:
        raise ValueError('Saved effective train batch size differs from the requested batch size')
    step, end = state.get('global_step'), state.get('max_steps')
    if type(step) is not int or type(end) is not int or not 0 <= step <= end or end < 1:
        raise ValueError('Invalid checkpoint optimizer-step/schedule metadata')
    if step >= (args.stop_after if args.stop_after is not None else end):
        raise ValueError('Checkpoint already reached the requested stopping step or full schedule')
    world = recipe['execution']['world_size']
    required = ['optimizer.pt', 'scheduler.pt']
    required += ['rng_state.pth'] if world == 1 else [f'rng_state_{rank}.pth' for rank in range(world)]
    if recipe['execution']['mixed_precision'] == 'fp16' and recipe['execution']['device_type'] == 'cuda':
        required.append('scaler.pt')
    missing = [name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise ValueError('Incomplete resume checkpoint: missing/empty ' + ', '.join(missing))


def save_training_recipe(directory, recipe):
    path = Path(directory) / FILENAME
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(recipe, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)
