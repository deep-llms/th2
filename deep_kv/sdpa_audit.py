"""Observe real SDPA dispatch without changing masks, loss, or backend policy."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_SITE = ContextVar('sdpa_audit_site', default='unattributed')
_PREFIX = 'sdpa_call/'
_BACKENDS = ('aten::_scaled_dot_product_cudnn_attention',
             'aten::_scaled_dot_product_flash_attention',
             'aten::_scaled_dot_product_efficient_attention',
             'aten::_scaled_dot_product_attention_math')


def tensor_spec(value):
    if value is None:
        return None
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                requires_grad=value.requires_grad)


def runtime_metadata():
    versions = {}
    for name in ('torch', 'transformers', 'accelerate', 'nvidia-cudnn-cu13'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    driver = None
    if torch.cuda.is_available():
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        driver = result.stdout.strip() if result.returncode == 0 else 'query failed'
    return dict(packages=versions, torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), driver=driver,
        sdpa_enabled={key: getattr(torch.backends.cuda, key + '_sdp_enabled')()
                      for key in ('flash', 'mem_efficient', 'cudnn', 'math')},
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark)


class SDPAAudit:
    """Temporary API wrappers + profiler scopes; never selects a backend.

    Forward/backend events are attributed by CPU ancestry. Fused backward events
    are linked to their forward calls through autograd sequence/thread IDs.
    Saved-tensor metadata exposes actual tensors retained after autocast in
    training; eval has no such observations and reports that explicitly.
    """
    def __init__(self, model):
        self.model = model
        self.calls = []
        self.current = None
        self.backward_started = False

    def backward_marker(self, gradient):
        self.backward_started = True
        return gradient

    @contextmanager
    def observe(self):
        if self.calls:
            raise ValueError('Use a fresh audit per capture')
        original_sdpa = F.scaled_dot_product_attention
        original_hf = ALL_ATTENTION_FUNCTIONS['sdpa']
        auxiliary = self.model.aux
        original_aux = auxiliary.forward if auxiliary is not None else None
        had_forward = auxiliary is not None and 'forward' in auxiliary.__dict__

        def hf(module, *args, **kwargs):
            token = _SITE.set(f'backbone.layer_{module.layer_idx}')
            try:
                return original_hf(module, *args, **kwargs)
            finally:
                _SITE.reset(token)

        def aux(*args, **kwargs):
            site = 'auxiliary.detached_consumer' if kwargs.get('decode_only', False) else 'auxiliary.branch'
            token = _SITE.set(site)
            try:
                return original_aux(*args, **kwargs)
            finally:
                _SITE.reset(token)

        def sdpa(query, key, value, *args, **kwargs):
            mask = kwargs.get('attn_mask', args[0] if args else None)
            causal = kwargs.get('is_causal', args[2] if len(args) > 2 else False)
            call = dict(id=len(self.calls), site=_SITE.get(),
                stage='recompute' if self.backward_started else 'forward',
                query=tensor_spec(query), key=tensor_spec(key), value=tensor_spec(value),
                mask=tensor_spec(mask), is_causal=bool(causal),
                enable_gqa=kwargs.get('enable_gqa', False),
                autocast_enabled=torch.is_autocast_enabled(query.device.type),
                autocast_dtype=str(torch.get_autocast_dtype(query.device.type)),
                saved_tensors=[], forward_operators=[], backward_operators=[])
            self.calls.append(call)
            previous = self.current
            self.current = call
            try:
                with record_function(f'{_PREFIX}{call["id"]}/{call["site"]}'):
                    return original_sdpa(query, key, value, *args, **kwargs)
            finally:
                self.current = previous

        def pack(tensor):
            if self.current is not None:
                spec = tensor_spec(tensor)
                if spec not in self.current['saved_tensors']:
                    self.current['saved_tensors'].append(spec)
            return tensor

        activities = [ProfilerActivity.CPU]
        if next(self.model.parameters()).is_cuda:
            activities.append(ProfilerActivity.CUDA)
        try:
            F.scaled_dot_product_attention = sdpa
            ALL_ATTENTION_FUNCTIONS.register('sdpa', hf)
            if auxiliary is not None:
                auxiliary.forward = aux
            # API wrappers already record shapes. Avoid profiler shape recording
            # retaining large activation tensors and inflating checkpoint memory.
            with profile(activities=activities, record_shapes=False) as trace:
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    yield self
        finally:
            F.scaled_dot_product_attention = original_sdpa
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original_hf)
            if auxiliary is not None:
                if had_forward:
                    auxiliary.forward = original_aux
                else:
                    del auxiliary.forward
        self._summarize(trace)

    def _summarize(self, trace):
        sequences = {}
        events = trace.events()
        def owner(event):
            parent = event.cpu_parent
            while parent is not None:
                if parent.name.startswith(_PREFIX):
                    return int(parent.name.split('/')[1])
                parent = parent.cpu_parent
            return None
        for event in events:
            index = owner(event)
            if index is not None:
                if event.sequence_nr >= 0:
                    sequences[(event.thread, event.sequence_nr)] = index
                if event.name.startswith(_BACKENDS) and 'backward' not in event.name:
                    self.calls[index]['forward_operators'].append(event.name)
        self.unattributed_backward = []
        for event in events:
            if not event.name.startswith(_BACKENDS) or 'backward' not in event.name:
                continue
            parent = event
            index = None
            while parent is not None:
                key = (getattr(parent, 'fwd_thread', None), parent.sequence_nr)
                if key in sequences:
                    index = sequences[key]
                    break
                parent = parent.cpu_parent
            if index is None:
                self.unattributed_backward.append(event.name)
            else:
                self.calls[index]['backward_operators'].append(event.name)

    def result(self):
        missing = [c['id'] for c in self.calls if c['site'] == 'unattributed' or not c['forward_operators']]
        sites = {}
        for call in self.calls:
            site = sites.setdefault(call['site'], dict(forward_calls=0, recompute_calls=0,
                                                       forward_operators=Counter(), backward_operators=Counter()))
            site['recompute_calls' if call['stage'] == 'recompute' else 'forward_calls'] += 1
            site['forward_operators'].update(call['forward_operators'])
            site['backward_operators'].update(call['backward_operators'])
        return dict(calls=self.calls, sites=sites, unattributed_calls=missing,
                    unattributed_backward=self.unattributed_backward,
                    has_math=any('aten::_scaled_dot_product_attention_math' in c['forward_operators'] for c in self.calls),
                    mask_metadata='API arguments plus saved-tensor observations; saved tensors have no semantic labels; eval saves none')


@contextmanager
def trainer_audit(trainer, phase):
    """Record the first actual train/eval microbatch on rank zero each invocation."""
    if (trainer.args.device.type != 'cuda' or not trainer.is_world_process_zero()
            or phase in trainer._sdpa_seen):
        yield
        return
    if getattr(trainer.model, 'attention_backend', 'sdpa') == 'fa4':
        from .fa4 import trainer_audit as fa4_audit
        with fa4_audit(trainer, phase):
            yield
        trainer._sdpa_seen.add(phase)
        return
    audit = SDPAAudit(trainer.model)
    # Backward marker separates checkpoint recomputation from the first forward.
    trainer._active_sdpa_audit = audit
    try:
        with audit.observe():
            yield
    finally:
        trainer._active_sdpa_audit = None
    result = audit.result()
    result.update(runtime=runtime_metadata(), phase=phase, arm=trainer.model.arm,
        model_config=trainer.model.backbone.config.to_dict(),
        global_step=trainer.state.global_step, rank=trainer.args.process_index,
        world_size=trainer.args.world_size,
        checkpointing={key: getattr(trainer.model, key) for key in
                       ('checkpoint_layers', 'checkpoint_lm', 'checkpoint_aux')},
        note='First actual microbatch, includes startup; not a throughput measurement')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(trainer.args.output_dir)/f'sdpa-{phase}-{stamp}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    trainer._sdpa_receipts.append(path.name)
    trainer._sdpa_seen.add(phase)
    if result['unattributed_calls'] or result['unattributed_backward']:
        raise RuntimeError(f'Incomplete SDPA dispatch attribution: {path}')
