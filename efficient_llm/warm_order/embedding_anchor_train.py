"""Bounded CPU fixture trainer for the anchor pilot's full-state contract.

Not a production loader: tensor fixtures are content-bound, but do not certify
the historical corpus/reservation. No CUDA or remote entry point is provided.
"""
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from .data import sha
from .embedding_anchor import EmbeddingAnchor, exposure
from .embedding_anchor_data import BoundRows
from .exposure_recovery import restore_rng
from .model import Config


def sources():
    root = Path(__file__).resolve().parents[2]
    names = ['warm_order/embedding_anchor.py', 'warm_order/embedding_anchor_train.py',
             'warm_order/model.py', 'dot_pool/model.py', 'warm_order/exposure_recovery.py',
             'warm_order/data.py', 'warm_order/embedding_anchor_data.py',
             'warm_order/adaptation_reservation.py', 'warm_order/embedding_anchor_prepare.py',
             'warm_order/embedding_anchor_eval.py', 'warm_order/adaptation_runtime.py']
    return {name: sha(root / 'efficient_llm' / name) for name in names}


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


class CPUFixtureTrainer:
    def __init__(self, config, tokens, row_ids, *, seed=0):
        if (tokens.device.type != 'cpu' or tokens.dtype != torch.int64 or tokens.ndim != 2
                or tokens.shape[1] != config.length or config.d > 64 or config.length > 64):
            raise ValueError('Tiny CPU fixture only; production admission absent')
        if (row_ids.device.type != 'cpu' or row_ids.dtype != torch.int64
                or row_ids.shape != (61440,) or len(torch.unique(row_ids)) != 61440
                or row_ids.min() < 0 or row_ids.max() >= len(tokens)):
            raise ValueError('Exactly61440 unique in-range training row IDs required')
        if tokens.min() < 0 or tokens.max() >= config.vocab:
            raise ValueError('Fixture token outside vocabulary')
        self.tokens, self.row_ids = tokens.clone(), row_ids.clone()
        self.recipe = dict(schema=1, fixture_only=True, model=asdict(config), seed=seed,
                           source=sources(), torch=str(torch.__version__), numpy=np.__version__,
                           threads=torch.get_num_threads(), tokens_shape=list(tokens.shape),
                           tokens_sha256=tensor_sha(tokens), rows_sha256=tensor_sha(row_ids))
        self._initialize(config, seed)

    def _initialize(self, config, seed):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.model = EmbeddingAnchor(config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.0006,
                                          betas=(.9, .95), eps=1e-8, weight_decay=0.)
        self.step = 0
        self.arm = 'shared'
        self.fork_sha256 = None
        self.trace = []

    def batch(self, ids):
        return self.tokens[ids]

    def validate(self):
        if (type(self.step) is not int or not 0 <= self.step <= 1920
                or self.arm not in ('shared', 'A', 'C')
                or (self.arm == 'shared' and self.step > 768)
                or (self.arm != 'shared' and (self.step < 768 or not self.fork_sha256))):
            raise ValueError('Invalid phase/fork state')
        if (len(self.trace) != self.step or any(r['step'] != s + 1 for s, r in enumerate(self.trace))):
            raise ValueError('Trace/cursor mismatch')
        if self.recipe['source'] != sources():
            raise ValueError('Source changed after initialization')
        counts = [0] * 4
        for s, row in enumerate(self.trace):
            expected = exposure(s, 'C' if self.arm == 'shared' else self.arm)
            if row['exposure'] != asdict(expected):
                raise ValueError('Trace exposure mismatch')
            ids = self.row_ids[expected.row_start:expected.row_stop]
            if row['row_ids_sha256'] != tensor_sha(ids):
                raise ValueError('Trace row order mismatch')
            for i in expected.path: counts[i] += 1
        group = self.optimizer.param_groups[0]
        wanted_lr = exposure(self.step - 1, 'C').lr if self.step else .0006
        if (len(self.optimizer.param_groups) != 1 or group['lr'] != wanted_lr
                or group['betas'] != (.9, .95) or group['eps'] != 1e-8
                or group['weight_decay'] != 0. or group['amsgrad'] or group['maximize']):
            raise ValueError('Optimizer policy mismatch')
        for name, p in self.model.named_parameters():
            if not torch.isfinite(p).all(): raise ValueError('Nonfinite parameter')
            count = counts[int(name.split('.')[1])] if name.startswith('blocks.') else self.step
            state = self.optimizer.state.get(p, {})
            if count == 0:
                if state: raise ValueError('Inactive parameter has Adam state')
            else:
                if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or state['step'].item() != count:
                    raise ValueError('Per-parameter Adam counter mismatch: ' + name)
                for key in ('exp_avg', 'exp_avg_sq'):
                    if (state[key].shape != p.shape or state[key].dtype != p.dtype
                            or not torch.isfinite(state[key]).all()):
                        raise ValueError('Invalid Adam moment')
                if (state['exp_avg_sq'] < 0).any(): raise ValueError('Negative Adam second moment')

    def advance(self, end):
        self.validate()
        cap = 768 if self.arm == 'shared' else 1920
        if type(end) is not int or not self.step < end <= cap:
            raise ValueError('Invalid stop or missing shared-checkpoint fork')
        self.model.train()
        for s in range(self.step, end):
            e = exposure(s, 'C' if self.arm == 'shared' else self.arm)
            ids = self.row_ids[e.row_start:e.row_stop]
            x = self.batch(ids)
            self.optimizer.param_groups[0]['lr'] = e.lr
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.nll(x, e.path, alpha=e.alpha, donor=e.donor).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.step = s + 1
            self.trace.append(dict(step=self.step, exposure=asdict(e),
                                   row_ids_sha256=tensor_sha(ids), loss=float(loss.detach()),
                                   grad_norm=float(norm)))
        self.validate()

    def save(self, path):
        self.validate()
        state = dict(recipe=self.recipe, step=self.step, arm=self.arm,
                     fork_sha256=self.fork_sha256, trace=self.trace,
                     parameter_names=[n for n, _ in self.model.named_parameters()],
                     model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=[],
                     numpy_rng=np.random.get_state(), python_rng=random.getstate())
        # Refuse overwriting snapshots, including the common warm-up ancestor.
        with Path(path).open('xb') as f: torch.save(state, f)
        return sha(path)

    @classmethod
    def restore(cls, path, digest, config, tokens, row_ids, *, seed=0, fork_arm=None):
        if sha(path) != digest: raise ValueError('Checkpoint hash mismatch')
        # Only locally created, explicitly hash-pinned trusted fixture checkpoints.
        state = torch.load(path, map_location='cpu', weights_only=False)
        trainer = cls(config, tokens, row_ids, seed=seed)
        return trainer._restore_state(state, digest, fork_arm)

    def _restore_state(self, state, digest, fork_arm):
        trainer = self
        if state['recipe'] != trainer.recipe:
            raise ValueError('Checkpoint source/runtime/data/model binding mismatch')
        if state['parameter_names'] != [n for n, _ in trainer.model.named_parameters()]:
            raise ValueError('Optimizer parameter identity/order mismatch')
        trainer.model.load_state_dict(state['model'], strict=True)
        trainer.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        trainer.step, trainer.arm = state['step'], state['arm']
        trainer.fork_sha256, trainer.trace = state['fork_sha256'], copy.deepcopy(state['trace'])
        trainer.validate()
        if fork_arm is not None:
            if fork_arm not in ('A', 'C') or trainer.arm != 'shared' or trainer.step != 768:
                raise ValueError('A/C may fork only from shared warm-up step768')
            trainer.arm, trainer.fork_sha256 = fork_arm, digest
            trainer.validate()
        restore_rng(state, 'cpu')
        return trainer


class CPUStreamTrainer(CPUFixtureTrainer):
    """Same update/fork engine over hash-bound read-only corpus memmaps.

Actual-data CPU work is restricted to40 warm-up updates for qualification.
This is not GPU admission or permission to perform a full production CPU fit.
"""
    def __init__(self, config, data, *, seed=0):
        if (not isinstance(data, BoundRows) or data.binding['seed'] != seed
                or config.length != data.length):
            raise ValueError('Bound data/seed/model mismatch')
        if not data.binding['fixture_only'] and config != Config():
            raise ValueError('Production model must match registered dimensions')
        self.data = data
        self.row_ids = torch.from_numpy(data.train_ids.copy())
        self.recipe = dict(schema=2, fixture_only=data.binding['fixture_only'],
                           model=asdict(config), seed=seed, source=sources(),
                           torch=str(torch.__version__), numpy=np.__version__,
                           threads=torch.get_num_threads(), data_binding=copy.deepcopy(data.binding))
        self._initialize(config, seed)

    def batch(self, ids):
        return self.data.batch(ids.numpy())

    def advance(self, end):
        if not self.recipe['fixture_only'] and (self.arm != 'shared' or end > 40):
            raise ValueError('Actual CPU qualification limited to40; full-fit admission pending')
        return super().advance(end)

    @classmethod
    def restore(cls, path, digest, config, data, *, seed=0, fork_arm=None):
        if sha(path) != digest: raise ValueError('Checkpoint hash mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        trainer = cls(config, data, seed=seed)
        return trainer._restore_state(state, digest, fork_arm)
