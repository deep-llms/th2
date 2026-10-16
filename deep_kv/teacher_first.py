"""Bounded frozen-backbone test of useful deep memory, before proxy distillation.

The memory belongs to the unchanged baseline trajectory. This is deliberately
an external teacher, not a claim of self-consistent recurrent feedback.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def rms(x):
    return F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6)


class MemoryBranch(nn.Module):
    def __init__(self, hidden_size, width=128, heads=2):
        super().__init__()
        if width % heads:
            raise ValueError('Code width must divide into heads')
        self.heads = heads
        self.extract = nn.Linear(hidden_size, width, bias=False)
        self.query = nn.Linear(hidden_size, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, hidden_size, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, shallow, source, context):
        b, length, _ = shallow.shape
        z = rms(self.extract(rms(source)))
        def heads(x):
            return x.reshape(b, length, self.heads, -1).transpose(1, 2)
        q, k, v = heads(self.query(rms(shallow))), heads(self.key(z)), heads(self.value(z))
        # No current-token or cross-document deep information is visible.
        strict = context.allowed() & torch.ones(length, length, device=q.device, dtype=torch.bool).tril(-1)
        nonempty = strict.any(-1, keepdim=True)
        dummy = torch.eye(length, device=q.device, dtype=torch.bool)[None, None]
        safe = strict | (~nonempty & dummy)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=safe, dropout_p=0.)
        y = y.masked_fill(~nonempty, 0).transpose(1, 2).reshape(b, length, -1)
        return self.out(y)


class TeacherFirst(nn.Module):
    def __init__(self, base, source='deep', consumer=4, target=8, seed=43, width=128):
        super().__init__()
        if base.arm != 'A' or source not in ('deep', 'shallow'):
            raise ValueError('Require a baseline A and deep/shallow source')
        if not 1 <= consumer < target <= len(base.backbone.model.layers):
            raise ValueError('Invalid 1-based blocks')
        self.base = base.requires_grad_(False).eval()
        self.source, self.consumer, self.target = source, consumer, target
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.branch = MemoryBranch(base.backbone.config.hidden_size, width)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def layout(self, context):
        if context.segments is None or not bool(context.valid.all()):
            raise ValueError('Require packed document-isolated inputs')
        if self.base.attention_backend == 'fa4':
            from .fa4 import document_layout
            return document_layout(context)
        return context.allowed()

    def tail(self, hidden, start, mask, rotary):
        for index in range(start, len(self.base.backbone.model.layers)):
            hidden = self.base.block(index, hidden, mask, rotary, None)[0]
        return self.base.backbone.model.norm(hidden)

    def nll(self, hidden, context):
        labels = (context.labels if context.labels is not None else context.input_ids)[:, 1:]
        labels = labels.masked_fill(~context.targets(), -100)
        rows = hidden.new_zeros(hidden.shape[0], dtype=torch.float32)
        def ce(h, y):
            logits = self.base.backbone.lm_head(h).float()
            return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100,
                                   reduction='none').view_as(y).sum(-1)
        for begin in range(0, labels.shape[1], self.base.lm_chunk):
            y = labels[:, begin:begin+self.base.lm_chunk]
            h = hidden[:, begin:begin+y.shape[1]]
            rows = rows + (checkpoint(ce, h, y, use_reentrant=False)
                           if torch.is_grad_enabled() and h.requires_grad else ce(h, y))
        return rows

    def forward(self, context, *, compare=False):
        mask = self.layout(context)
        with torch.no_grad():
            hidden = self.base.backbone.model.embed_tokens(context.input_ids)
            rotary = self.base.backbone.model.rotary_emb(hidden, context.position_ids)
            # Deep arm extracts the OUTPUT of target block 8; consumer receives
            # the INPUT of block 4. Shallow control skips unused teacher layers.
            stop = self.target if self.source == 'deep' else self.consumer-1
            shallow = hidden if self.consumer == 1 else None
            for index in range(stop):
                hidden = self.base.block(index, hidden, mask, rotary, None)[0]
                if index+1 == self.consumer-1:
                    shallow = hidden
            source = hidden if self.source == 'deep' else shallow
            baseline = self.nll(self.tail(hidden, stop, mask, rotary), context) if compare else None
        changed = shallow + self.branch(shallow, source, context).to(shallow.dtype)
        result = self.tail(changed, self.consumer-1, mask, rotary)
        nll = self.nll(result, context)
        counts = context.targets().sum(-1)
        return dict(nll=nll, counts=counts, baseline=baseline)


def make_trainer_class():
    """Use the ordinary HF loop with exact global scored-token normalization."""
    from transformers import Trainer
    from .training import DeepKVTrainer

    class BranchTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True

        def _get_num_items_in_batch(self, batches, device):
            if not batches:
                return None
            count = sum(DeepKVTrainer.context(x).targets().sum() for x in batches).to(device)
            return self.accelerator.reduce(count, reduction='sum')

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = model(DeepKVTrainer.context(inputs))
            count = result['counts'].sum() if num_items_in_batch is None else num_items_in_batch
            loss = result['nll'].sum() / count
            if num_items_in_batch is not None:
                loss = loss * self.accelerator.num_processes
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite branch objective')
            return (loss, result) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad(), self.compute_loss_context_manager():
                result = model(DeepKVTrainer.context(inputs), compare=True)
            rows = torch.stack((result['nll'], result['baseline'], result['counts']), -1).double()
            return None, rows, torch.zeros(len(rows), device=rows.device, dtype=torch.long)

    return BranchTrainer
