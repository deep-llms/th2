"""Frozen joint teacher and a shallow-only, single-pass correction student."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import CorrectionAdapter, FrozenQwen


class FeedbackOff(nn.Module):
    """The teacher's exact trained backbone, with its feedback branch removed."""
    def __init__(self, teacher):
        super().__init__()
        self.model = teacher.model

    def forward(self, context):
        return self.model.model(input_ids=context.input_ids,
            attention_mask=context.additive_mask(self.model.dtype),
            position_ids=context.position_ids, use_cache=False,
            return_dict=True).last_hidden_state

    def losses(self, hidden, context):
        return FrozenQwen.losses(self, hidden, context)


class CorrectionStudent(nn.Module):
    def __init__(self, teacher, arm, scale, *, seed=2901, checkpoint_layers=True):
        super().__init__()
        if teacher.arm != 'Deep' or teacher.adapter is None:
            raise ValueError('Student requires a trained Deep teacher')
        if arm not in ('LM', 'PCC') or not 1e-8 < scale < float('inf'):
            raise ValueError('Invalid distillation arm or correction scale')
        self.teacher = teacher.eval().requires_grad_(False)
        for parameter in self.teacher.parameters():
            parameter.grad = None
        self.arm, self.scale = arm, float(scale)
        self.s, self.d = teacher.s, teacher.d
        self.checkpoint_layers = checkpoint_layers
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.student = CorrectionAdapter(teacher.model.config.hidden_size,
                                             teacher.model.config.rms_norm_eps)
        self.student.to(next(teacher.parameters()).device)

    @property
    def model(self):
        return self.teacher.model

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _setup(self, context):
        hidden = self.model.model.embed_tokens(context.input_ids)
        rotary = self.model.model.rotary_emb(hidden, context.position_ids)
        mask = context.additive_mask(hidden.dtype)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        return hidden, rotary, mask, positions

    def _block(self, index, hidden, context, rotary, mask, positions):
        layer = self.model.model.layers[index]
        def call(x):
            return layer(x, attention_mask=mask, position_ids=context.position_ids,
                         cache_position=positions, position_embeddings=rotary, use_cache=False)
        if self.checkpoint_layers and torch.is_grad_enabled() and hidden.requires_grad:
            return checkpoint(call, hidden, use_reentrant=False)
        return call(hidden)

    @torch.no_grad()
    def teacher_correction(self, context):
        hidden, rotary, mask, positions = self._setup(context)
        for index in range(self.s):
            hidden = self._block(index, hidden, context, rotary, mask, positions)
        return self._target(hidden, context, rotary, mask, positions)

    @torch.no_grad()
    def _target(self, shallow, context, rotary, mask, positions):
        source = shallow
        for index in range(self.s, self.d):
            source = self._block(index, source, context, rotary, mask, positions)
        return self.teacher.adapter(shallow, source, context.allowed(), rotary).detach()

    def _forward(self, context, supervised):
        with torch.no_grad():
            hidden, rotary, mask, positions = self._setup(context)
            for index in range(self.s):
                hidden = self._block(index, hidden, context, rotary, mask, positions)
            target = self._target(hidden, context, rotary, mask, positions) if supervised else None
        prediction = self.student(hidden, hidden, context.allowed(), rotary)
        hidden = hidden + prediction
        for index in range(self.s, len(self.model.model.layers)):
            hidden = self._block(index, hidden, context, rotary, mask, positions)
        return self.model.model.norm(hidden), prediction, target

    def forward(self, context):
        # Evaluation/inference never executes the teacher source path.
        return self._forward(context, supervised=False)[0]

    def losses(self, hidden, context):
        return FrozenQwen.losses(self, hidden, context)

    def training_losses(self, context):
        hidden, prediction, target = self._forward(context, supervised=self.arm == 'PCC')
        lm_sums, counts = self.losses(hidden, context)
        correlation = torch.zeros((), device=hidden.device)
        if target is not None:
            eligible = context.targets()
            correlation = F.smooth_l1_loss(
                prediction[:, :-1][eligible].float() / self.scale,
                target[:, :-1][eligible].float() / self.scale, beta=1., reduction='sum')
        return lm_sums, counts, correlation


class StudentLoss(nn.Module):
    """LM head, alignment and checkpointed tail all live inside DDP forward."""
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, context):
        return self.student.training_losses(context)
