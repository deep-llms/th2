"""Classification wrappers for causal LMs.

CausalLMClassifier: last-token pooling + linear head (AG News, SST-2,
XNLI, PAWS-X). CausalLMMultipleChoice: per-candidate scoring + linear
head (HellaSwag).
"""

import torch
import torch.nn as nn


class CausalLMClassifier(nn.Module):
    """Last-token pooling → linear classifier for standard classification."""

    def __init__(self, base_model, num_classes, hidden_size=1024):
        super().__init__()
        self.model = base_model
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device),
                        seq_lengths].float()
        return self.classifier(pooled)


class CausalLMMultipleChoice(nn.Module):
    """Per-candidate scoring for multiple-choice tasks (HellaSwag)."""

    def __init__(self, base_model, hidden_size=1024):
        super().__init__()
        self.model = base_model
        self.score_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.score_head.weight, std=0.02)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, input_ids, attention_mask):
        B, C, L = input_ids.shape
        flat_ids = input_ids.view(B * C, L)
        flat_mask = attention_mask.view(B * C, L)
        outputs = self.model(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        seq_lengths = flat_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(B * C, device=hidden.device),
                        seq_lengths].float()
        return self.score_head(pooled).view(B, C)
