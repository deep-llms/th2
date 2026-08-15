"""No classification heads needed for the generative approach.

The model's own lm_head handles everything. Training uses the standard
causal LM loss with labels=-100 masking for prompt tokens. Evaluation
uses lm-evaluation-harness (log-likelihood scoring).

This file is kept for backward compatibility but contains no classes.
"""
