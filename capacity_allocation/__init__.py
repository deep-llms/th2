"""Qwen3 English capacity-allocation experiments (six-layer default)."""

from .modeling import AllocationConfig, AllocationForCausalLM, build_model, experiment_config

__all__ = ["AllocationConfig", "AllocationForCausalLM", "build_model", "experiment_config"]
