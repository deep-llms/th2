"""Qwen3 English capacity-allocation experiments (six-layer default)."""

from .modeling import (ARMS, EXPECTED_COUNTS, FULLY_TIED_ARMS, NEW_ARMS,
                       ORIGINAL_ARMS, SHARED_ARMS, AllocationConfig,
                       AllocationForCausalLM, build_model, experiment_config)

__all__ = ["ARMS", "EXPECTED_COUNTS", "FULLY_TIED_ARMS", "NEW_ARMS",
           "ORIGINAL_ARMS", "SHARED_ARMS", "AllocationConfig",
           "AllocationForCausalLM", "build_model", "experiment_config"]
