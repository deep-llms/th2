"""Correctness tests for language-balanced GroupReduce and Tiered RankLift."""

import copy
import os
import tempfile

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .compressed_baselines import GroupReduceEmbed
from .compression_init import frequency_group_ids_from_populations
from .loading import (
    EmbeddingShim,
    _infer_comp_config_from_state,
    load_compositional_model,
)
from .nonlinear_factorizations import TieredRankLiftEmbed
from .tied_head import make_tied_head


def _tiny_module():
    group_ids = torch.tensor([2, 0, 1, 2, 0, 2, 1, 1, 2])
    return TieredRankLiftEmbed(
        9,
        7,
        code_dims=(4, 3, 2),
        lift_dims=(0, 2, 3),
        group_ids=group_ids,
        rms_eps=3.25e-7,
    )


def test_tiered_ranklift_tied_values_and_gradients_match_materialized_table():
    torch.manual_seed(201)
    actual_embed = _tiny_module().double()
    reference_embed = copy.deepcopy(actual_embed)
    actual_hidden = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    reference_hidden = actual_hidden.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 3, 9, dtype=torch.float64)

    actual = make_tied_head(
        actual_embed, "tiered_ranklift", 9
    )(actual_hidden)
    reference = (
        reference_hidden @ reference_embed.materialize_effective_table().T
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-12)

    (actual * upstream).sum().backward()
    (reference * upstream).sum().backward()
    torch.testing.assert_close(
        actual_hidden.grad, reference_hidden.grad, rtol=1e-11, atol=1e-12
    )
    actual_parameters = dict(actual_embed.named_parameters())
    reference_parameters = dict(reference_embed.named_parameters())
    assert actual_parameters.keys() == reference_parameters.keys()
    for name in actual_parameters:
        assert actual_parameters[name].grad is not None, f"{name}: no gradient"
        torch.testing.assert_close(
            actual_parameters[name].grad,
            reference_parameters[name].grad,
            rtol=1e-10,
            atol=1e-12,
        )


def test_zero_lift_tiers_are_exact_groupreduce_blocks():
    torch.manual_seed(202)
    groups = torch.tensor([1, 0, 1, 0, 1, 0, 1])
    tiered = TieredRankLiftEmbed(
        7, 6, code_dims=(4, 2), lift_dims=(0, 0), group_ids=groups
    )
    control = GroupReduceEmbed(7, 6, group_ranks=(4, 2), group_ids=groups)
    with torch.no_grad():
        for group in range(2):
            control.left_factors[group].copy_(tiered.token_codes[group])
            control.right_factors[group].copy_(tiered.right_factors[group])

    ids = torch.arange(7).view(1, -1)
    tiered_table = tiered(ids)[0]
    control_table = control(ids)[0]
    torch.testing.assert_close(tiered_table, control_table, rtol=0, atol=0)
    hidden = torch.randn(2, 3, 6)
    torch.testing.assert_close(
        make_tied_head(tiered, "tiered_ranklift", 7)(hidden),
        make_tied_head(control, "groupreduce", 7)(hidden),
        rtol=0,
        atol=0,
    )


def test_tiered_ranklift_production_budget_and_delta_are_recorded():
    module = TieredRankLiftEmbed(
        151_936,
        1_024,
        code_dims=(1_024, 512, 192, 64),
        lift_dims=(0, 0, 320, 192),
        group_ids=torch.repeat_interleave(
            torch.arange(4),
            torch.tensor([2_048, 6_144, 24_576, 119_168]),
        ),
    )
    assert sum(parameter.numel() for parameter in module.parameters()) \
        == 20_096_000
    groupreduce_budget = sum(
        population * rank + 1_024 * rank
        for population, rank in zip(
            (2_048, 6_144, 24_576, 119_168),
            (1_024, 512, 192, 64),
        )
    )
    assert groupreduce_budget == 19_423_232
    assert 20_096_000 - groupreduce_budget == 672_768


def test_language_balanced_grouping_builder_is_deterministic_and_matched():
    from train_compositional import CompositionalArguments, build_arm

    importance = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6])
    expected_groups = frequency_group_ids_from_populations(
        torch.from_numpy(importance), (2, 2, 2, 2)
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "language_balanced.npz")
        np.savez(path, counts=importance)
        groupreduce_args = CompositionalArguments(
            arm="groupreduce",
            groupreduce_num_groups=4,
            groupreduce_ranks="6,4,3,2",
            groupreduce_populations="2,2,2,2",
            groupreduce_frequency_path=path,
            allow_from_scratch_baseline_init=True,
            tie_output=True,
        )
        tiered_args = CompositionalArguments(
            arm="tiered_ranklift",
            tiered_ranklift_code_dims="6,3,3,2",
            tiered_ranklift_lift_dims="0,1,2,2",
            tiered_ranklift_populations="2,2,2,2",
            tiered_ranklift_frequency_path=path,
            tie_output=True,
        )
        groupreduce = build_arm(groupreduce_args, 8, 6)
        tiered = build_arm(tiered_args, 8, 6)
        resumed = build_arm(
            tiered_args, 8, 6, initial_state=tiered.state_dict()
        )
        resumed.load_state_dict(tiered.state_dict(), strict=True)
        incompatible_args = copy.deepcopy(tiered_args)
        incompatible_args.tiered_ranklift_lift_dims = "0,1,2,3"
        with pytest.raises(ValueError, match="checkpoint structure"):
            build_arm(
                incompatible_args,
                8,
                6,
                initial_state=tiered.state_dict(),
            )

    assert torch.equal(groupreduce.group_ids, expected_groups)
    assert torch.equal(tiered.group_ids, expected_groups)
    assert tiered.group_sizes == groupreduce.group_sizes == (2, 2, 2, 2)
    torch.testing.assert_close(
        resumed.materialize_effective_table(),
        tiered.materialize_effective_table(),
        rtol=0,
        atol=0,
    )


def test_tiered_ranklift_state_inference_and_structure_validation():
    module = _tiny_module()
    state = module.state_dict()
    inferred = _infer_comp_config_from_state(state)
    assert inferred == {
        "arm": "tiered_ranklift",
        "tiered_ranklift_code_dims": "4,3,2",
        "tiered_ranklift_lift_dims": "0,2,3",
        "tiered_ranklift_populations": "2,3,4",
        "tiered_ranklift_rms_eps": 3.25e-7,
    }

    rebuilt = TieredRankLiftEmbed(
        9,
        7,
        code_dims=(4, 3, 2),
        lift_dims=(0, 2, 3),
        group_ids=state["group_ids"],
        rms_eps=3.25e-7,
    )
    rebuilt.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        rebuilt.materialize_effective_table(),
        module.materialize_effective_table(),
        rtol=0,
        atol=0,
    )

    corrupt = copy.deepcopy(state)
    corrupt["inverse_grouped_order"] = corrupt["inverse_grouped_order"].roll(1)
    with pytest.raises(ValueError, match="inverse token order"):
        rebuilt.load_state_dict(corrupt, strict=True)


def test_tiered_ranklift_bfloat16_backward_connects_every_parameter():
    module = TieredRankLiftEmbed(
        23,
        12,
        code_dims=(6, 4, 3),
        lift_dims=(0, 3, 5),
        group_ids=torch.arange(23).remainder(3),
    ).to(torch.bfloat16)
    # Input deliberately touches only group zero.  The full tied classifier
    # must nevertheless connect all tier parameters for DDP unused-parameter
    # detection to remain disabled safely.
    input_ids = torch.tensor([[0, 3, 6, 9]])
    hidden = torch.randn(
        1, 4, 12, dtype=torch.bfloat16, requires_grad=True
    )
    embeddings, _ = module(input_ids)
    logits = make_tied_head(module, "tiered_ranklift", 23)(hidden)
    loss = embeddings.float().square().mean() + logits.float().square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"{name}: no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name}: non-finite gradient"


def test_tiered_ranklift_configless_qwen_round_trip():
    torch.manual_seed(203)
    config = Qwen3Config(
        vocab_size=23,
        hidden_size=12,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config)
    groups = torch.arange(23).remainder(3)
    embed = TieredRankLiftEmbed(
        23,
        12,
        code_dims=(6, 4, 3),
        lift_dims=(0, 3, 5),
        group_ids=groups,
        rms_eps=4.5e-7,
    )
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "tiered_ranklift", 23)
    model.eval()
    input_ids = torch.randint(0, 23, (2, 5))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt"))
        loaded, inferred = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits

    assert inferred["arm"] == "tiered_ranklift"
    assert inferred["tiered_ranklift_code_dims"] == "6,4,3"
    assert inferred["tiered_ranklift_lift_dims"] == "0,3,5"
    assert inferred["tie_output"] is True
    assert loaded.lm_head.embed is loaded.model.embed_tokens.embed
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
