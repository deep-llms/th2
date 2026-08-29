"""Correctness and checkpoint tests for the Phase-1 Nested Ladder method."""

import json
import os
import tempfile

import pytest
import torch

from .compression_init import (
    frequency_group_ids_from_populations,
)
from .embeddings import LowRankEmbed
from .loading import EmbeddingShim, load_compositional_model
from .nested_ladder import (
    NestedLadderEmbed,
    nested_ladder_parameter_count,
)
from .test_tied_head import _tiny_qwen
from .tied_head import make_tied_head


def _table(embed):
    return embed.materialize()


def test_nested_ladder_tied_logits_are_exact_in_float64():
    counts = torch.tensor([5, 1, 9, 4, 9, 0, 2, 7, 3], dtype=torch.float64)
    embed = NestedLadderEmbed(9, 7, (2, 3, 4), (9, 5, 2), counts=counts)
    embed = embed.double()
    hidden = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)

    actual = make_tied_head(embed, "nested_ladder", 9)(hidden)
    expected = hidden @ _table(embed).T
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

    actual.square().sum().backward()
    assert hidden.grad is not None
    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"


def test_inplace_tied_head_gradients_match_materialized_reference():
    counts = torch.tensor([4, 9, 1, 7, 0, 3, 8], dtype=torch.float64)
    actual_embed = NestedLadderEmbed(
        7, 6, (2, 3, 4), (7, 4, 2), counts=counts
    ).double()
    reference_embed = NestedLadderEmbed(
        7,
        6,
        (2, 3, 4),
        (7, 4, 2),
        member_ids=(actual_embed.member_ids_2, actual_embed.member_ids_3),
    ).double()
    reference_embed.load_state_dict(actual_embed.state_dict(), strict=True)
    actual_hidden = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    reference_hidden = actual_hidden.detach().clone().requires_grad_(True)
    weights = torch.randn(2, 3, 7, dtype=torch.float64)

    actual = make_tied_head(actual_embed, "nested_ladder", 7)(actual_hidden)
    reference = reference_hidden @ reference_embed.materialize().T
    (actual * weights).sum().backward()
    (reference * weights).sum().backward()

    torch.testing.assert_close(
        actual_hidden.grad, reference_hidden.grad, rtol=1e-12, atol=1e-12
    )
    for (actual_name, actual_parameter), (
        reference_name, reference_parameter
    ) in zip(actual_embed.named_parameters(), reference_embed.named_parameters()):
        assert actual_name == reference_name
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=1e-12,
            atol=1e-12,
        )


def test_nested_ladder_parameter_budget_reference_and_irregular_case():
    expected = 18_637_824
    assert nested_ladder_parameter_count(
        151_936, 1_024, (64, 128, 320, 512),
        (151_936, 32_768, 8_192, 2_048),
    ) == expected

    embed = NestedLadderEmbed(
        11, 7, (2, 3, 5), (11, 6, 2), counts=torch.arange(11)
    )
    manual = 7 + 11 * 2 + 7 * 2 + 6 * 3 + 7 * 3 + 2 * 5 + 7 * 5
    assert embed.parameter_count == manual
    assert sum(parameter.numel() for parameter in embed.parameters()) == manual


def test_frequency_assignment_is_deterministic_nested_and_rebuildable():
    counts = torch.tensor([3, 8, 8, 0, 4, 4, 9, 1], dtype=torch.float64)
    first = NestedLadderEmbed(8, 6, (2, 3, 4), (8, 5, 2), counts=counts)
    second = NestedLadderEmbed(8, 6, (2, 3, 4), (8, 5, 2), counts=counts)
    assert torch.equal(first.member_ids_2, second.member_ids_2)
    assert torch.equal(first.member_ids_3, second.member_ids_3)
    assert set(first.member_ids_3.tolist()) <= set(first.member_ids_2.tolist())
    # Equal count 8 is resolved by token id: ids 1 then 2 in the rank order.
    assert first.member_slot_2[1] >= 0 and first.member_slot_2[2] >= 0

    rebuilt = NestedLadderEmbed(
        8,
        6,
        (2, 3, 4),
        (8, 5, 2),
        member_ids=(first.member_ids_2, first.member_ids_3),
    )
    rebuilt.load_state_dict(first.state_dict(), strict=True)
    torch.testing.assert_close(_table(rebuilt), _table(first), rtol=0, atol=0)

    changed_counts = counts.clone()
    changed_counts[3] = 100
    changed = NestedLadderEmbed(
        8, 6, (2, 3, 4), (8, 5, 2), counts=changed_counts
    )
    assert not torch.equal(first.member_ids_3, changed.member_ids_3)


def test_structure_uses_no_rng_and_corrupt_inverse_is_rejected():
    counts = torch.arange(8, dtype=torch.float64)
    torch.manual_seed(123)
    first = NestedLadderEmbed(8, 6, (2, 3, 4), (8, 5, 2), counts=counts)
    torch.manual_seed(123)
    changed_membership = NestedLadderEmbed(
        8, 6, (2, 3, 4), (8, 5, 2), counts=counts.flip(0)
    )
    for (first_name, first_parameter), (other_name, other_parameter) in zip(
        first.named_parameters(), changed_membership.named_parameters()
    ):
        assert first_name == other_name
        torch.testing.assert_close(
            first_parameter, other_parameter, rtol=0, atol=0
        )

    state = first.state_dict()
    corrupt = {key: value.clone() for key, value in state.items()}
    token = int(corrupt["member_ids_2"][0])
    corrupt["member_slot_2"][token] = -1
    rebuilt = NestedLadderEmbed(
        8,
        6,
        (2, 3, 4),
        (8, 5, 2),
        member_ids=(first.member_ids_2, first.member_ids_3),
    )
    with pytest.raises(ValueError, match="exact inverse"):
        rebuilt.load_state_dict(corrupt, strict=True)

    missing = dict(state)
    missing.pop("member_ids_2")
    with pytest.raises(ValueError, match="missing tier-2 buffers"):
        NestedLadderEmbed.structure_from_state(missing)


def test_depth_metrics_are_exact_device_side_sums_and_pop_once():
    counts = torch.arange(10, dtype=torch.float64)
    embed = NestedLadderEmbed(10, 6, (2, 3, 4), (10, 5, 2), counts=counts)
    ids = torch.tensor([[0, 5, 8, 9, 1, 9]])
    embeddings, _ = embed(ids)
    metrics = embed.pop_step_metrics()
    assert embed.pop_step_metrics() is None

    flat_ids = ids.flatten()
    norms = embeddings.detach().reshape(-1, 6).float().norm(dim=-1)
    depths = torch.ones_like(flat_ids)
    depths[embed.member_slot_2[flat_ids] >= 0] = 2
    depths[embed.member_slot_3[flat_ids] >= 0] = 3
    for depth in range(1, 4):
        metric_sum, metric_count = metrics[
            f"nested_depth_{depth}_embedding_norm"
        ]
        selected = norms[depths == depth]
        torch.testing.assert_close(metric_sum, selected.sum())
        assert int(metric_count) == selected.numel()


def test_tail_batch_keeps_every_tier_connected_to_autograd():
    counts = torch.arange(12, dtype=torch.float64)
    embed = NestedLadderEmbed(12, 8, (2, 3, 4), (12, 5, 2), counts=counts)
    # Token zero is outside every upper tier.
    output, _ = embed(torch.tensor([[0, 0, 0]]))
    output.sum().backward()
    for index, parameter in enumerate(embed.Z):
        assert parameter.grad is not None, f"Z.{index} was unused"
    for index, parameter in enumerate(embed.W):
        assert parameter.grad is not None, f"W.{index} was unused"


def test_single_tier_is_exactly_lowrank_embedding_and_head():
    torch.manual_seed(4)
    lowrank = LowRankEmbed(13, 7, rank=3).double()
    ladder = NestedLadderEmbed(13, 7, (3,), (13,)).double()
    with torch.no_grad():
        ladder.Z[0].copy_(lowrank.X)
        ladder.W[0].copy_(lowrank.proj.weight)
        ladder.bias.copy_(lowrank.proj.bias)

    ids = torch.tensor([[0, 4, 12]])
    hidden = torch.randn(2, 3, 7, dtype=torch.float64)
    torch.testing.assert_close(
        ladder(ids)[0], lowrank(ids)[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        make_tied_head(ladder, "nested_ladder", 13)(hidden),
        make_tied_head(lowrank, "lowrank", 13)(hidden),
        rtol=0,
        atol=0,
    )
    assert sum(p.numel() for p in ladder.parameters()) == sum(
        p.numel() for p in lowrank.parameters()
    )


def test_nested_ladder_configless_checkpoint_roundtrip_is_strict():
    torch.manual_seed(8)
    counts = torch.arange(31, dtype=torch.float64)
    embed = NestedLadderEmbed(31, 16, (2, 3, 4), (31, 10, 3), counts=counts)
    model = _tiny_qwen()
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "nested_ladder", 31)
    model.eval()
    input_ids = torch.randint(0, 31, (2, 6))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    config = {
        "arm": "nested_ladder",
        "nested_tier_ranks": "2,3,4",
        "nested_tier_populations": "31,10,3",
        "tie_output": True,
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt"))
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": config}, handle)

        loaded, loaded_config = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert loaded_config == config

        os.remove(os.path.join(checkpoint_dir, "train_config.json"))
        inferred_model, inferred = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            inferred_logits = inferred_model(input_ids=input_ids).logits
        torch.testing.assert_close(inferred_logits, expected, rtol=0, atol=0)
        assert inferred["arm"] == "nested_ladder"
        assert inferred["tie_output"] is True


def test_zero_row_reassignment_is_an_exact_noop():
    counts = torch.arange(14, dtype=torch.float64)
    embed = NestedLadderEmbed(14, 8, (2, 3, 4), (14, 7, 3), counts=counts)
    tier_2 = set(embed.member_ids_2.tolist())
    tier_3 = set(embed.member_ids_3.tolist())
    old_token = next(iter(tier_2 - tier_3))
    new_token = next(iter(set(range(14)) - tier_2))
    old_row = int(embed.member_slot_2[old_token])
    with torch.no_grad():
        embed.Z[1][old_row].zero_()
    hidden = torch.randn(2, 8)
    before_table = _table(embed).clone()
    before_logits = make_tied_head(embed, "nested_ladder", 14)(hidden).clone()

    embed.reassign_zero_row(2, old_token, new_token)
    torch.testing.assert_close(_table(embed), before_table, rtol=0, atol=0)
    torch.testing.assert_close(
        make_tied_head(embed, "nested_ladder", 14)(hidden),
        before_logits,
        rtol=0,
        atol=0,
    )


def test_matched_groupreduce_blocks_equal_nested_depth_level_sets():
    vocab_size = 40
    counts = torch.tensor(
        [float((token * 7) % 13) for token in range(vocab_size)]
    )
    populations = (40, 18, 7, 2)
    ladder = NestedLadderEmbed(
        vocab_size, 12, (2, 3, 4, 5), populations, counts=counts
    )
    level_populations = (
        populations[3],
        populations[2] - populations[3],
        populations[1] - populations[2],
        populations[0] - populations[1],
    )
    group_ids = frequency_group_ids_from_populations(
        counts, level_populations
    )
    tier_2, tier_3, tier_4 = (
        set(ladder.member_ids_2.tolist()),
        set(ladder.member_ids_3.tolist()),
        set(ladder.member_ids_4.tolist()),
    )
    nested_levels = (
        tier_4,
        tier_3 - tier_4,
        tier_2 - tier_3,
        set(range(vocab_size)) - tier_2,
    )
    for group, expected_ids in enumerate(nested_levels):
        actual_ids = set(torch.nonzero(group_ids == group).flatten().tolist())
        assert actual_ids == expected_ids


def test_training_builders_accept_nested_and_explicit_group_populations():
    from train_compositional import (
        CompositionalArguments,
        build_arm,
        validate_output_configuration,
    )

    # A zero-count low-id token and one-count high-id token catch accidental
    # pseudocount clamping: the explicit-population path must rank raw counts.
    counts = torch.tensor([0, 8, 8, 4, 7, 3, 2, 1], dtype=torch.float64)
    with tempfile.TemporaryDirectory() as directory:
        frequency_path = os.path.join(directory, "counts.pt")
        torch.save({"counts": counts}, frequency_path)

        nested_args = CompositionalArguments(
            arm="nested_ladder",
            nested_tier_ranks="2,3,4",
            nested_tier_populations="8,5,2",
            nested_frequency_path=frequency_path,
            tie_output=True,
        )
        nested = build_arm(nested_args, 8, 6)
        assert isinstance(nested, NestedLadderEmbed)
        assert validate_output_configuration(nested_args) == "tied"

        # On resume the state buffers are authoritative; the frequency file
        # may be unavailable and must not be contacted.
        nested_args.nested_frequency_path = "/does/not/exist"
        resumed = build_arm(
            nested_args, 8, 6, initial_state=nested.state_dict()
        )
        resumed.load_state_dict(nested.state_dict(), strict=True)
        torch.testing.assert_close(
            resumed.materialize(), nested.materialize(), rtol=0, atol=0
        )

        group_args = CompositionalArguments(
            arm="groupreduce",
            groupreduce_num_groups=3,
            groupreduce_ranks="4,3,2",
            groupreduce_populations="2,5,1",
            groupreduce_frequency_path=frequency_path,
            allow_from_scratch_baseline_init=True,
            tie_output=True,
        )
        grouped = build_arm(group_args, 8, 6)
        expected_ids = frequency_group_ids_from_populations(counts, (2, 5, 1))
        assert torch.equal(grouped.group_ids, expected_ids)
