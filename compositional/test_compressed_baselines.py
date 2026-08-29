"""Focused correctness tests for published compressed embedding baselines."""

import itertools
import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from .compressed_baselines import (
    GroupReduceEmbed,
    PVQEmbed,
    SlimEmbed,
    TTEmbedding,
    balanced_exact_modes,
    balanced_padded_modes,
)
from .loading import EmbeddingShim, load_compositional_model
from .compression_init import (
    allocate_frequency_proportional_ranks,
    capacity_constrained_kmeans,
    frequency_group_ids,
    group_parameter_count,
    initialize_groupreduce_from_dense,
    initialize_pvq_from_dense,
    refine_groupreduce_from_dense,
    weighted_low_rank_factors,
)
from .tied_head import make_tied_head
from .test_tied_head import _tiny_qwen


def _effective_table(embed, vocab_size):
    ids = torch.arange(vocab_size).unsqueeze(0)
    return embed(ids)[0].squeeze(0)


def _assert_logits_and_gradients(embed, arm, vocab_size, embed_dim):
    hidden = torch.randn(2, 3, embed_dim, requires_grad=True)
    table = _effective_table(embed, vocab_size)
    actual = make_tied_head(embed, arm, vocab_size)(hidden)
    expected = hidden @ table.T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"{arm}.{name} has no gradient"
        assert torch.isfinite(parameter.grad).all(), \
            f"{arm}.{name} has non-finite gradients"


def test_pvq_exact_compact_algebra_parameters_and_exclusive_identity():
    assignments = torch.tensor([0, 0, 1, 2, 1, 2, 0])
    embed = PVQEmbed(
        vocab_size=7,
        embed_dim=6,
        shared_dim=4,
        num_codes=3,
        assignments=assignments,
    )
    table = _effective_table(embed, 7)
    expected = torch.cat((
        embed.codebook[assignments], embed.exclusive,
    ), dim=-1)
    torch.testing.assert_close(table, expected, rtol=0, atol=0)

    assert sum(parameter.numel() for parameter in embed.parameters()) == 3 * 4 + 7 * 2
    assert embed.assignments.requires_grad is False
    # Tokens 0 and 1 share the codebook slice but retain distinct exclusive
    # vectors and therefore are not forced to have identical classifiers.
    torch.testing.assert_close(table[0, :4], table[1, :4], rtol=0, atol=0)
    assert not torch.equal(table[0], table[1])
    _assert_logits_and_gradients(embed, "pvq", 7, 6)


def test_pvq_dense_initialization_uses_exact_fixed_assignment_centroids():
    dense = torch.arange(7 * 6, dtype=torch.float32).reshape(7, 6) / 10
    assignments = torch.tensor([0, 0, 1, 2, 1, 2, 0])
    embed = PVQEmbed(7, 6, 4, 3, assignments=assignments)
    embed.initialize_from_dense(dense)

    for code in range(3):
        expected = dense[assignments == code, :4].mean(dim=0)
        torch.testing.assert_close(embed.codebook[code], expected)
    torch.testing.assert_close(embed.exclusive, dense[:, 4:])


def test_pvq_rejects_missing_and_out_of_range_codes():
    with pytest.raises(ValueError, match="every P-VQ code"):
        PVQEmbed(5, 4, 3, 3, assignments=[0, 0, 1, 1, 0])
    with pytest.raises(ValueError, match="must be in"):
        PVQEmbed(5, 4, 3, 2, assignments=[0, 0, 1, 1, 2])
    embed = PVQEmbed(5, 4, 3, 2, assignments=[0, 0, 1, 1, 0])
    corrupt = embed.state_dict()
    corrupt["assignments"] = torch.tensor([0, 0, 1, 1, 2])
    with pytest.raises(ValueError, match="outside"):
        embed.load_state_dict(corrupt, strict=True)


def test_slim_mapping_is_balanced_persistent_and_rng_isolated():
    torch.manual_seed(123)
    first = SlimEmbed(31, 16, num_components=4, num_subvectors=32,
                      mapping_seed=10)
    torch.manual_seed(123)
    second = SlimEmbed(31, 16, num_components=4, num_subvectors=32,
                       mapping_seed=999)

    # Mapping uses private generators; changing its seed does not perturb the
    # learned subvector initialization drawn from the global model RNG.
    torch.testing.assert_close(first.subvectors, second.subvectors, rtol=0, atol=0)
    assert not torch.equal(first.mapping, second.mapping)
    for component in range(first.num_components):
        counts = torch.bincount(
            first.mapping[:, component], minlength=first.codes_per_component
        )
        assert counts.max().item() - counts.min().item() <= 1

    state = first.state_dict()
    assert "mapping" in state
    restored = SlimEmbed(31, 16, 4, 32, mapping=state["mapping"])
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        _effective_table(restored, 31), _effective_table(first, 31),
        rtol=0, atol=0,
    )
    _assert_logits_and_gradients(first, "slim", 31, 16)


@pytest.mark.parametrize("kwargs, message", [
    ({"embed_dim": 15, "num_components": 4, "num_subvectors": 32}, "divisible"),
    ({"embed_dim": 16, "num_components": 4, "num_subvectors": 30}, "divisible"),
    ({"embed_dim": 16, "num_components": 4, "num_subvectors": 128}, "cannot exceed"),
])
def test_slim_rejects_invalid_factorizations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SlimEmbed(vocab_size=11, **kwargs)


def test_groupreduce_supports_noncontiguous_unequal_groups_and_ranks():
    group_ids = torch.tensor([2, 0, 1, 2, 0, 2, 1, 1, 2])
    embed = GroupReduceEmbed(
        vocab_size=9,
        embed_dim=7,
        group_ranks=(1, 2, 3),
        group_ids=group_ids,
    )
    assert embed.group_sizes == (2, 3, 4)
    gathered = torch.cat([
        embed.token_ids_for_group(group) for group in range(3)
    ])
    assert sorted(gathered.tolist()) == list(range(9))

    expected = torch.empty(9, 7)
    for token in range(9):
        group = group_ids[token].item()
        offset = embed.group_offsets[token].item()
        expected[token] = (
            embed.left_factors[group][offset]
            @ embed.right_factors[group].T
        )
    torch.testing.assert_close(_effective_table(embed, 9), expected)
    _assert_logits_and_gradients(embed, "groupreduce", 9, 7)
    corrupt = embed.state_dict()
    corrupt["group_offsets"] = corrupt["group_offsets"].roll(1)
    with pytest.raises(ValueError, match="offsets"):
        embed.load_state_dict(corrupt, strict=True)


def _explicit_tt_table(embed):
    """Slow scalar definition of a TT matrix, independent of fast paths."""
    table = []
    for flat_i in range(embed.padded_vocab_size):
        i_digits = []
        value = flat_i
        for mode in reversed(embed.vocab_modes):
            i_digits.append(value % mode)
            value //= mode
        i_digits.reverse()
        row = []
        for j_digits in itertools.product(*[
            range(mode) for mode in embed.embedding_modes
        ]):
            product = embed.cores[0][:, i_digits[0], j_digits[0], :]
            for position in range(1, embed.order):
                product = product @ embed.cores[position][
                    :, i_digits[position], j_digits[position], :
                ]
            row.append(product.squeeze())
        table.append(torch.stack(row))
    return torch.stack(table)[:embed.vocab_size]


def test_tt_lookup_and_direct_output_match_scalar_definition_with_padding():
    embed = TTEmbedding(
        vocab_size=11,
        embed_dim=12,
        vocab_modes=(2, 2, 3),
        embedding_modes=(2, 2, 3),
        tt_ranks=(1, 2, 3, 1),
    )
    expected = _explicit_tt_table(embed)
    actual = _effective_table(embed, 11)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    hidden = torch.randn(2, 4, 12)
    logits = embed.project_hidden(hidden)
    assert logits.shape == (2, 4, 11)
    torch.testing.assert_close(logits, hidden @ expected.T,
                               rtol=1e-5, atol=1e-6)
    _assert_logits_and_gradients(embed, "tt", 11, 12)

    direct = TTEmbedding(
        11, 12, (2, 2, 3), (2, 2, 3), (1, 2, 3, 1),
        implementation="direct",
    )
    direct.load_state_dict(embed.state_dict(), strict=True)
    torch.testing.assert_close(
        _effective_table(direct, 11), expected, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        direct.project_hidden(hidden), hidden @ expected.T,
        rtol=1e-5, atol=1e-6,
    )


def test_tt_auto_shapes_are_valid_and_deterministic():
    assert balanced_padded_modes(151_936, 4) == (20, 20, 20, 19)
    assert math_prod(balanced_padded_modes(151_936, 3)) >= 151_936
    assert math_prod(balanced_exact_modes(1024, 3)) == 1024
    assert balanced_exact_modes(1024, 3) == (8, 8, 16)


def math_prod(values):
    result = 1
    for value in values:
        result *= value
    return result


@pytest.mark.parametrize("arm,embed,config", [
    (
        "pvq",
        PVQEmbed(31, 16, 12, 5),
        {"arm": "pvq", "pvq_shared_dim": 12, "pvq_num_codes": 5,
         "tie_output": True},
    ),
    (
        "slim",
        SlimEmbed(31, 16, 4, 32),
        {"arm": "slim", "slim_num_components": 4,
         "slim_num_subvectors": 32, "tie_output": True},
    ),
    (
        "groupreduce",
        GroupReduceEmbed(31, 16, (2, 3, 4, 5)),
        {"arm": "groupreduce", "groupreduce_num_groups": 4,
         "groupreduce_ranks": "2,3,4,5", "tie_output": True},
    ),
    (
        "tt",
        TTEmbedding(31, 16, (4, 4, 2), (2, 2, 4), (1, 3, 3, 1)),
        {"arm": "tt", "tt_order": 3, "tt_vocab_shape": "4,4,2",
         "tt_embedding_shape": "2,2,4", "tt_ranks": "3,3",
         "tie_output": True},
    ),
])
def test_tiny_qwen_save_load_and_configless_inference(arm, embed, config):
    torch.manual_seed(9)
    model = _tiny_qwen()
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, arm, 31)
    input_ids = torch.randint(0, 31, (2, 7))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

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

        # The persistent structural buffers and factor shapes are sufficient
        # for strict recovery even if the parent train_config is unavailable.
        os.remove(os.path.join(checkpoint_dir, "train_config.json"))
        inferred_model, inferred = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            inferred_logits = inferred_model(input_ids=input_ids).logits
        torch.testing.assert_close(inferred_logits, expected, rtol=0, atol=0)
        assert inferred["arm"] == arm
        assert inferred["tie_output"] is True


def test_published_compact_arms_require_tying():
    from train_compositional import (
        CompositionalArguments,
        validate_output_configuration,
    )

    for arm in ("pvq", "slim", "groupreduce", "tt"):
        args = CompositionalArguments(arm=arm, tie_output=False)
        with pytest.raises(ValueError, match="requires --tie_output"):
            validate_output_configuration(args)


def test_capacity_constrained_kmeans_is_balanced_and_reproducible():
    generator = torch.Generator().manual_seed(12)
    data = torch.randn(23, 5, generator=generator)
    first_centroids, first_assignments = capacity_constrained_kmeans(
        data, 4, num_iters=5, num_restarts=2, seed=7, chunk_size=6
    )
    second_centroids, second_assignments = capacity_constrained_kmeans(
        data, 4, num_iters=5, num_restarts=2, seed=7, chunk_size=6
    )
    torch.testing.assert_close(first_centroids, second_centroids, rtol=0, atol=0)
    torch.testing.assert_close(first_assignments, second_assignments, rtol=0, atol=0)
    counts = torch.bincount(first_assignments, minlength=4)
    assert counts.max().item() - counts.min().item() == 1
    for code in range(4):
        torch.testing.assert_close(
            first_centroids[code], data[first_assignments == code].mean(0)
        )


def test_pvq_conversion_state_reconstructs_centroids_and_exclusive_slice():
    dense = torch.randn(13, 8)
    assignments = torch.arange(13).remainder(3)
    state = initialize_pvq_from_dense(dense, 6, 3, assignments)
    embed = PVQEmbed(13, 8, 6, 3, assignments=assignments)
    embed.load_state_dict(state, strict=True)
    torch.testing.assert_close(embed.exclusive, dense[:, 6:])
    for code in range(3):
        torch.testing.assert_close(
            embed.codebook[code], dense[assignments == code, :6].mean(0)
        )


def test_groupreduce_weighted_svd_and_budget_allocation():
    dense = torch.randn(12, 5)
    counts = torch.tensor([100, 100, 50, 50, 20, 20, 10, 10, 5, 5, 1, 1],
                          dtype=torch.float64)
    groups = frequency_group_ids(counts, 3)
    # Stable tie-breaking keeps lower token ids first inside each frequency tie.
    assert groups.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    minimum_budget = group_parameter_count(groups, (1, 1, 1), 5)
    ranks = allocate_frequency_proportional_ranks(
        counts, groups, 5, minimum_budget + 30
    )
    assert group_parameter_count(groups, ranks, 5) <= minimum_budget + 30
    assert ranks[0] >= ranks[1] >= ranks[2]

    # Full-rank weighted SVD must reconstruct exactly regardless of row weights.
    left, right = weighted_low_rank_factors(dense[:4], counts[:4], rank=4)
    torch.testing.assert_close(left @ right.T, dense[:4], rtol=1e-5, atol=1e-5)

    full_ranks = (4, 4, 4)
    state = initialize_groupreduce_from_dense(
        dense, counts, groups, full_ranks
    )
    embed = GroupReduceEmbed(12, 5, full_ranks, group_ids=groups)
    embed.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        _effective_table(embed, 12), dense, rtol=1e-5, atol=1e-5
    )


def test_groupreduce_rank_allocator_rejects_zero_average_group():
    counts = torch.tensor([10.0, 5.0, 0.0, 0.0])
    groups = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="positive average frequency"):
        allocate_frequency_proportional_ranks(
            counts, groups, embed_dim=4, target_params=24
        )


def test_groupreduce_refinement_preserves_an_exact_partition():
    # Two obvious one-dimensional subspaces, deliberately mixed initially.
    dense = torch.tensor([
        [3.0, 0.0], [2.0, 0.0], [1.0, 0.0],
        [0.0, 3.0], [0.0, 2.0], [0.0, 1.0],
    ])
    counts = torch.ones(6)
    initial = torch.tensor([0, 1, 0, 0, 1, 1])
    state, refined, history = refine_groupreduce_from_dense(
        dense, counts, initial, (1, 1),
        max_iters=5, move_fraction=1.0, min_candidates=1, chunk_size=2,
    )
    assert refined.shape == initial.shape
    assert refined.min().item() == 0 and refined.max().item() == 1
    assert torch.bincount(refined, minlength=2).min().item() >= 1
    assert history and history[0]["moved"] > 0
    embed = GroupReduceEmbed(6, 2, (1, 1), group_ids=refined)
    embed.load_state_dict(state, strict=True)
    assert torch.isfinite(_effective_table(embed, 6)).all()


def test_pvq_dense_curriculum_quantizes_in_place_and_resumes_exactly():
    from .pvq_curriculum import (
        PVQCurriculumCallback,
        PVQ_CURRICULUM_STATE,
    )

    torch.manual_seed(3)
    model = _tiny_qwen()
    model.lm_head.weight = model.model.embed_tokens.weight
    assert model.get_input_embeddings().weight.data_ptr() \
        == model.get_output_embeddings().weight.data_ptr()
    callback = PVQCurriculumCallback(
        shared_dim=8,
        k_begin=5,
        k_end=3,
        k_decay=1,
        cluster_every=2,
        curriculum_steps=4,
        cluster_iters=2,
        cluster_restarts=1,
        cluster_chunk_size=7,
        cluster_device="cpu",
        seed=11,
        final_recluster=False,
    )
    parameter_id = id(model.get_input_embeddings().weight)
    args = SimpleNamespace(should_save=True)
    state = SimpleNamespace(global_step=0)
    callback.on_train_begin(args, state, None, model=model)
    assert id(model.get_input_embeddings().weight) == parameter_id
    assert callback.num_codes == 5
    counts = torch.bincount(callback.assignments, minlength=5)
    assert counts.max().item() - counts.min().item() <= 1
    shared = model.get_input_embeddings().weight[:, :8].detach()
    for code in range(5):
        rows = shared[callback.assignments == code]
        torch.testing.assert_close(rows, rows[0].expand_as(rows), rtol=0, atol=0)

    # The event at step 2 lowers K and can be checkpointed/resumed without
    # replaying the already-applied mutation.
    state.global_step = 2
    callback.on_step_begin(args, state, None, model=model)
    assert callback.num_codes == 4
    with tempfile.TemporaryDirectory() as output_dir:
        args.output_dir = output_dir
        checkpoint_dir = os.path.join(output_dir, "checkpoint-2")
        os.makedirs(checkpoint_dir)
        callback.on_save(args, state, None)
        assert os.path.isfile(os.path.join(checkpoint_dir, PVQ_CURRICULUM_STATE))
        resumed = PVQCurriculumCallback(
            shared_dim=8,
            k_begin=5,
            k_end=3,
            k_decay=1,
            cluster_every=2,
            curriculum_steps=4,
            cluster_iters=2,
            cluster_restarts=1,
            cluster_chunk_size=7,
            cluster_device="cpu",
            seed=11,
            resume_checkpoint=checkpoint_dir,
            final_recluster=False,
        )
        before = model.get_input_embeddings().weight.detach().clone()
        resumed.on_train_begin(args, state, None, model=model)
        torch.testing.assert_close(
            model.get_input_embeddings().weight, before, rtol=0, atol=0
        )
