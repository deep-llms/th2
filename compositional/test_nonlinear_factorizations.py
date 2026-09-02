"""Correctness tests for RankLift and its nearest nonlinear controls."""

import copy
import json
import math
import os
import tempfile

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from .compressed_baselines import TTEmbedding
from .loading import (
    EmbeddingShim,
    _infer_comp_config_from_state,
    load_compositional_model,
)
from .nonlinear_factorizations import (
    DeFINEEmbed,
    FunnelingEmbed,
    RankLiftEmbed,
)
from .tied_head import make_tied_head


def _assert_parameter_grads_match(actual, reference, rtol=1e-5, atol=1e-6):
    actual_named = dict(actual.named_parameters())
    reference_named = dict(reference.named_parameters())
    assert actual_named.keys() == reference_named.keys()
    for name in actual_named:
        assert actual_named[name].grad is not None, f"{name}: missing actual grad"
        assert reference_named[name].grad is not None, f"{name}: missing ref grad"
        torch.testing.assert_close(
            actual_named[name].grad,
            reference_named[name].grad,
            rtol=rtol,
            atol=atol,
        )


def test_ranklift_production_parameter_count_is_exact():
    embed = RankLiftEmbed(
        151_936, 1024, code_dim=124, lift_dim=336
    )
    assert sum(parameter.numel() for parameter in embed.parameters()) \
        == 19_396_128
    assert embed.feature_dim == 460


def test_reference_control_parameter_counts_are_exact():
    funneling = FunnelingEmbed(151_936, 1024, rank=128)
    define = DeFINEEmbed(
        151_936,
        1024,
        code_dim=112,
        expansion_dims=(656, 1184, 1724),
        group_counts=(16, 8, 4),
    )
    assert sum(parameter.numel() for parameter in funneling.parameters()) \
        == 19_579_904
    assert sum(parameter.numel() for parameter in define.parameters()) \
        == 19_579_556


def test_ranklift_head_matches_effective_table_values_and_gradients():
    torch.manual_seed(101)
    actual_embed = RankLiftEmbed(17, 12, code_dim=4, lift_dim=5)
    reference_embed = copy.deepcopy(actual_embed)
    actual_hidden = torch.randn(2, 3, 12, requires_grad=True)
    reference_hidden = actual_hidden.detach().clone().requires_grad_(True)
    weights = torch.randn(2, 3, 17)

    actual = make_tied_head(actual_embed, "ranklift", 17)(actual_hidden)
    expected = reference_hidden @ reference_embed.materialize_effective_table().T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(
        actual_hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6
    )
    # The two algebraically equivalent matmul association orders can sum FP32
    # products in a slightly different order.
    _assert_parameter_grads_match(
        actual_embed, reference_embed, rtol=3e-5, atol=1e-5
    )


def test_ranklift_nonlinear_features_raise_centered_rank_but_linear_lift_cannot():
    torch.manual_seed(102)
    embed = RankLiftEmbed(31, 16, code_dim=3, lift_dim=7)
    features = embed.materialize_features().detach().double()
    codes = embed.token_codes.detach().double()
    centered_codes = codes - codes.mean(dim=0, keepdim=True)
    centered_features = features - features.mean(dim=0, keepdim=True)

    linear_weight = torch.randn(3, 7, dtype=torch.double)
    linear_features = torch.cat((codes, codes @ linear_weight), dim=-1)
    centered_linear = linear_features - linear_features.mean(
        dim=0, keepdim=True
    )

    assert torch.linalg.matrix_rank(centered_codes) <= 3
    assert torch.linalg.matrix_rank(centered_linear) <= 3
    assert torch.linalg.matrix_rank(centered_features) > 3


def test_ranklift_initial_effective_table_scale_is_controlled():
    torch.manual_seed(107)
    embed = RankLiftEmbed(4096, 128, code_dim=32, lift_dim=64)
    table = embed.materialize_effective_table().detach()
    assert torch.isfinite(table).all()
    # The exact value varies with dimensions, but it must remain in the same
    # order of magnitude as Qwen's 0.02 initialization, not the O(0.1-1.0)
    # table produced by the rejected Xavier lift initialization.
    assert 0.001 < table.std().item() < 0.05


def test_ranklift_input_lookup_matches_effective_table_rows():
    torch.manual_seed(108)
    embed = RankLiftEmbed(29, 12, code_dim=4, lift_dim=7)
    input_ids = torch.tensor([[0, 7, 28], [11, 7, 2]])

    actual, auxiliary = embed(input_ids)
    expected = F.embedding(input_ids, embed.materialize_effective_table())

    assert auxiliary is None
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_ranklift_rejects_invalid_or_inconsistent_rms_epsilon():
    for invalid in (0.0, -1e-6, math.inf, -math.inf, math.nan):
        try:
            RankLiftEmbed(17, 12, code_dim=4, lift_dim=5, rms_eps=invalid)
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError(f"invalid rms_eps={invalid!r} was accepted")

    saved = RankLiftEmbed(
        17, 12, code_dim=4, lift_dim=5, rms_eps=3.25e-7
    ).state_dict()
    mismatched = RankLiftEmbed(
        17, 12, code_dim=4, lift_dim=5, rms_eps=1e-6
    )
    try:
        mismatched.load_state_dict(saved, strict=True)
    except RuntimeError as error:
        assert "encodes rms_eps=3.25e-07" in str(error)
    else:
        raise AssertionError("a mismatched RankLift rms_eps was silently loaded")


def test_funneling_input_lookup_matches_effective_table_rows():
    torch.manual_seed(103)
    embed = FunnelingEmbed(19, 11, rank=4)
    input_ids = torch.tensor([[0, 7, 18], [3, 7, 11]])

    actual, auxiliary = embed(input_ids)
    expected = F.embedding(input_ids, embed.materialize_effective_table())

    assert auxiliary is None
    # Batched selected-row GEMM and full-table GEMM may accumulate FP32 products
    # in a different order.
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-9)


def test_funneling_head_matches_effective_table_values_and_gradients():
    torch.manual_seed(103)
    actual_embed = FunnelingEmbed(19, 11, rank=4)
    reference_embed = copy.deepcopy(actual_embed)
    actual_hidden = torch.randn(2, 5, 11, requires_grad=True)
    reference_hidden = actual_hidden.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 5, 19)

    actual = make_tied_head(actual_embed, "funneling", 19)(actual_hidden)
    expected = reference_hidden @ reference_embed.materialize_effective_table().T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(
        actual_hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6
    )
    _assert_parameter_grads_match(
        actual_embed, reference_embed, rtol=3e-5, atol=1e-5
    )


def test_funneling_centered_effective_table_remains_rank_limited():
    torch.manual_seed(104)
    embed = FunnelingEmbed(19, 11, rank=4)
    # Recompute the tiny reference in FP64 so FP32 matmul roundoff cannot be
    # mistaken for additional singular directions by matrix_rank().
    features = embed.materialize_features().detach().double()
    table = features @ embed.projection.weight.detach().double().T
    table = table + embed.projection.bias.detach().double()
    centered = table - table.mean(dim=0, keepdim=True)
    assert torch.linalg.matrix_rank(centered) <= 4


def _released_define_expand_reference(embed, codes):
    """Literal bmm ordering used by the authors' released GroupLinear code."""
    original = codes
    state = codes
    flat_batch = codes.numel() // codes.shape[-1]
    for index, (output_dim, groups, weight, bias, norm) in enumerate(zip(
        embed.expansion_dims,
        embed.group_counts,
        embed.expand_weights,
        embed.expand_biases,
        embed.expand_norms,
    )):
        if index == 0:
            mixed = original.reshape(flat_batch, groups, -1)
        else:
            previous = state.reshape(flat_batch, groups, -1)
            direct = original.reshape(flat_batch, groups, -1)
            mixed = torch.cat((previous, direct), dim=-1)
        # Released implementation: [B,g,i] -> [g,B,i], bmm with
        # [g,i,o], add [g,1,o], then restore batch-first order.
        grouped = torch.bmm(mixed.transpose(0, 1), weight)
        grouped = grouped + bias.unsqueeze(1)
        grouped = F.gelu(norm(grouped.transpose(0, 1)))
        state = grouped.reshape(*codes.shape[:-1], output_dim)
    return state


def test_define_group_mixer_matches_released_code_and_has_tied_classifier():
    torch.manual_seed(104)
    embed = DeFINEEmbed(
        23,
        12,
        code_dim=4,
        expansion_dims=(8, 12),
        group_counts=(2, 1),
    )
    ids = torch.tensor([[0, 3, 7], [2, 5, 11]])
    input_embeddings, auxiliary = embed(ids)
    assert auxiliary is None
    assert input_embeddings.shape == (2, 3, 12)
    codes = F.embedding(ids, embed.token_codes)
    torch.testing.assert_close(
        embed.expand_codes(codes),
        _released_define_expand_reference(embed, codes),
        rtol=1e-6,
        atol=1e-7,
    )
    assert [tuple(weight.shape) for weight in embed.expand_weights] == [
        (2, 2, 4),
        (1, 12, 12),
    ]

    hidden = torch.randn(2, 3, 12)
    head = make_tied_head(embed, "define", 23)
    logits = head(hidden)
    expected = embed.output_projection(hidden) @ embed.token_codes.T
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    # Input and output paths together must connect every parameter for DDP with
    # find_unused_parameters=False.
    loss = input_embeddings.square().mean() + logits.square().mean()
    loss.backward()
    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"{name}: missing gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name}: non-finite gradient"


def test_configless_schema_inference_distinguishes_new_arms():
    ranklift = RankLiftEmbed(
        17, 12, code_dim=4, lift_dim=5, rms_eps=3.25e-7
    )
    funneling = FunnelingEmbed(17, 12, rank=4)
    define = DeFINEEmbed(
        17, 12, code_dim=4, expansion_dims=(8, 12), group_counts=(2, 1)
    )
    inferred_ranklift = _infer_comp_config_from_state(ranklift.state_dict())
    assert inferred_ranklift["arm"] == "ranklift"
    assert inferred_ranklift["ranklift_rms_eps"] == 3.25e-7
    assert _infer_comp_config_from_state(funneling.state_dict())["arm"] \
        == "funneling"
    inferred_define = _infer_comp_config_from_state(define.state_dict())
    assert inferred_define["arm"] == "define"
    assert inferred_define["define_expansion_dims"] == "8,12"
    assert inferred_define["define_group_counts"] == "2,1"


def test_ranklift_configless_qwen_round_trip_preserves_epsilon_and_logits():
    torch.manual_seed(109)
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
    embed = RankLiftEmbed(
        23, 12, code_dim=4, lift_dim=5, rms_eps=3.25e-7
    )
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "ranklift", 23)
    model.eval()
    input_ids = torch.randint(0, 23, (2, 5))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        loaded, inferred = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits

    loaded_embed = loaded.model.embed_tokens.embed
    assert inferred == {
        "arm": "ranklift",
        "ranklift_code_dim": 4,
        "ranklift_lift_dim": 5,
        "ranklift_rms_eps": 3.25e-7,
        "tie_output": True,
    }
    assert loaded_embed.rms_eps == 3.25e-7
    assert loaded.lm_head.embed is loaded_embed
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ranklift_bfloat16_qwen_backward_with_gradient_checkpointing_is_finite():
    torch.manual_seed(110)
    config = Qwen3Config(
        vocab_size=23,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    embed = RankLiftEmbed(
        23, 12, code_dim=4, lift_dim=5, rms_eps=1e-6
    ).to(torch.bfloat16)
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "ranklift", 23)
    model.gradient_checkpointing_enable()
    model.train()

    input_ids = torch.randint(0, 23, (1, 6))
    loss = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
    ).loss
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"{name}: missing gradient"
        assert torch.isfinite(parameter.grad).all(), \
            f"{name}: non-finite gradient"


def test_funneling_configless_qwen_round_trip_preserves_logits():
    torch.manual_seed(111)
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
    embed = FunnelingEmbed(23, 12, rank=4)
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "funneling", 23)
    model.eval()
    input_ids = torch.randint(0, 23, (2, 5))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        loaded, inferred = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits

    loaded_embed = loaded.model.embed_tokens.embed
    assert inferred == {
        "arm": "funneling",
        "funneling_rank": 4,
        "tie_output": True,
    }
    assert loaded.lm_head.embed is loaded_embed
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_funneling_bfloat16_qwen_backward_with_gradient_checkpointing_is_finite():
    torch.manual_seed(112)
    config = Qwen3Config(
        vocab_size=23,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    embed = FunnelingEmbed(23, 12, rank=4).to(torch.bfloat16)
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "funneling", 23)
    model.gradient_checkpointing_enable()
    model.train()

    input_ids = torch.randint(0, 23, (1, 6))
    loss = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
    ).loss
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"{name}: missing gradient"
        assert torch.isfinite(parameter.grad).all(), \
            f"{name}: non-finite gradient"


def test_define_qwen_save_load_round_trip_preserves_both_paths():
    torch.manual_seed(106)
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
    embed = DeFINEEmbed(
        23, 12, code_dim=4, expansion_dims=(8, 12), group_counts=(2, 1)
    )
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "define", 23)
    model.eval()
    input_ids = torch.randint(0, 23, (2, 5))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    comp_config = {
        "arm": "define",
        "define_code_dim": 4,
        "define_expansion_dims": "8,12",
        "define_group_counts": "2,1",
        "tie_output": True,
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": comp_config}, handle)
        loaded, loaded_config = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits

    assert loaded_config == comp_config
    assert loaded.lm_head.embed is loaded.model.embed_tokens.embed
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_training_builder_constructs_all_new_arms():
    from train_compositional import CompositionalArguments, build_arm

    ranklift = build_arm(
        CompositionalArguments(
            arm="ranklift", ranklift_code_dim=4, ranklift_lift_dim=5
        ),
        23,
        12,
    )
    funneling = build_arm(
        CompositionalArguments(arm="funneling", funneling_rank=4),
        23,
        12,
    )
    define = build_arm(
        CompositionalArguments(
            arm="define",
            define_code_dim=4,
            define_expansion_dims="8,12",
            define_group_counts="2,1",
        ),
        23,
        12,
    )
    assert isinstance(ranklift, RankLiftEmbed)
    assert isinstance(funneling, FunnelingEmbed)
    assert isinstance(define, DeFINEEmbed)


def test_tt_chunked_materialization_matches_direct_values_and_gradients():
    torch.manual_seed(105)
    materialized = TTEmbedding(
        31,
        16,
        vocab_modes=(4, 4, 2),
        embedding_modes=(2, 2, 4),
        tt_ranks=(1, 3, 3, 1),
        implementation="materialize",
        materialize_chunk_size=5,
    )
    direct = copy.deepcopy(materialized)
    direct.implementation = "direct"
    hidden_a = torch.randn(2, 3, 16, requires_grad=True)
    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weights = torch.randn(2, 3, 31)

    logits_a = materialized.project_hidden(hidden_a)
    logits_b = direct.project_hidden(hidden_b)
    torch.testing.assert_close(logits_a, logits_b, rtol=1e-5, atol=1e-6)
    (logits_a * weights).sum().backward()
    (logits_b * weights).sum().backward()
    torch.testing.assert_close(hidden_a.grad, hidden_b.grad, rtol=1e-5, atol=1e-6)
    _assert_parameter_grads_match(materialized, direct, rtol=1e-4, atol=1e-5)
