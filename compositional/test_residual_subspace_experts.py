"""Correctness tests for tied residual subspace experts."""

import copy
import json
import os
import tempfile
from dataclasses import asdict

import pytest
import torch

from .residual_subspace_experts import (
    ResidualSubspaceExpertsEmbed,
    residual_subspace_experts_parameter_count,
)
from .tied_head import make_tied_head


def _small_embed():
    return ResidualSubspaceExpertsEmbed(
        19,
        11,
        base_rank=5,
        expert_rank=4,
        num_experts=4,
        router_dim=3,
        top_k=2,
        router_temperature=0.75,
    )


def test_production_parameter_budget_is_exact_and_matched():
    expected = 19_471_968
    actual = residual_subspace_experts_parameter_count(
        151_936, 1_024, 120, 80, 12, 32
    )
    assert actual == expected
    lowrank_r128 = 151_936 * 128 + 1_024 * 128 + 1_024
    assert lowrank_r128 == 19_579_904
    assert lowrank_r128 - actual == 107_936
    assert (lowrank_r128 - actual) / lowrank_r128 < 0.006

    embed = _small_embed()
    assert embed.parameter_count == sum(
        parameter.numel() for parameter in embed.parameters()
    )


def test_initialization_is_exactly_the_global_low_rank_path():
    torch.manual_seed(2)
    embed = _small_embed().double()
    ids = torch.tensor([[0, 3, 8, 18]])
    hidden = torch.randn(2, 4, 11, dtype=torch.float64)

    expected_table = (
        embed.token_factors @ embed.base_proj.weight.T
        + embed.base_proj.bias
    )
    actual_table = embed.materialize()
    torch.testing.assert_close(actual_table, expected_table, rtol=0, atol=0)
    torch.testing.assert_close(
        embed(ids)[0], expected_table[ids], rtol=0, atol=0
    )
    torch.testing.assert_close(
        make_tied_head(embed, "residual_subspace_experts", 19)(hidden),
        hidden @ expected_table.T,
        rtol=0,
        atol=1e-15,
    )
    assert torch.count_nonzero(embed.expert_down_weight) == 0
    assert torch.count_nonzero(embed.expert_down_bias) == 0
    assert torch.count_nonzero(embed.expert_up_weight) > 0


def test_factorized_tied_logits_and_every_gradient_match_materialization():
    torch.manual_seed(3)
    actual_embed = _small_embed().double()
    reference_embed = copy.deepcopy(actual_embed)
    # Exercise the complete residual algebra instead of only its zero-init
    # special case.
    with torch.no_grad():
        actual_embed.expert_down_weight.normal_(std=0.1)
        actual_embed.expert_down_bias.normal_(std=0.1)
        actual_embed.expert_up_bias.normal_(std=0.1)
    reference_embed.load_state_dict(actual_embed.state_dict(), strict=True)

    actual_hidden = torch.randn(
        2, 3, 11, dtype=torch.float64, requires_grad=True
    )
    reference_hidden = actual_hidden.detach().clone().requires_grad_(True)
    weights = torch.randn(2, 3, 19, dtype=torch.float64)

    actual = make_tied_head(
        actual_embed, "residual_subspace_experts", 19
    )(actual_hidden)
    reference = reference_hidden @ reference_embed.materialize().T
    torch.testing.assert_close(actual, reference, rtol=0, atol=2e-15)

    (actual * weights).sum().backward()
    (reference * weights).sum().backward()
    torch.testing.assert_close(
        actual_hidden.grad, reference_hidden.grad, rtol=1e-12, atol=1e-12
    )
    reference_parameters = dict(reference_embed.named_parameters())
    for name, parameter in actual_embed.named_parameters():
        reference_parameter = reference_parameters[name]
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert reference_parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            rtol=1e-11,
            atol=1e-11,
        )


def test_routing_is_sparse_normalized_and_token_deterministic():
    torch.manual_seed(4)
    embed = _small_embed()
    ids = torch.tensor([[2, 7, 2], [18, 2, 7]])
    _, theta = embed(ids)
    assert theta.shape == (2, 3, 4)
    torch.testing.assert_close(theta.sum(-1), torch.ones(2, 3))
    assert torch.equal((theta > 0).sum(-1), torch.full((2, 3), 2))
    # The same id receives exactly the same route in every sequence position.
    torch.testing.assert_close(theta[0, 0], theta[0, 2], rtol=0, atol=0)
    torch.testing.assert_close(theta[0, 0], theta[1, 1], rtol=0, atol=0)
    torch.testing.assert_close(theta[0, 1], theta[1, 2], rtol=0, atol=0)

    aux_loss = embed.pop_router_aux_loss()
    assert aux_loss is not None and torch.isfinite(aux_loss)
    assert embed.pop_router_aux_loss() is None
    aux_loss.backward()
    # The auxiliary path uses pre-top-k probabilities, so all router keys—not
    # only currently selected keys—remain connected to a balancing gradient.
    assert embed.expert_keys.grad is not None
    assert torch.isfinite(embed.expert_keys.grad).all()
    assert torch.count_nonzero(embed.expert_keys.grad) > 0


def test_zero_init_has_useful_down_gradient_and_all_parameters_are_connected():
    torch.manual_seed(5)
    embed = _small_embed()
    hidden = torch.randn(2, 3, 11)
    logits = make_tied_head(
        embed, "residual_subspace_experts", 19
    )(hidden)
    logits.square().mean().backward()

    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"{name} is outside autograd"
        assert torch.isfinite(parameter.grad).all(), f"non-finite {name} grad"
    # The one-zero-factor initialization must permit residual learning on the
    # first step even though the step-zero residual function is exactly zero.
    assert embed.expert_down_weight.grad.abs().sum() > 0
    assert embed.expert_down_bias.grad.abs().sum() > 0
    assert embed.expert_up_bias.grad.abs().sum() > 0
    # The up weight and router acquire signal after the down path becomes
    # nonzero; zero first-step gradients here are intentional, not disconnection.
    assert torch.count_nonzero(embed.expert_up_weight.grad) == 0
    assert torch.count_nonzero(embed.expert_keys.grad) == 0
    assert torch.count_nonzero(embed.router_proj.weight.grad) == 0


def test_experts_with_no_routed_tokens_remain_in_autograd_for_ddp():
    torch.manual_seed(9)
    embed = _small_embed()
    with torch.no_grad():
        # Give every token one identical query and force top-2 to experts 0/1.
        embed.router_proj.weight.zero_()
        embed.router_proj.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        embed.expert_keys.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))
        embed.expert_down_weight.normal_(std=0.1)
    _, indices, _, _ = embed.route(
        embed.token_factors, return_dense=False
    )
    assert set(indices.flatten().tolist()) == {0, 1}

    hidden = torch.randn(2, 3, 11)
    make_tied_head(embed, "residual_subspace_experts", 19)(hidden).sum().backward()
    for name in (
        "expert_down_weight",
        "expert_down_bias",
        "expert_up_weight",
        "expert_up_bias",
    ):
        gradient = getattr(embed, name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[2:]) == 0

    embed.zero_grad(set_to_none=True)
    embed(torch.tensor([[0, 1, 2, 3]]))
    aux = embed.pop_router_aux_loss()
    assert aux is not None
    aux.backward()
    # Experts 2/3 are outside every hard top-2 route, but full pre-top-k
    # probabilities give their keys a recovery gradient through softmax.
    assert torch.count_nonzero(embed.expert_keys.grad[2:]) > 0


def test_eval_cache_is_exact_reused_and_invalidated_safely():
    torch.manual_seed(7)
    embed = _small_embed().double()
    with torch.no_grad():
        embed.expert_down_weight.normal_(std=0.1)
    head = make_tied_head(embed, "residual_subspace_experts", 19)
    hidden = torch.randn(2, 3, 11, dtype=torch.float64)

    embed.eval()
    with torch.no_grad():
        first = head(hidden)
        cache = embed._inference_expert_cache
        assert cache is not None
        second = head(hidden)
        assert embed._inference_expert_cache is cache
        expected = hidden @ embed.materialize().T
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, expected, rtol=0, atol=2e-15)

    embed.train()
    assert embed._inference_expert_cache is None
    embed.eval()
    with torch.no_grad():
        head(hidden)
    assert embed._inference_expert_cache is not None
    embed.load_state_dict(embed.state_dict(), strict=True)
    assert embed._inference_expert_cache is None

    embed.eval()
    with torch.no_grad():
        head(hidden)
    assert embed._inference_expert_cache is not None
    embed.float()
    assert embed._inference_expert_cache is None
    assert embed.routing_temperature.dtype == torch.float64
    embed.bfloat16()
    assert embed.routing_temperature.dtype == torch.float64

    # A training caller that does not use the auxiliary loss must not carry
    # its last graph into evaluation.
    embed.train()
    embed(torch.tensor([[1, 2, 3]]))
    assert embed._router_aux is not None
    embed.eval()
    assert embed._router_aux is None
    assert embed._output_route_cache is None


def test_bfloat16_forward_head_auxiliary_and_backward_are_finite():
    torch.manual_seed(10)
    embed = _small_embed().to(torch.bfloat16)
    with torch.no_grad():
        embed.expert_down_weight.normal_(std=0.1)
    ids = torch.tensor([[0, 3, 7, 18]])
    embeddings, theta = embed(ids)
    hidden = torch.randn(2, 3, 11, dtype=torch.bfloat16)
    logits = make_tied_head(
        embed, "residual_subspace_experts", 19
    )(hidden)
    aux = embed.pop_router_aux_loss()
    assert embeddings.dtype == torch.bfloat16
    assert theta.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(theta).all()
    assert torch.isfinite(logits).all()
    assert aux is not None and torch.isfinite(aux)
    (logits.float().square().mean() + 0.01 * aux.float()).backward()
    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"missing BF16 gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite BF16 {name}"


def test_input_and_head_consume_one_identical_vocabulary_route():
    torch.manual_seed(11)
    embed = _small_embed().to(torch.bfloat16)
    with torch.no_grad():
        embed.expert_down_weight.normal_(std=0.1)
        embed.expert_down_bias.normal_(std=0.1)
        embed.expert_up_bias.normal_(std=0.1)
    all_ids = torch.arange(19).unsqueeze(0)
    table, _ = embed(all_ids)
    assert embed._output_route_cache is not None

    # If the head tried to recompute routing using a differently shaped GPU
    # GEMM, this guard would fail. It must consume the exact route tensors
    # created by the input lookup.
    original_route = embed.route

    def forbid_second_route(*args, **kwargs):
        raise AssertionError("tied head recomputed the vocabulary route")

    embed.route = forbid_second_route
    hidden = torch.randn(2, 3, 11, dtype=torch.bfloat16)
    try:
        logits = make_tied_head(
            embed, "residual_subspace_experts", 19
        )(hidden)
    finally:
        embed.route = original_route
    expected = hidden @ table.squeeze(0).T
    torch.testing.assert_close(logits, expected, rtol=2e-2, atol=2e-2)
    assert embed._output_route_cache is None


def test_state_structure_is_strict_and_recovers_routing_hyperparameters():
    embed = _small_embed()
    state = embed.state_dict()
    structure = ResidualSubspaceExpertsEmbed.structure_from_state(state)
    assert structure == {
        "vocab_size": 19,
        "embed_dim": 11,
        "base_rank": 5,
        "expert_rank": 4,
        "num_experts": 4,
        "router_dim": 3,
        "top_k": 2,
        "router_temperature": 0.75,
    }

    corrupt = dict(state)
    corrupt["expert_up_weight"] = corrupt["expert_up_weight"][:, :-1]
    with pytest.raises(ValueError, match="up-projection weight"):
        ResidualSubspaceExpertsEmbed.structure_from_state(corrupt)

    missing = dict(state)
    missing.pop("routing_top_k")
    with pytest.raises(ValueError, match="invalid schema"):
        ResidualSubspaceExpertsEmbed.structure_from_state(missing)

    # Direct state loading must synchronize non-shape routing settings too.
    different = ResidualSubspaceExpertsEmbed(
        19,
        11,
        base_rank=5,
        expert_rank=4,
        num_experts=4,
        router_dim=3,
        top_k=1,
        router_temperature=0.5,
    )
    different.load_state_dict(state, strict=True)
    assert different.top_k == 2
    assert different.router_temperature == 0.75
    torch.testing.assert_close(
        different.materialize(), embed.materialize(), rtol=0, atol=0
    )


def test_build_and_configless_inference_recover_the_exact_architecture():
    from train_compositional import CompositionalArguments, build_arm
    from .loading import _build_arm_from_config, _infer_comp_config_from_state

    args = CompositionalArguments(
        arm="residual_subspace_experts",
        rse_base_rank=5,
        rse_expert_rank=4,
        rse_num_experts=4,
        rse_router_dim=3,
        rse_top_k=2,
        rse_router_temperature=0.75,
        tie_output=True,
    )
    built = build_arm(args, 19, 11)
    state = built.state_dict()
    inferred = _infer_comp_config_from_state(state)
    assert inferred == {
        "arm": "residual_subspace_experts",
        "rse_base_rank": 5,
        "rse_expert_rank": 4,
        "rse_num_experts": 4,
        "rse_router_dim": 3,
        "rse_top_k": 2,
        "rse_router_temperature": 0.75,
    }
    rebuilt = _build_arm_from_config(inferred, 19, 11, state=state)
    rebuilt.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        rebuilt.materialize(), built.materialize(), rtol=0, atol=0
    )

    args.rse_top_k = 1
    with pytest.raises(ValueError, match="does not match"):
        build_arm(args, 19, 11, initial_state=state)


def test_full_configless_hf_checkpoint_load_is_exact():
    from .loading import EmbeddingShim, load_compositional_model
    from .test_tied_head import _tiny_qwen

    torch.manual_seed(12)
    model = _tiny_qwen()
    embed = ResidualSubspaceExpertsEmbed(
        31,
        16,
        base_rank=6,
        expert_rank=4,
        num_experts=4,
        router_dim=3,
        top_k=2,
        router_temperature=0.75,
    )
    with torch.no_grad():
        embed.expert_down_weight.normal_(std=0.1)
        embed.expert_down_bias.normal_(std=0.1)
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(
        embed, "residual_subspace_experts", 31
    )
    model.eval()
    ids = torch.randint(0, 31, (2, 7))
    with torch.no_grad():
        before = model(input_ids=ids).logits

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        loaded, inferred = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            after = loaded(input_ids=ids).logits

    assert inferred["arm"] == "residual_subspace_experts"
    assert inferred["rse_base_rank"] == 6
    assert inferred["rse_expert_rank"] == 4
    assert inferred["rse_num_experts"] == 4
    assert inferred["rse_router_dim"] == 3
    assert inferred["rse_top_k"] == 2
    assert inferred["rse_router_temperature"] == 0.75
    assert inferred["tie_output"] is True
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_trainer_resume_preflight_accepts_exact_state_and_rejects_config_drift():
    from train_compositional import (
        CompositionalArguments,
        EmbeddingShim,
        validate_resume_compatibility,
    )
    from .test_tied_head import _tiny_qwen, _write_minimal_resume_state

    args = CompositionalArguments(
        arm="residual_subspace_experts",
        rse_base_rank=6,
        rse_expert_rank=4,
        rse_num_experts=4,
        rse_router_dim=3,
        rse_top_k=2,
        rse_router_temperature=0.75,
        lambda_div=0.01,
        tie_output=True,
    )
    model = _tiny_qwen()
    embed = ResidualSubspaceExpertsEmbed(
        31,
        16,
        base_rank=6,
        expert_rank=4,
        num_experts=4,
        router_dim=3,
        top_k=2,
        router_temperature=0.75,
    )
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(
        embed, "residual_subspace_experts", 31
    )

    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        _write_minimal_resume_state(checkpoint_dir, 7)
        with open(os.path.join(output_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": asdict(args)}, handle)

        validate_resume_compatibility(
            checkpoint_dir, args, expected_embed_module=embed
        )

        drifted = copy.deepcopy(args)
        drifted.rse_top_k = 1
        with pytest.raises(ValueError, match="rse_top_k"):
            validate_resume_compatibility(
                checkpoint_dir, drifted, expected_embed_module=embed
            )


def test_trainer_uses_full_probability_router_auxiliary_loss():
    from transformers import TrainingArguments
    from train_compositional import (
        CompositionalArguments,
        CompositionalTrainer,
        EmbeddingShim,
    )
    from .test_tied_head import _tiny_qwen

    torch.manual_seed(6)
    embed = ResidualSubspaceExpertsEmbed(
        31,
        16,
        base_rank=6,
        expert_rank=4,
        num_experts=4,
        router_dim=3,
        top_k=2,
    )
    model = _tiny_qwen()
    shim = EmbeddingShim(embed)
    model.model.embed_tokens = shim
    model.lm_head = make_tied_head(
        embed, "residual_subspace_experts", 31
    )
    comp_args = CompositionalArguments(
        arm="residual_subspace_experts",
        rse_base_rank=6,
        rse_expert_rank=4,
        rse_num_experts=4,
        rse_router_dim=3,
        rse_top_k=2,
        lambda_div=0.01,
        tie_output=True,
    )
    with tempfile.TemporaryDirectory() as directory:
        trainer = CompositionalTrainer(
            model=model,
            args=TrainingArguments(output_dir=directory, report_to=[]),
            embed_shim=shim,
            comp_args=comp_args,
        )
        decay_names = set(trainer.get_decay_parameter_names(model))
        assert "model.embed_tokens.embed.expert_keys" not in decay_names
        assert "model.embed_tokens.embed.token_factors" in decay_names
        assert "model.embed_tokens.embed.expert_down_weight" in decay_names
        ids = torch.randint(0, 31, (2, 7))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        down_before = embed.expert_down_weight.detach().clone()
        keys_before = embed.expert_keys.detach().clone()
        loss = trainer.compute_loss(model, {"input_ids": ids})
        assert torch.isfinite(loss)
        assert embed._router_aux is None
        loss.backward()
        assert embed.expert_keys.grad is not None
        assert torch.isfinite(embed.expert_keys.grad).all()
        assert torch.count_nonzero(embed.expert_keys.grad) > 0
        optimizer.step()
        assert not torch.equal(embed.expert_down_weight, down_before)
        assert not torch.equal(embed.expert_keys, keys_before)

        # Once the zero-initialized down path has learned, the up factors and
        # language-model router path must also receive nonzero gradients.
        optimizer.zero_grad(set_to_none=True)
        second_loss = trainer.compute_loss(model, {"input_ids": ids.flip(1)})
        assert torch.isfinite(second_loss)
        second_loss.backward()
        assert torch.count_nonzero(embed.expert_up_weight.grad) > 0
        optimizer.step()

        model.eval()
        with torch.no_grad():
            generated = model.generate(ids[:1, :3], max_new_tokens=2)
        assert generated.shape == (1, 5)
        assert embed._inference_expert_cache is not None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": 5}, "top_k"),
        ({"router_temperature": 0.0}, "temperature"),
        ({"router_temperature": 1e-8}, "temperature"),
        ({"router_temperature": float("nan")}, "temperature"),
    ],
)
def test_invalid_routing_configuration_fails_early(kwargs, message):
    base = dict(
        vocab_size=19,
        embed_dim=11,
        base_rank=5,
        expert_rank=4,
        num_experts=4,
        router_dim=3,
        top_k=2,
        router_temperature=1.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ResidualSubspaceExpertsEmbed(**base)
