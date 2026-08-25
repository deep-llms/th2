"""Regression tests for tied compositional output heads.

These tests cover the algebra, gradient flow, serialization ownership, and a
complete save/load round trip for every supported tied-output arm.
"""

import copy
import json
import os
import tempfile
from dataclasses import asdict
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import (
    HfArgumentParser,
    Qwen3Config,
    Qwen3ForCausalLM,
    TrainingArguments,
)

from .embeddings import (
    ANTEmbed,
    LowRankEmbed,
    OriginalANT,
    ResidualANTEmbed,
    SharedLocalEmbed,
    PureLocalEmbed,
)
from .compressed_baselines import (
    PVQEmbed,
    SlimEmbed,
    GroupReduceEmbed,
    TTEmbedding,
)
from .loading import (
    EmbeddingShim,
    _load_checkpoint_tensors,
    is_compositional,
    load_compositional_model,
)
from .tied_head import (
    INDEPENDENT_OUTPUT_FILENAME,
    IndependentLowRankHead,
    make_tied_head,
)


VOCAB_SIZE = 31
HIDDEN_SIZE = 16
CODEBOOK_SIZE = 8
BASE_DIM = 6
KEY_DIM = 4


def _arm_cases():
    return [
        (
            "lowrank",
            LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM),
            {"arm": "lowrank", "d_x": BASE_DIM, "tie_output": True},
        ),
        (
            "shared_local",
            SharedLocalEmbed(
                VOCAB_SIZE,
                HIDDEN_SIZE,
                shared_rank=BASE_DIM // 2,
                local_rank=BASE_DIM // 2,
                num_groups=4,
            ),
            {
                "arm": "shared_local",
                "shared_rank": BASE_DIM // 2,
                "local_embed_rank": BASE_DIM // 2,
                "num_groups": 4,
                "tie_output": True,
            },
        ),
        (
            "pure_local",
            PureLocalEmbed(
                VOCAB_SIZE,
                HIDDEN_SIZE,
                rank=BASE_DIM,
                num_groups=4,
            ),
            {
                "arm": "pure_local",
                "pure_local_rank": BASE_DIM,
                "num_groups": 4,
                "tie_output": True,
            },
        ),
        (
            "pvq",
            PVQEmbed(
                VOCAB_SIZE, HIDDEN_SIZE,
                shared_dim=12, num_codes=5,
            ),
            {
                "arm": "pvq",
                "pvq_shared_dim": 12,
                "pvq_num_codes": 5,
                "tie_output": True,
            },
        ),
        (
            "slim",
            SlimEmbed(
                VOCAB_SIZE, HIDDEN_SIZE,
                num_components=4, num_subvectors=32,
            ),
            {
                "arm": "slim",
                "slim_num_components": 4,
                "slim_num_subvectors": 32,
                "tie_output": True,
            },
        ),
        (
            "groupreduce",
            GroupReduceEmbed(
                VOCAB_SIZE, HIDDEN_SIZE,
                group_ranks=(2, 3, 4, 5),
            ),
            {
                "arm": "groupreduce",
                "groupreduce_num_groups": 4,
                "groupreduce_ranks": "2,3,4,5",
                "tie_output": True,
            },
        ),
        (
            "tt",
            TTEmbedding(
                VOCAB_SIZE, HIDDEN_SIZE,
                vocab_modes=(4, 4, 2),
                embedding_modes=(2, 2, 4),
                tt_ranks=(1, 3, 3, 1),
            ),
            {
                "arm": "tt",
                "tt_order": 3,
                "tt_vocab_shape": "4,4,2",
                "tt_embedding_shape": "2,2,4",
                "tt_ranks": "3,3",
                "tie_output": True,
            },
        ),
        (
            "original_ant",
            OriginalANT(VOCAB_SIZE, CODEBOOK_SIZE, HIDDEN_SIZE),
            {"arm": "original_ant", "K": CODEBOOK_SIZE, "tie_output": True},
        ),
        (
            "ant",
            ANTEmbed(
                VOCAB_SIZE,
                CODEBOOK_SIZE,
                HIDDEN_SIZE,
                d_x=BASE_DIM,
                d_k=KEY_DIM,
            ),
            {
                "arm": "ant",
                "K": CODEBOOK_SIZE,
                "d_x": BASE_DIM,
                "d_k": KEY_DIM,
                "gamma": 1.0,
                "num_heads": 1,
                "tie_output": True,
            },
        ),
        (
            "residual_ant",
            ResidualANTEmbed(
                VOCAB_SIZE,
                CODEBOOK_SIZE,
                HIDDEN_SIZE,
                d_x=BASE_DIM,
                d_k=KEY_DIM,
            ),
            {
                "arm": "residual_ant",
                "K": CODEBOOK_SIZE,
                "d_x": BASE_DIM,
                "d_k": KEY_DIM,
                "gamma": 1.0,
                "num_heads": 1,
                "tie_output": True,
            },
        ),
    ]


def _tiny_qwen():
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config)


def _install_tied_embedding(model, embed, arm):
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, arm, VOCAB_SIZE)


def _install_independent_lowrank(model, embed):
    model.model.embed_tokens = EmbeddingShim(embed)
    head = IndependentLowRankHead(embed)
    model.lm_head = head
    return head


def _write_minimal_resume_state(checkpoint_dir, step):
    """Create nonempty Trainer-state fixtures for resume validation tests."""
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as handle:
        json.dump({"global_step": step}, handle)
    torch.save({"state": {}, "param_groups": []}, os.path.join(
        checkpoint_dir, "optimizer.pt"
    ))
    torch.save({"last_epoch": step}, os.path.join(
        checkpoint_dir, "scheduler.pt"
    ))
    torch.save({"cpu": torch.random.get_rng_state()}, os.path.join(
        checkpoint_dir, "rng_state.pth"
    ))


def test_tied_heads_match_explicit_tables_and_backpropagate():
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, HIDDEN_SIZE)

    for arm, embed, _ in _arm_cases():
        head = make_tied_head(embed, arm, VOCAB_SIZE)
        all_ids = torch.arange(VOCAB_SIZE).unsqueeze(0)
        table, _ = embed(all_ids)
        expected = hidden @ table.squeeze(0).T
        actual = head(hidden)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        weights = torch.randn_like(actual)
        (actual * weights).sum().backward()
        for name, parameter in embed.named_parameters():
            assert parameter.grad is not None, f"{arm}.{name}: missing gradient"
            assert torch.isfinite(parameter.grad).all(), \
                f"{arm}.{name}: non-finite gradient"


def test_independent_lowrank_clones_factors_without_alias_or_rng_change():
    torch.manual_seed(8)
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    with torch.no_grad():
        embed.proj.bias.copy_(torch.linspace(-0.2, 0.2, HIDDEN_SIZE))

    rng_before = torch.random.get_rng_state().clone()
    head = IndependentLowRankHead(embed)
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_after, rng_before)
    torch.testing.assert_close(head.X, embed.X, rtol=0, atol=0)
    torch.testing.assert_close(
        head.proj_weight, embed.proj.weight, rtol=0, atol=0
    )
    assert head.X is not embed.X
    assert head.proj_weight is not embed.proj.weight
    assert head.X.data_ptr() != embed.X.data_ptr()
    assert head.proj_weight.data_ptr() != embed.proj.weight.data_ptr()

    hidden = torch.randn(2, 5, HIDDEN_SIZE)
    explicit_table = head.X @ head.proj_weight.T
    actual = head(hidden)
    expected = hidden @ explicit_table.T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    # The input embedding bias is deliberately absent from the independent
    # classifier. In the tied head it contributes one common scalar per hidden
    # state, so both heads still define exactly the same token probabilities.
    tied_logits = make_tied_head(embed, "lowrank", VOCAB_SIZE)(hidden)
    common_shift = (hidden @ embed.proj.bias).unsqueeze(-1)
    torch.testing.assert_close(
        tied_logits - actual,
        common_shift.expand_as(tied_logits),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        tied_logits.log_softmax(-1), actual.log_softmax(-1),
        rtol=1e-5, atol=1e-6,
    )

    input_before = embed.X.detach().clone()
    with torch.no_grad():
        head.X.add_(1.0)
    torch.testing.assert_close(embed.X, input_before, rtol=0, atol=0)


def test_independent_lowrank_qwen_gradients_ownership_and_generation():
    torch.manual_seed(10)
    model = _tiny_qwen()
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    head = _install_independent_lowrank(model, embed)

    parameter_ids = [id(parameter) for parameter in model.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))
    for parameter in list(embed.parameters()) + list(head.parameters()):
        assert parameter_ids.count(id(parameter)) == 1
    assert set(key for key in model.state_dict() if key.startswith("lm_head.")) \
        == {"lm_head.X", "lm_head.proj_weight"}

    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    input_x_before = embed.X.detach().clone()
    output_x_before = head.X.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in list(embed.named_parameters()) + list(
        head.named_parameters()
    ):
        assert parameter.grad is not None, f"{name}: missing gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name}: non-finite gradient"
        assert parameter.grad.abs().sum() > 0, f"{name}: zero gradient"

    optimizer.step()
    assert not torch.equal(embed.X, input_x_before)
    assert not torch.equal(head.X, output_x_before)
    assert not torch.equal(embed.X, head.X)

    model.eval()
    with torch.no_grad():
        generated = model.generate(input_ids[:1, :3], max_new_tokens=2)
    assert generated.shape == (1, 5)


def test_shared_local_balanced_padding_and_backward_are_exact():
    """Non-divisible V uses every group and ignores only padding rows."""
    torch.manual_seed(9)
    vocab_size, num_groups = 10, 4
    embed = SharedLocalEmbed(
        vocab_size, HIDDEN_SIZE,
        shared_rank=3, local_rank=3, num_groups=num_groups,
    )
    explicit_embed = copy.deepcopy(embed)

    bounds = [embed.group_bounds(group) for group in range(num_groups)]
    sizes = [end - start for start, end in bounds]
    assert bounds == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert min(sizes) > 0
    assert max(sizes) - min(sizes) <= 1

    hidden = torch.randn(2, 5, HIDDEN_SIZE, requires_grad=True)
    explicit_hidden = hidden.detach().clone().requires_grad_(True)
    weights = torch.randn(2, 5, vocab_size)

    actual = make_tied_head(embed, "shared_local", vocab_size)(hidden)
    all_ids = torch.arange(vocab_size).unsqueeze(0)
    explicit_table, _ = explicit_embed(all_ids)
    expected = explicit_hidden @ explicit_table.squeeze(0).T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(hidden.grad, explicit_hidden.grad,
                               rtol=1e-5, atol=1e-6)
    for (name, parameter), (explicit_name, explicit_parameter) in zip(
        embed.named_parameters(), explicit_embed.named_parameters()
    ):
        assert name == explicit_name
        torch.testing.assert_close(
            parameter.grad, explicit_parameter.grad, rtol=1e-5, atol=1e-6
        )

    valid_factor_mask = (
        torch.arange(embed.group_size).unsqueeze(0)
        < torch.tensor(embed.group_sizes).unsqueeze(1)
    )
    padding_grad = embed.token_factors.grad[~valid_factor_mask]
    assert torch.count_nonzero(padding_grad) == 0


def test_shared_local_g16_divisible_fast_path_is_exact():
    """The production G=16 layout must take the padded-free batched path."""
    torch.manual_seed(15)
    vocab_size, num_groups = 32, 16
    embed = SharedLocalEmbed(
        vocab_size, HIDDEN_SIZE,
        shared_rank=3, local_rank=3, num_groups=num_groups,
    )
    assert embed.num_large_groups == 0
    assert embed.group_size == 2
    assert all(size == 2 for size in embed.group_sizes)

    hidden = torch.randn(2, 5, HIDDEN_SIZE, requires_grad=True)
    head = make_tied_head(embed, "shared_local", vocab_size)
    actual = head(hidden)
    table, _ = embed(torch.arange(vocab_size).unsqueeze(0))
    expected = hidden @ table.squeeze(0).T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    actual.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"G16 {name}: missing gradient"
        assert torch.isfinite(parameter.grad).all(), \
            f"G16 {name}: non-finite gradient"


def test_pure_local_balanced_padding_and_g16_fast_path_are_exact():
    """Pure-local handles both uneven tiny groups and the production layout."""
    torch.manual_seed(16)
    for vocab_size, num_groups in ((10, 4), (32, 16)):
        embed = PureLocalEmbed(
            vocab_size, HIDDEN_SIZE, rank=BASE_DIM, num_groups=num_groups
        )
        with torch.no_grad():
            embed.bias.copy_(
                torch.linspace(-0.25, 0.25, HIDDEN_SIZE)
            )
        expected_params = (
            num_groups * embed.group_size * BASE_DIM
            + num_groups * HIDDEN_SIZE * BASE_DIM
            + HIDDEN_SIZE
        )
        assert sum(parameter.numel() for parameter in embed.parameters()) \
            == expected_params
        explicit_embed = copy.deepcopy(embed)
        hidden = torch.randn(2, 5, HIDDEN_SIZE, requires_grad=True)
        explicit_hidden = hidden.detach().clone().requires_grad_(True)
        weights = torch.randn(2, 5, vocab_size)

        actual = make_tied_head(embed, "pure_local", vocab_size)(hidden)
        table, _ = explicit_embed(torch.arange(vocab_size).unsqueeze(0))
        expected = explicit_hidden @ table.squeeze(0).T
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(
            hidden.grad, explicit_hidden.grad, rtol=1e-5, atol=1e-6
        )
        for (name, parameter), (other_name, other_parameter) in zip(
            embed.named_parameters(), explicit_embed.named_parameters()
        ):
            assert name == other_name
            torch.testing.assert_close(
                parameter.grad, other_parameter.grad, rtol=1e-5, atol=1e-6
            )

        valid_factor_mask = (
            torch.arange(embed.group_size).unsqueeze(0)
            < torch.tensor(embed.group_sizes).unsqueeze(1)
        )
        assert torch.count_nonzero(
            embed.token_factors.grad[~valid_factor_mask]
        ) == 0

    production_params = (
        151_936 * 128 + 16 * 1024 * 128 + 1024
    )
    assert production_params == 21_545_984


def test_pure_local_tied_qwen_uses_every_group_when_inputs_do_not():
    """The tied classifier keeps all local bases active for DDP training."""
    torch.manual_seed(17)
    vocab_size, num_groups = 32, 16
    model = _tiny_qwen()
    # Override the helper's V=31 model with a divisible production-style V.
    config = copy.deepcopy(model.config)
    config.vocab_size = vocab_size
    model = Qwen3ForCausalLM(config)
    embed = PureLocalEmbed(
        vocab_size, HIDDEN_SIZE, rank=BASE_DIM, num_groups=num_groups
    )
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "pure_local", vocab_size)

    # Every input token is in group zero. All other local bases can receive a
    # gradient only through the exactly tied full-vocabulary output path.
    input_ids = torch.tensor([[0, 1, 0, 1, 0, 1]])
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()

    for group in range(num_groups):
        weight_grad = embed.local_weight.grad[group]
        factor_grad = embed.token_factors.grad[group]
        assert torch.isfinite(weight_grad).all()
        assert torch.isfinite(factor_grad).all()
        assert torch.count_nonzero(weight_grad) > 0, \
            f"group {group} basis was unused"
        assert torch.count_nonzero(factor_grad) > 0, \
            f"group {group} token factors were unused"

    assert embed.bias.grad is not None
    assert torch.isfinite(embed.bias.grad).all()


def test_tied_embedding_has_one_registered_owner():
    for arm, embed, _ in _arm_cases():
        model = _tiny_qwen()
        _install_tied_embedding(model, embed, arm)

        assert model.lm_head.embed is model.model.embed_tokens.embed
        assert "_embed_ref" not in model.lm_head._modules
        assert not any(key.startswith("lm_head.") for key in model.state_dict()), \
            f"{arm}: embedding tensors were registered under lm_head"


def test_shared_local_optimizer_owns_each_parameter_exactly_once():
    model = _tiny_qwen()
    embed = SharedLocalEmbed(
        VOCAB_SIZE, HIDDEN_SIZE,
        shared_rank=BASE_DIM // 2,
        local_rank=BASE_DIM // 2,
        num_groups=4,
    )
    _install_tied_embedding(model, embed, "shared_local")

    model_parameters = list(model.parameters())
    parameter_ids = [id(parameter) for parameter in model_parameters]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert list(model.lm_head.parameters()) == []
    for parameter in embed.parameters():
        assert parameter_ids.count(id(parameter)) == 1

    optimizer = torch.optim.AdamW(model_parameters, lr=1e-3)
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    for parameter in embed.parameters():
        assert optimizer_ids.count(id(parameter)) == 1

    before = {
        name: parameter.detach().clone()
        for name, parameter in embed.named_parameters()
    }
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert not torch.equal(parameter, before[name]), f"{name} did not update"


def test_tied_qwen_save_and_load_round_trip():
    torch.manual_seed(11)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))

    for arm, embed, comp_config in _arm_cases():
        model = _tiny_qwen()
        _install_tied_embedding(model, embed, arm)
        model.eval()

        with torch.no_grad():
            logits_before = model(input_ids=input_ids).logits

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            torch.save(embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt"))
            with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
                json.dump({"compositional": comp_config}, handle)

            loaded, loaded_config = load_compositional_model(
                checkpoint_dir, device="cpu", dtype=torch.float32
            )
            with torch.no_grad():
                logits_after = loaded(input_ids=input_ids).logits

        assert loaded_config["tie_output"] is True
        assert loaded.lm_head.embed is loaded.model.embed_tokens.embed
        torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)


def test_independent_lowrank_qwen_save_and_load_round_trip():
    torch.manual_seed(21)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model = _tiny_qwen()
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    head = _install_independent_lowrank(model, embed)
    model.eval()

    # Make input and output observably independent before serializing.
    with torch.no_grad():
        head.X.add_(torch.linspace(-0.1, 0.1, VOCAB_SIZE).unsqueeze(1))
    with torch.no_grad():
        logits_before = model(input_ids=input_ids).logits

    comp_config = {
        "arm": "lowrank",
        "d_x": BASE_DIM,
        "tie_output": False,
        "independent_lowrank_output": True,
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        torch.save(
            head.state_dict(),
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": comp_config}, handle)

        loaded, loaded_config = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            logits_after = loaded(input_ids=input_ids).logits

    assert loaded_config == comp_config
    assert isinstance(loaded.lm_head, IndependentLowRankHead)
    assert loaded.lm_head.X.data_ptr() \
        != loaded.model.embed_tokens.embed.X.data_ptr()
    torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)


def test_independent_lowrank_hf_state_restores_for_trainer_resume():
    """Trainer resume restores both independently registered factor sets."""
    torch.manual_seed(22)
    model = _tiny_qwen()
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    head = _install_independent_lowrank(model, embed)
    with torch.no_grad():
        head.X.add_(0.25)
        head.proj_weight.mul_(1.5)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        saved_state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))

        resumed = _tiny_qwen()
        resumed_embed = LowRankEmbed(
            VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM
        )
        _install_independent_lowrank(resumed, resumed_embed)
        incompatible = resumed.load_state_dict(saved_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            resumed.state_dict()[key], value, rtol=0, atol=0
        )


def test_shared_local_hf_state_restores_directly_for_trainer_resume():
    """HF's model state alone must load into an already-installed custom arm."""
    torch.manual_seed(12)
    model = _tiny_qwen()
    embed = SharedLocalEmbed(
        VOCAB_SIZE, HIDDEN_SIZE,
        shared_rank=BASE_DIM // 2,
        local_rank=BASE_DIM // 2,
        num_groups=4,
    )
    _install_tied_embedding(model, embed, "shared_local")

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        saved_state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))

        resumed = _tiny_qwen()
        resumed_embed = SharedLocalEmbed(
            VOCAB_SIZE, HIDDEN_SIZE,
            shared_rank=BASE_DIM // 2,
            local_rank=BASE_DIM // 2,
            num_groups=4,
        )
        _install_tied_embedding(resumed, resumed_embed, "shared_local")
        incompatible = resumed.load_state_dict(saved_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for key, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value,
                                   rtol=0, atol=0)


def test_pure_local_hf_state_and_resume_validation_are_exact():
    """Pure-local must restore every factor and pass strict Trainer preflight."""
    from train_compositional import (
        CompositionalArguments,
        validate_resume_compatibility,
    )

    torch.manual_seed(18)
    model = _tiny_qwen()
    embed = PureLocalEmbed(
        VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM, num_groups=4
    )
    _install_tied_embedding(model, embed, "pure_local")
    args = CompositionalArguments(
        arm="pure_local",
        pure_local_rank=BASE_DIM,
        num_groups=4,
        tie_output=True,
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
            checkpoint_dir,
            args,
            expected_embed_module=embed,
        )

        saved_state = load_file(os.path.join(
            checkpoint_dir, "model.safetensors"
        ))
        resumed = _tiny_qwen()
        resumed_embed = PureLocalEmbed(
            VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM, num_groups=4
        )
        _install_tied_embedding(resumed, resumed_embed, "pure_local")
        incompatible = resumed.load_state_dict(saved_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            resumed.state_dict()[key], value, rtol=0, atol=0
        )


def test_shared_local_periodic_checkpoint_saves_embedding_once():
    from train_compositional import SaveEmbeddingCallback

    embed = SharedLocalEmbed(
        VOCAB_SIZE, HIDDEN_SIZE,
        shared_rank=BASE_DIM // 2,
        local_rank=BASE_DIM // 2,
        num_groups=4,
    )
    shim = EmbeddingShim(embed)
    callback = SaveEmbeddingCallback(shim)

    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        os.makedirs(checkpoint_dir)
        callback.on_save(
            SimpleNamespace(output_dir=output_dir, should_save=True),
            SimpleNamespace(global_step=7),
            control=None,
        )
        saved = torch.load(
            os.path.join(checkpoint_dir, "embedding.pt"),
            map_location="cpu", weights_only=True,
        )

    assert saved.keys() == embed.state_dict().keys()
    for key, value in embed.state_dict().items():
        torch.testing.assert_close(saved[key], value, rtol=0, atol=0)


def test_independent_lowrank_periodic_checkpoint_saves_both_sidecars():
    from train_compositional import SaveEmbeddingCallback

    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    shim = EmbeddingShim(embed)
    head = IndependentLowRankHead(embed)
    callback = SaveEmbeddingCallback(
        shim, independent_output_head=head
    )

    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        os.makedirs(checkpoint_dir)
        callback.on_save(
            SimpleNamespace(output_dir=output_dir, should_save=True),
            SimpleNamespace(global_step=7),
            control=None,
        )
        saved_embed = torch.load(
            os.path.join(checkpoint_dir, "embedding.pt"),
            map_location="cpu", weights_only=True,
        )
        saved_head = torch.load(
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
            map_location="cpu", weights_only=True,
        )

    assert saved_embed.keys() == embed.state_dict().keys()
    assert saved_head.keys() == head.state_dict().keys()
    for key, value in head.state_dict().items():
        torch.testing.assert_close(saved_head[key], value, rtol=0, atol=0)


def test_shared_local_loads_periodic_checkpoint_with_parent_config():
    torch.manual_seed(14)
    model = _tiny_qwen()
    embed = SharedLocalEmbed(
        VOCAB_SIZE, HIDDEN_SIZE,
        shared_rank=BASE_DIM // 2,
        local_rank=BASE_DIM // 2,
        num_groups=4,
    )
    _install_tied_embedding(model, embed, "shared_local")
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    comp_config = {
        "arm": "shared_local",
        "shared_rank": BASE_DIM // 2,
        "local_embed_rank": BASE_DIM // 2,
        "num_groups": 4,
        "tie_output": True,
    }
    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        with open(os.path.join(output_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": comp_config}, handle)

        loaded, loaded_config = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits

    assert loaded_config == comp_config
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_independent_lowrank_loads_periodic_checkpoint_with_parent_config():
    torch.manual_seed(25)
    model = _tiny_qwen()
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    head = _install_independent_lowrank(model, embed)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    comp_config = {
        "arm": "lowrank",
        "d_x": BASE_DIM,
        "tie_output": False,
        "independent_lowrank_output": True,
    }
    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        torch.save(
            head.state_dict(),
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )
        with open(os.path.join(output_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": comp_config}, handle)

        loaded, loaded_config = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits

    assert loaded_config == comp_config
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_independent_loader_rejects_missing_or_stale_sidecar():
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    model = _tiny_qwen()
    head = _install_independent_lowrank(model, embed)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump(
                {
                    "compositional": {
                        "arm": "lowrank",
                        "d_x": BASE_DIM,
                        "tie_output": False,
                        "independent_lowrank_output": True,
                    }
                },
                handle,
            )
        try:
            load_compositional_model(
                checkpoint_dir, device="cpu", dtype=torch.float32
            )
        except FileNotFoundError as error:
            assert INDEPENDENT_OUTPUT_FILENAME in str(error)
        else:
            raise AssertionError("missing output sidecar should fail loudly")

        torch.save(
            head.state_dict(),
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )
        mismatched_head_state = {
            key: value.detach().clone()
            for key, value in head.state_dict().items()
        }
        mismatched_head_state["X"].add_(0.5)
        torch.save(
            mismatched_head_state,
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )
        try:
            load_compositional_model(
                checkpoint_dir, device="cpu", dtype=torch.float32
            )
        except ValueError as error:
            assert "does not match" in str(error)
        else:
            raise AssertionError(
                "loader should reject same-shaped mixed output sidecars"
            )

        torch.save(
            head.state_dict(),
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump(
                {
                    "compositional": {
                        "arm": "lowrank",
                        "d_x": BASE_DIM,
                        "tie_output": False,
                        "independent_lowrank_output": False,
                    }
                },
                handle,
            )
        try:
            load_compositional_model(
                checkpoint_dir, device="cpu", dtype=torch.float32
            )
        except ValueError as error:
            assert INDEPENDENT_OUTPUT_FILENAME in str(error)
        else:
            raise AssertionError("stale output sidecar should fail loudly")


def test_damaged_compositional_checkpoint_cannot_fall_back_to_stock():
    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        os.makedirs(checkpoint_dir)
        with open(os.path.join(output_dir, "train_config.json"), "w") as handle:
            json.dump(
                {
                    "compositional": {
                        "arm": "lowrank",
                        "d_x": BASE_DIM,
                        "tie_output": False,
                    }
                },
                handle,
            )
        assert is_compositional(checkpoint_dir)
        try:
            load_compositional_model(
                checkpoint_dir, device="cpu", dtype=torch.float32
            )
        except FileNotFoundError as error:
            assert "embedding.pt" in str(error)
        else:
            raise AssertionError(
                "damaged compositional checkpoint should not load as stock"
            )

    # Even if config and both sidecars are lost, the registered custom tensor
    # names in the HF state must prevent silent stock-model fallback.
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model = _tiny_qwen()
        embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
        _install_independent_lowrank(model, embed)
        model.save_pretrained(checkpoint_dir)
        assert is_compositional(checkpoint_dir)
        try:
            load_compositional_model(
                checkpoint_dir, device="cpu", dtype=torch.float32
            )
        except FileNotFoundError as error:
            assert "embedding.pt" in str(error)
        else:
            raise AssertionError(
                "custom HF tensors must not fall back to random stock weights"
            )


def test_checkpoint_tensor_reader_matches_hf_file_precedence():
    """A stale index must not override the single safetensors HF will load."""
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        tensor_name = "model.embed_tokens.embed.X"
        expected = torch.ones(2, 3)
        save_file(
            {tensor_name: expected},
            os.path.join(checkpoint_dir, "model.safetensors"),
        )
        save_file(
            {tensor_name: torch.zeros_like(expected)},
            os.path.join(checkpoint_dir, "stale-shard.safetensors"),
        )
        with open(os.path.join(
            checkpoint_dir, "model.safetensors.index.json"
        ), "w") as handle:
            json.dump({
                "metadata": {},
                "weight_map": {tensor_name: "stale-shard.safetensors"},
            }, handle)

        loaded = _load_checkpoint_tensors(checkpoint_dir, {tensor_name})

    torch.testing.assert_close(
        loaded[tensor_name], expected, rtol=0, atol=0
    )


def test_loader_default_dtype_matches_bfloat16_checkpoint():
    """dtype=None must follow the checkpoint dtype for separately saved weights."""
    torch.manual_seed(13)
    model = _tiny_qwen().to(torch.bfloat16)
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM).to(torch.bfloat16)
    _install_tied_embedding(model, embed, "lowrank")

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt"))
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump(
                {
                    "compositional": {
                        "arm": "lowrank",
                        "d_x": BASE_DIM,
                        "tie_output": True,
                    }
                },
                handle,
            )

        loaded, _ = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=None
        )
        with torch.no_grad():
            logits = loaded(input_ids=torch.randint(0, VOCAB_SIZE, (2, 7))).logits

    assert next(loaded.model.layers.parameters()).dtype == torch.bfloat16
    assert loaded.model.embed_tokens.embed.X.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16


def test_pure_local_loader_and_logits_match_bfloat16_checkpoint():
    torch.manual_seed(24)
    model = _tiny_qwen().to(torch.bfloat16)
    embed = PureLocalEmbed(
        VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM, num_groups=4
    ).to(torch.bfloat16)
    _install_tied_embedding(model, embed, "pure_local")

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump(
                {
                    "compositional": {
                        "arm": "pure_local",
                        "pure_local_rank": BASE_DIM,
                        "num_groups": 4,
                        "tie_output": True,
                    }
                },
                handle,
            )

        loaded, _ = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=None
        )
        with torch.no_grad():
            logits = loaded(
                input_ids=torch.randint(0, VOCAB_SIZE, (2, 7))
            ).logits

    loaded_embed = loaded.model.embed_tokens.embed
    assert loaded_embed.token_factors.dtype == torch.bfloat16
    assert loaded_embed.local_weight.dtype == torch.bfloat16
    assert loaded_embed.bias.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


def test_independent_loader_default_dtype_matches_bfloat16_checkpoint():
    torch.manual_seed(23)
    model = _tiny_qwen().to(torch.bfloat16)
    embed = LowRankEmbed(
        VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM
    ).to(torch.bfloat16)
    head = _install_independent_lowrank(model, embed)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        torch.save(
            head.state_dict(),
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )
        with open(os.path.join(checkpoint_dir, "train_config.json"), "w") as handle:
            json.dump(
                {
                    "compositional": {
                        "arm": "lowrank",
                        "d_x": BASE_DIM,
                        "tie_output": False,
                        "independent_lowrank_output": True,
                    }
                },
                handle,
            )

        loaded, _ = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=None
        )
        with torch.no_grad():
            logits = loaded(
                input_ids=torch.randint(0, VOCAB_SIZE, (2, 7))
            ).logits

    assert loaded.model.embed_tokens.embed.X.dtype == torch.bfloat16
    assert loaded.lm_head.X.dtype == torch.bfloat16
    assert loaded.lm_head.proj_weight.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16


def test_loader_infers_tied_output_when_train_config_is_missing():
    """Interrupted legacy runs can recover tying from the absent lm_head weights."""
    torch.manual_seed(17)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model = _tiny_qwen()
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    _install_tied_embedding(model, embed, "lowrank")
    model.eval()
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

    assert inferred == {"arm": "lowrank", "d_x": BASE_DIM, "tie_output": True}
    assert loaded.lm_head.embed is loaded.model.embed_tokens.embed
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_loader_infers_independent_output_when_train_config_is_missing():
    torch.manual_seed(24)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model = _tiny_qwen()
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    head = _install_independent_lowrank(model, embed)
    with torch.no_grad():
        head.X.mul_(1.1)
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        torch.save(
            head.state_dict(),
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )

        loaded, inferred = load_compositional_model(
            checkpoint_dir, device="cpu", dtype=torch.float32
        )
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits

    assert inferred == {
        "arm": "lowrank",
        "d_x": BASE_DIM,
        "tie_output": False,
        "independent_lowrank_output": True,
        "output_rank": BASE_DIM,
    }
    assert isinstance(loaded.lm_head, IndependentLowRankHead)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_loader_rejects_configless_independent_head_without_sidecar():
    model = _tiny_qwen()
    embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
    _install_independent_lowrank(model, embed)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        try:
            load_compositional_model(
                checkpoint_dir, device="cpu", dtype=torch.float32
            )
        except FileNotFoundError as error:
            assert INDEPENDENT_OUTPUT_FILENAME in str(error)
        else:
            raise AssertionError(
                "configless independent head must not be inferred as tied"
            )


def test_loader_infers_shared_local_shape_when_train_config_is_missing():
    torch.manual_seed(19)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model = _tiny_qwen()
    embed = SharedLocalEmbed(
        VOCAB_SIZE,
        HIDDEN_SIZE,
        shared_rank=BASE_DIM // 2,
        local_rank=BASE_DIM // 2,
        num_groups=4,
    )
    _install_tied_embedding(model, embed, "shared_local")
    model.eval()
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

    assert inferred == {
        "arm": "shared_local",
        "shared_rank": BASE_DIM // 2,
        "local_embed_rank": BASE_DIM // 2,
        "num_groups": 4,
        "tie_output": True,
    }
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_loader_infers_pure_local_shape_when_train_config_is_missing():
    torch.manual_seed(20)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    model = _tiny_qwen()
    embed = PureLocalEmbed(
        VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM, num_groups=4
    )
    _install_tied_embedding(model, embed, "pure_local")
    model.eval()
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

    assert inferred == {
        "arm": "pure_local",
        "pure_local_rank": BASE_DIM,
        "num_groups": 4,
        "tie_output": True,
    }
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_local_cli_rank_does_not_collide_with_distributed_local_rank():
    from train_compositional import CompositionalArguments

    parser = HfArgumentParser((TrainingArguments, CompositionalArguments))
    training_args, comp_args = parser.parse_args_into_dataclasses(args=[
        "--output_dir", "/tmp/shared-local-parser-test",
        "--local_rank", "2",
        "--arm", "shared_local",
        "--local_embed_rank", "7",
    ])
    assert training_args.local_rank == 2
    assert comp_args.local_embed_rank == 7


def test_pure_local_cli_build_and_output_validation():
    from train_compositional import (
        CompositionalArguments,
        build_arm,
        validate_output_configuration,
    )

    args = CompositionalArguments(
        arm="pure_local",
        pure_local_rank=BASE_DIM,
        num_groups=4,
        tie_output=True,
    )
    embed = build_arm(args, VOCAB_SIZE, HIDDEN_SIZE)
    assert isinstance(embed, PureLocalEmbed)
    assert embed.rank == BASE_DIM
    assert embed.num_groups == 4
    assert validate_output_configuration(args) == "tied"

    args.tie_output = False
    try:
        validate_output_configuration(args)
    except ValueError as error:
        assert "requires --tie_output" in str(error)
    else:
        raise AssertionError("pure_local with a dense output must be rejected")


def test_independent_output_cli_and_validation():
    from train_compositional import (
        CompositionalArguments,
        validate_output_configuration,
        validate_resume_compatibility,
    )

    parser = HfArgumentParser((TrainingArguments, CompositionalArguments))
    _, comp_args = parser.parse_args_into_dataclasses(args=[
        "--output_dir", "/tmp/independent-output-parser-test",
        "--arm", "lowrank",
        "--d_x", str(BASE_DIM),
        "--independent_lowrank_output",
    ])
    assert validate_output_configuration(comp_args) == "independent_lowrank"

    comp_args.tie_output = True
    try:
        validate_output_configuration(comp_args)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("tied and independent output flags should conflict")

    comp_args.tie_output = False
    comp_args.arm = "shared_local"
    try:
        validate_output_configuration(comp_args)
    except ValueError as error:
        assert "requires --arm lowrank" in str(error)
    else:
        raise AssertionError("independent output should reject non-lowrank input")

    resume_args = CompositionalArguments(
        arm="lowrank", d_x=BASE_DIM, independent_lowrank_output=True
    )
    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        model = _tiny_qwen()
        embed = LowRankEmbed(VOCAB_SIZE, HIDDEN_SIZE, rank=BASE_DIM)
        head = _install_independent_lowrank(model, embed)
        model.save_pretrained(checkpoint_dir)
        torch.save(
            embed.state_dict(), os.path.join(checkpoint_dir, "embedding.pt")
        )
        torch.save(
            head.state_dict(),
            os.path.join(checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME),
        )
        _write_minimal_resume_state(checkpoint_dir, 7)
        with open(os.path.join(output_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": asdict(resume_args)}, handle)
        validate_resume_compatibility(
            checkpoint_dir,
            resume_args,
            expected_embed_module=embed,
            expected_output_head=head,
        )

        os.remove(os.path.join(checkpoint_dir, "rng_state.pth"))
        for rank in range(8):
            torch.save({"cpu": torch.random.get_rng_state()}, os.path.join(
                checkpoint_dir, f"rng_state_{rank}.pth"
            ))
        distributed_args = SimpleNamespace(world_size=8)
        validate_resume_compatibility(
            checkpoint_dir, resume_args, distributed_args
        )
        try:
            validate_resume_compatibility(
                checkpoint_dir, resume_args, SimpleNamespace(world_size=4)
            )
        except ValueError as error:
            assert "world size 4" in str(error)
        else:
            raise AssertionError(
                "resume should reject a changed distributed world size"
            )
        os.remove(os.path.join(checkpoint_dir, "rng_state_7.pth"))
        try:
            validate_resume_compatibility(
                checkpoint_dir, resume_args, distributed_args
            )
        except ValueError as error:
            assert "rng_state_7.pth" in str(error)
        else:
            raise AssertionError(
                "distributed resume should require every rank's RNG state"
            )
        torch.save({"cpu": torch.random.get_rng_state()}, os.path.join(
            checkpoint_dir, "rng_state_7.pth"
        ))
        for rank in range(8):
            os.remove(os.path.join(
                checkpoint_dir, f"rng_state_{rank}.pth"
            ))
        torch.save({"cpu": torch.random.get_rng_state()}, os.path.join(
            checkpoint_dir, "rng_state.pth"
        ))

        os.remove(os.path.join(checkpoint_dir, "optimizer.pt"))
        try:
            validate_resume_compatibility(checkpoint_dir, resume_args)
        except FileNotFoundError as error:
            assert "optimizer.pt" in str(error)
        else:
            raise AssertionError(
                "resume should reject a missing optimizer state"
            )
        torch.save({"state": {}, "param_groups": []}, os.path.join(
            checkpoint_dir, "optimizer.pt"
        ))

        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as handle:
            json.dump({"global_step": 6}, handle)
        try:
            validate_resume_compatibility(checkpoint_dir, resume_args)
        except ValueError as error:
            assert "step mismatch" in str(error)
        else:
            raise AssertionError(
                "resume should reject a mismatched Trainer global step"
            )
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as handle:
            json.dump({"global_step": 7}, handle)

        mismatched = copy.deepcopy(resume_args)
        mismatched.d_x += 1
        try:
            validate_resume_compatibility(checkpoint_dir, mismatched)
        except ValueError as error:
            assert "d_x" in str(error)
        else:
            raise AssertionError("resume should reject mismatched factor rank")

        # Sidecars alone are insufficient: Trainer restores the registered
        # head from the HF model state, so a partial state must fail before
        # training can continue with freshly initialized output factors.
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        valid_state = load_file(model_path)
        output_path = os.path.join(
            checkpoint_dir, INDEPENDENT_OUTPUT_FILENAME
        )
        valid_output_state = torch.load(
            output_path, map_location="cpu", weights_only=True
        )
        torch.save({"X": valid_output_state["X"]}, output_path)
        save_file(
            {
                key: value
                for key, value in valid_state.items()
                if key != "lm_head.proj_weight"
            },
            model_path,
        )
        try:
            validate_resume_compatibility(
                checkpoint_dir,
                resume_args,
                expected_embed_module=embed,
                expected_output_head=head,
            )
        except ValueError as error:
            assert "wrong parameter schema" in str(error)
        else:
            raise AssertionError(
                "resume should reject matching partial HF/sidecar states"
            )
        torch.save(valid_output_state, output_path)
        save_file(valid_state, model_path)

        save_file(
            {
                key: value
                for key, value in valid_state.items()
                if not key.startswith("lm_head.")
            },
            model_path,
        )
        try:
            validate_resume_compatibility(
                checkpoint_dir,
                resume_args,
                expected_embed_module=embed,
                expected_output_head=head,
            )
        except ValueError as error:
            assert "output-head topology" in str(error)
        else:
            raise AssertionError(
                "resume should reject an incomplete HF output-head state"
            )

    legacy_dense = CompositionalArguments(arm="lowrank", d_x=BASE_DIM)
    requested_tied = copy.deepcopy(legacy_dense)
    requested_tied.tie_output = True
    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_dir = os.path.join(output_dir, "checkpoint-7")
        os.makedirs(checkpoint_dir)
        torch.save({}, os.path.join(checkpoint_dir, "embedding.pt"))
        _write_minimal_resume_state(checkpoint_dir, 7)
        saved_legacy = asdict(legacy_dense)
        saved_legacy.pop("tie_output")
        saved_legacy.pop("independent_lowrank_output")
        with open(os.path.join(output_dir, "train_config.json"), "w") as handle:
            json.dump({"compositional": saved_legacy}, handle)
        try:
            validate_resume_compatibility(checkpoint_dir, requested_tied)
        except ValueError as error:
            assert "tie_output" in str(error)
        else:
            raise AssertionError(
                "legacy dense checkpoint must not resume as tied output"
            )


def test_context_dependent_arms_are_rejected():
    embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    for arm in ("v0", "v1", "v2", "isolation_control"):
        try:
            make_tied_head(embed, arm, VOCAB_SIZE)
        except ValueError as error:
            assert arm in str(error)
        else:
            raise AssertionError(f"{arm}: expected make_tied_head to reject the arm")
