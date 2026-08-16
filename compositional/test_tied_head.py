"""Regression tests for tied compositional output heads.

These tests cover the algebra, gradient flow, serialization ownership, and a
complete save/load round trip for every supported tied-output arm.
"""

import copy
import json
import os
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file
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
)
from .loading import EmbeddingShim, load_compositional_model
from .tied_head import make_tied_head


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


def test_context_dependent_arms_are_rejected():
    embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    for arm in ("v0", "v1", "v2", "isolation_control"):
        try:
            make_tied_head(embed, arm, VOCAB_SIZE)
        except ValueError as error:
            assert arm in str(error)
        else:
            raise AssertionError(f"{arm}: expected make_tied_head to reject the arm")
