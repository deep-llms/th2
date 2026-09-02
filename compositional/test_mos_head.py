"""Independent algebra, gradient, serialization, and budget tests for BT-MoS."""

import copy
from dataclasses import asdict
import json
import os

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from .compressed_baselines import GroupReduceEmbed
from .loading import EmbeddingShim, load_compositional_model
from .mos_head import MixtureOfSoftmaxesHead
from .tied_head import TiedGroupReduceHead, make_tied_head


def _embed(vocab=19, dim=8, dtype=torch.float64):
    group_ids = torch.arange(vocab) % 3
    return GroupReduceEmbed(
        vocab, dim, group_ranks=(5, 4, 3), group_ids=group_ids
    ).to(dtype=dtype)


def _head(embed, components=3, context_rank=4, chunk_size=3):
    return make_tied_head(
        embed,
        "groupreduce",
        embed.vocab_size,
        mos_components=components,
        mos_context_rank=context_rank,
        mos_chunk_size=chunk_size,
    ).to(dtype=next(embed.parameters()).dtype)


def _naive_log_probs(head, hidden):
    ids = torch.arange(head.embed.vocab_size, device=hidden.device)
    table, _ = head.embed(ids)
    if head.num_components == 1:
        contexts = hidden.unsqueeze(0)
        log_prior = hidden.new_zeros((1, hidden.shape[0], 1)).float()
    else:
        down = torch.einsum("nd,kcd->knc", hidden, head.context_down)
        extra = torch.einsum("knc,kdc->knd", down, head.context_up)
        extra = torch.tanh(extra + head.context_bias[:, None, :])
        contexts = torch.cat((hidden.unsqueeze(0), extra), dim=0)
        log_prior = F.log_softmax(head.prior(hidden).float(), -1).T.unsqueeze(-1)
    logits = torch.einsum("knd,vd->knv", contexts, table).float()
    return torch.logsumexp(F.log_softmax(logits, -1) + log_prior, dim=0)


def test_k1_is_exact_normalized_control_with_no_added_parameters():
    embed = _embed()
    inner = TiedGroupReduceHead(embed)
    head = MixtureOfSoftmaxesHead(inner, 8, 1, 4, 2).double()
    hidden = torch.randn(5, 8, dtype=torch.float64)
    expected = F.log_softmax(inner(hidden).float(), dim=-1)
    torch.testing.assert_close(head(hidden), expected, rtol=0, atol=0)
    assert sum(p.numel() for p in head.parameters()) == 0
    assert list(head.state_dict()) == []


def test_matches_independent_full_table_reference_and_is_normalized():
    torch.manual_seed(4)
    embed = _embed()
    head = _head(embed)
    hidden = torch.randn(7, 8, dtype=torch.float64)
    actual = head(hidden)
    reference = _naive_log_probs(head, hidden)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=2e-6)
    torch.testing.assert_close(
        actual.exp().sum(-1), torch.ones(7), rtol=1e-6, atol=2e-6
    )


def _run_with_chunk(base_embed, base_head, hidden, upstream, chunk_size):
    embed = copy.deepcopy(base_embed)
    head = _head(embed, chunk_size=chunk_size)
    head.load_state_dict(base_head.state_dict(), strict=True)
    h = hidden.detach().clone().requires_grad_(True)
    (head(h) * upstream).sum().backward()
    gradients = {"hidden": h.grad.detach().clone()}
    gradients.update({
        "embed." + name: parameter.grad.detach().clone()
        for name, parameter in embed.named_parameters()
    })
    gradients.update({
        "head." + name: parameter.grad.detach().clone()
        for name, parameter in head.named_parameters()
    })
    return head(h.detach()), gradients


def test_chunk_sizes_preserve_outputs_and_all_gradients():
    torch.manual_seed(7)
    embed = _embed()
    head = _head(embed)
    hidden = torch.randn(7, 8, dtype=torch.float64)
    upstream = torch.randn(7, 19)
    reference_output, reference_gradients = _run_with_chunk(
        embed, head, hidden, upstream, 7
    )
    for size in (1, 3, 14):
        output, gradients = _run_with_chunk(
            embed, head, hidden, upstream, size
        )
        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
        assert gradients.keys() == reference_gradients.keys()
        for name in gradients:
            torch.testing.assert_close(
                gradients[name], reference_gradients[name],
                rtol=2e-5, atol=2e-6, msg=name,
            )


def test_fast_gradients_match_naive_reference():
    torch.manual_seed(9)
    embed = _embed(vocab=11, dim=6)
    head = make_tied_head(
        embed, "groupreduce", 11,
        mos_components=3, mos_context_rank=3, mos_chunk_size=2,
    ).double()
    hidden = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(4, 11)
    (head(hidden) * upstream).sum().backward()
    fast = {"hidden": hidden.grad.detach().clone()}
    fast.update({"embed." + n: p.grad.detach().clone() for n, p in embed.named_parameters()})
    fast.update({"head." + n: p.grad.detach().clone() for n, p in head.named_parameters()})
    embed.zero_grad(); head.zero_grad(); hidden.grad = None
    (_naive_log_probs(head, hidden) * upstream).sum().backward()
    reference = {"hidden": hidden.grad}
    reference.update({"embed." + n: p.grad for n, p in embed.named_parameters()})
    reference.update({"head." + n: p.grad for n, p in head.named_parameters()})
    for name in fast:
        torch.testing.assert_close(
            fast[name], reference[name], rtol=2e-5, atol=2e-6, msg=name
        )


def test_checkpointing_does_not_save_component_vocabulary_slab():
    embed = _embed(vocab=23, dim=8, dtype=torch.float32)
    head = _head(embed, chunk_size=4)
    hidden = torch.randn(9, 8, requires_grad=True)
    saved_shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved_shapes.append(tuple(tensor.shape)) or tensor,
        lambda tensor: tensor,
    ):
        head(hidden).sum().backward()
    assert (3, 4, 23) not in saved_shapes
    assert not any(
        len(shape) == 3 and shape[0] == 3 and shape[-1] == 23
        for shape in saved_shapes
    )


def test_every_parameter_is_connected_and_metrics_are_well_formed():
    embed = _embed(dtype=torch.float32)
    head = _head(embed, chunk_size=8)
    head(torch.randn(1, 8, requires_grad=True)).sum().backward()
    assert all(p.grad is not None for p in embed.parameters())
    assert all(p.grad is not None for p in head.parameters())
    metrics = head.pop_step_metrics()
    usages = torch.stack([
        metrics[f"mos_prior_usage_{index}"][0]
        for index in range(3)
    ])
    count = metrics["mos_prior_usage_0"][1]
    torch.testing.assert_close(usages.sum(), count)
    assert metrics["mos_prior_entropy"][1] == count
    assert head.pop_step_metrics() is None


def test_trainer_collects_output_head_metrics(tmp_path):
    from transformers import TrainingArguments
    from train_compositional import CompositionalArguments, CompositionalTrainer

    model = _tiny_qwen()
    embed = _embed(dtype=torch.float32)
    shim = EmbeddingShim(embed)
    # The training shim records _last_theta; evaluation's loading shim does not.
    shim._last_theta = None
    model.model.embed_tokens = shim
    model.lm_head = _head(embed, chunk_size=4)
    trainer = CompositionalTrainer(
        model=model,
        args=TrainingArguments(output_dir=tmp_path, report_to=[]),
        embed_shim=shim,
        comp_args=CompositionalArguments(
            arm="groupreduce", tie_output=True, mos_components=3,
        ),
    )
    ids = torch.randint(0, 19, (1, 4))
    loss = trainer.compute_loss(model, {"input_ids": ids, "labels": ids})
    assert torch.isfinite(loss)
    assert set(trainer._device_metric_sums) == {
        "mos_prior_usage_0", "mos_prior_usage_1", "mos_prior_usage_2",
        "mos_prior_entropy",
    }


def _tiny_qwen(vocab=19, dim=8):
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=vocab,
        hidden_size=dim,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    ))


@pytest.mark.parametrize("with_train_config", [True, False])
def test_save_load_roundtrip_with_and_without_train_config(tmp_path, with_train_config):
    torch.manual_seed(12)
    model = _tiny_qwen()
    embed = _embed(dtype=torch.float32)
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = _head(embed, chunk_size=2)
    model.eval()
    input_ids = torch.randint(0, 19, (2, 5))
    with torch.no_grad():
        before = model(input_ids=input_ids).logits
    model.save_pretrained(tmp_path)
    torch.save(embed.state_dict(), tmp_path / "embedding.pt")
    if with_train_config:
        with open(tmp_path / "train_config.json", "w") as handle:
            json.dump({"compositional": {
                "arm": "groupreduce", "groupreduce_num_groups": 3,
                "groupreduce_ranks": "5,4,3", "tie_output": True,
                "independent_lowrank_output": False, "mos_components": 3,
                "mos_context_rank": 4, "mos_chunk_size": 2,
            }}, handle)
    loaded, config = load_compositional_model(
        str(tmp_path), device="cpu", dtype=torch.float32
    )
    with torch.no_grad():
        after = loaded(input_ids=input_ids).logits
    assert config["mos_components"] == 3
    assert config["mos_context_rank"] == 4
    assert loaded.lm_head.embed is loaded.model.embed_tokens.embed
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_reference_parameter_budget():
    vocab, dim = 151_936, 1024
    populations = (2048, 6144, 24576, 119168)
    group_ids = torch.repeat_interleave(
        torch.arange(4), torch.tensor(populations)
    )
    embed = GroupReduceEmbed(
        vocab, dim, (1024, 352, 192, 64), group_ids=group_ids
    )
    head = make_tied_head(
        embed, "groupreduce", vocab,
        mos_components=3, mos_context_rank=256, mos_chunk_size=2048,
    )
    count = sum(p.numel() for p in embed.parameters())
    count += sum(p.numel() for p in head.parameters())
    assert count == 19_330_051


def test_runner_entries_are_last_and_require_importance_artifact(tmp_path, monkeypatch):
    import run_experiments as runner

    names = [entry["name"] for entry in runner.EXPERIMENT_COMMANDS]
    assert names[-3:] == [
        "groupreduce_matched_lb_t4",
        "btmos_k3_c256_lb",
        "btmos_k3_c256_lb_r512",
    ]
    artifact = tmp_path / "importance.npz"
    artifact.write_bytes(b"nonempty")
    monkeypatch.setenv("BTMOS_IMPORTANCE_PATH", str(artifact))
    for entry in runner.EXPERIMENT_COMMANDS[-3:]:
        assert runner.missing_input_files(entry) == []
        assert any(
            "BTMOS_IMPORTANCE_PATH" in spec
            for spec in entry["required_input_files"]
        )


def _write_resume_files(checkpoint_dir, step):
    with open(checkpoint_dir / "trainer_state.json", "w") as handle:
        json.dump({"global_step": step}, handle)
    torch.save({}, checkpoint_dir / "optimizer.pt")
    torch.save({}, checkpoint_dir / "scheduler.pt")
    torch.save({}, checkpoint_dir / "rng_state.pth")


@pytest.mark.parametrize("saved_components,requested_components", [(3, 1), (1, 3)])
def test_resume_rejects_mos_topology_change(
    tmp_path, saved_components, requested_components
):
    from train_compositional import (
        CompositionalArguments, validate_resume_compatibility,
    )

    output_dir = tmp_path / "run"
    checkpoint_dir = output_dir / "checkpoint-3"
    checkpoint_dir.mkdir(parents=True)
    model = _tiny_qwen()
    embed = _embed(dtype=torch.float32)
    model.model.embed_tokens = EmbeddingShim(embed)
    head = _head(embed, components=saved_components, chunk_size=2)
    model.lm_head = head
    model.save_pretrained(checkpoint_dir)
    torch.save(embed.state_dict(), checkpoint_dir / "embedding.pt")
    _write_resume_files(checkpoint_dir, 3)
    saved_args = CompositionalArguments(
        arm="groupreduce", tie_output=True,
        groupreduce_num_groups=3, groupreduce_ranks="5,4,3",
        mos_components=saved_components, mos_context_rank=4,
        mos_chunk_size=2,
    )
    with open(output_dir / "train_config.json", "w") as handle:
        json.dump({"compositional": asdict(saved_args)}, handle)
    validate_resume_compatibility(
        str(checkpoint_dir), saved_args,
        expected_embed_module=embed,
        expected_output_head=head,
    )
    requested_args = copy.deepcopy(saved_args)
    requested_args.mos_components = requested_components
    requested_head = _head(
        embed, components=requested_components, chunk_size=2
    )
    with pytest.raises(ValueError, match="config mismatch|output-head topology"):
        validate_resume_compatibility(
            str(checkpoint_dir), requested_args,
            expected_embed_module=embed,
            expected_output_head=requested_head,
        )
