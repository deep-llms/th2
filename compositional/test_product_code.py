"""Correctness, structure, and checkpoint tests for ProductCodeEmbed."""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from .compression_init import frequency_rank_order
from .loading import EmbeddingShim, load_compositional_model
from .product_code import (
    ProductCodeEmbed,
    hashed_codes,
    head_tail_partition,
    pq_codes,
    product_code_parameter_count,
)
from .test_tied_head import HIDDEN_SIZE, VOCAB_SIZE, _tiny_qwen
from .tied_head import make_tied_head


def _importance(vocab_size, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(vocab_size, generator=generator, dtype=torch.float64)


def test_tied_logits_are_exact_in_float64_with_nonunit_gates():
    embed = ProductCodeEmbed(37, 8, 5, 3, 6, importance=_importance(37))
    with torch.no_grad():
        embed.gate_offsets.uniform_(-0.7, 0.7)
    embed = embed.double()
    hidden = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    actual = make_tied_head(embed, "product_code", 37)(hidden)
    expected = hidden @ embed.materialize().T
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    actual.square().sum().backward()
    for name, parameter in embed.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"


def test_parameter_budget_reference_and_irregular():
    assert product_code_parameter_count(151_936, 1_024, 2_048, 4, 4_096) == 19_474_944
    assert product_code_parameter_count(151_936, 1_024, 8_192, 4, 2_560) == 19_450_368
    embed = ProductCodeEmbed(29, 6, 4, 2, 7, importance=_importance(29))
    manual = 4 * 6 + 2 * 7 * 6 + 25 * 2 + 6
    assert embed.parameter_count == manual
    assert sum(p.numel() for p in embed.parameters()) == manual


def test_hashed_codes_are_deterministic_unique_and_in_range():
    tail = torch.arange(3, 60)
    a = hashed_codes(tail, 3, 5, seed=0)
    b = hashed_codes(tail, 3, 5, seed=0)
    c = hashed_codes(tail, 3, 5, seed=1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (57, 3)
    assert torch.all(a >= 0) and torch.all(a < 5)
    assert torch.unique(a, dim=0).size(0) == 57
    # Saturated space: 3^3 = 27 signatures for 27 tokens must still be unique.
    full = hashed_codes(torch.arange(27), 3, 3)
    assert torch.unique(full, dim=0).size(0) == 27
    with pytest.raises(ValueError, match="cannot cover"):
        hashed_codes(torch.arange(200), 2, 4)


def test_pq_codes_cover_tail_respect_capacity_and_are_unique():
    table = torch.randn(300, 16, generator=torch.Generator().manual_seed(3))
    table[100:110] = table[100]  # near-duplicate rows must still get unique codes
    codes = pq_codes(table, 4, 6, iters=5, seed=1, capacity_factor=2.0)
    assert codes.shape == (300, 4)
    assert torch.all(codes >= 0) and torch.all(codes < 6)
    assert torch.unique(codes, dim=0).size(0) == 300
    capacity = int(2.0 * 300 / 6)
    for index in range(4):
        assert torch.bincount(codes[:, index], minlength=6).max() <= capacity


def test_every_parameter_stays_in_autograd_for_head_only_and_tail_only_batches():
    embed = ProductCodeEmbed(20, 6, 4, 2, 5, importance=_importance(20))
    head_only = embed.head_ids[:2].view(1, 2)
    tail_only = embed.tail_ids[:3].view(1, 3)
    for batch in (head_only, tail_only):
        embed.zero_grad()
        out, _ = embed(batch)
        out.sum().backward()
        for name, parameter in embed.named_parameters():
            assert parameter.grad is not None, f"{name} unused for batch {batch.tolist()}"


def test_partition_helper_matches_module_and_scripts_contract():
    importance = _importance(50, seed=7)
    head_ids, tail_ids = head_tail_partition(importance, 9)
    embed = ProductCodeEmbed(50, 4, 9, 2, 8, importance=importance)
    assert torch.equal(head_ids, embed.head_ids)
    assert torch.equal(tail_ids, embed.tail_ids)
    assert torch.equal(torch.sort(torch.cat([head_ids, tail_ids])).values, torch.arange(50))


def test_head_selection_is_stable_by_importance_then_id():
    importance = torch.tensor([5.0, 9.0, 9.0, 1.0, 9.0, 0.0, 2.0], dtype=torch.float64)
    embed = ProductCodeEmbed(7, 4, 3, 2, 3, importance=importance)
    # Ties at 9.0 are resolved by ascending token id: 1, 2, 4 form the head.
    assert embed.head_ids.tolist() == [1, 2, 4]
    assert embed.tail_ids.tolist() == [0, 3, 5, 6]
    assert torch.equal(frequency_rank_order(importance)[:3].sort().values, embed.head_ids)
    torch.manual_seed(0)
    first = ProductCodeEmbed(7, 4, 3, 2, 3, importance=importance)
    torch.manual_seed(0)
    second = ProductCodeEmbed(7, 4, 3, 2, 3, importance=importance.flip(0))
    # Structure consumes no RNG: parameters agree even though the head differs.
    for (_, p1), (_, p2) in zip(first.named_parameters(), second.named_parameters()):
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)
    assert not torch.equal(first.head_ids, second.head_ids)


def test_corrupt_codes_are_rejected_on_load():
    embed = ProductCodeEmbed(12, 4, 2, 2, 4, importance=_importance(12))
    state = {k: v.clone() for k, v in embed.state_dict().items()}
    state["codes"][1] = state["codes"][0]
    rebuilt = ProductCodeEmbed(
        12, 4, 2, 2, 4, head_ids=embed.head_ids, codes=embed.codes,
        assignment="checkpoint",
    )
    with pytest.raises(ValueError, match="unique"):
        rebuilt.load_state_dict(state, strict=True)


def test_configless_checkpoint_roundtrip_is_strict():
    torch.manual_seed(5)
    embed = ProductCodeEmbed(
        VOCAB_SIZE, HIDDEN_SIZE, 6, 3, 5, importance=_importance(VOCAB_SIZE)
    )
    with torch.no_grad():
        embed.gate_offsets.uniform_(-0.5, 0.5)
    model = _tiny_qwen()
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "product_code", VOCAB_SIZE)
    model.eval()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 6))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
    config = {
        "arm": "product_code",
        "product_code_head_size": 6,
        "product_code_num_hashes": 3,
        "product_code_num_buckets": 5,
        "product_code_assignment": "hashed",
        "tie_output": True,
    }
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        torch.save(embed.state_dict(), os.path.join(directory, "embedding.pt"))
        with open(os.path.join(directory, "train_config.json"), "w") as handle:
            json.dump({"compositional": config}, handle)
        loaded, loaded_config = load_compositional_model(directory, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            torch.testing.assert_close(loaded(input_ids=input_ids).logits, expected, rtol=0, atol=0)
        assert loaded_config == config
        os.remove(os.path.join(directory, "train_config.json"))
        inferred_model, inferred = load_compositional_model(directory, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            torch.testing.assert_close(inferred_model(input_ids=input_ids).logits, expected, rtol=0, atol=0)
        assert inferred["arm"] == "product_code"
        assert inferred["product_code_assignment"] == "checkpoint"


def test_training_builders_hashed_resume_and_pq_paths():
    from train_compositional import (
        CompositionalArguments,
        _matches_declared_no_decay,
        build_arm,
        validate_output_configuration,
    )

    declared = ProductCodeEmbed.no_decay_parameters()
    assert _matches_declared_no_decay("model.embed_tokens.embed.gate_offsets", declared)
    assert _matches_declared_no_decay("module.model.embed_tokens.embed.gate_offsets", declared)
    assert not _matches_declared_no_decay("model.embed_tokens.embed.E_h", declared)
    vocab = 30
    importance = _importance(vocab).numpy()
    with tempfile.TemporaryDirectory() as directory:
        importance_path = os.path.join(directory, "importance.npz")
        np.savez(importance_path, counts=importance)
        args = CompositionalArguments(
            arm="product_code", product_code_head_size=5,
            product_code_num_hashes=2, product_code_num_buckets=7,
            product_code_importance_path=importance_path, tie_output=True,
        )
        fresh = build_arm(args, vocab, 8)
        assert isinstance(fresh, ProductCodeEmbed)
        assert validate_output_configuration(args) == "tied"

        # Resume: checkpoint structure is authoritative; the importance file
        # must not be needed.
        args.product_code_importance_path = "/does/not/exist"
        resumed = build_arm(args, vocab, 8, initial_state=fresh.state_dict())
        resumed.load_state_dict(fresh.state_dict(), strict=True)
        torch.testing.assert_close(resumed.materialize(), fresh.materialize(), rtol=0, atol=0)
        bad = CompositionalArguments(
            arm="product_code", product_code_head_size=6,
            product_code_num_hashes=2, product_code_num_buckets=7,
            product_code_importance_path=importance_path, tie_output=True,
        )
        with pytest.raises(ValueError, match="does not match"):
            build_arm(bad, vocab, 8, initial_state=fresh.state_dict())

        # PQ path: codes artifact must agree with the implied tail partition.
        args.product_code_importance_path = importance_path
        args.product_code_assignment = "pq"
        codes_path = os.path.join(directory, "codes.pt")
        table = torch.randn(vocab, 8)
        codes = pq_codes(table[fresh.tail_ids], 2, 7, iters=3)
        torch.save({"codes": codes, "tail_ids": fresh.tail_ids.clone(), "provenance": {"head_size": 5, "num_hashes": 2, "num_buckets": 7}}, codes_path)
        args.product_code_codes_path = codes_path
        pq = build_arm(args, vocab, 8)
        assert torch.equal(pq.codes, codes) and pq.assignment == "pq"
        torch.save({"codes": codes, "tail_ids": fresh.tail_ids.flip(0), "provenance": {"head_size": 5, "num_hashes": 2, "num_buckets": 7}}, codes_path)
        with pytest.raises(ValueError, match="tail ids do not match"):
            build_arm(args, vocab, 8)
        torch.save({"codes": codes, "tail_ids": fresh.tail_ids.clone(),
                    "provenance": {"head_size": 5, "num_hashes": 2, "num_buckets": 3}},
                   codes_path)
        with pytest.raises(ValueError, match="num_buckets=3"):
            build_arm(args, vocab, 8)
        torch.save({"codes": codes, "tail_ids": fresh.tail_ids.clone()}, codes_path)
        with pytest.raises(ValueError, match="no provenance"):
            build_arm(args, vocab, 8)

        # Resume with a different model dimension must fail clearly.
        with pytest.raises(ValueError, match="do not match the model"):
            build_arm(args, vocab + 1, 8, initial_state=fresh.state_dict())


def test_trainer_excludes_gates_from_weight_decay():
    from transformers import TrainingArguments
    from train_compositional import CompositionalArguments, CompositionalTrainer

    embed = ProductCodeEmbed(VOCAB_SIZE, HIDDEN_SIZE, 4, 3, 5, importance=_importance(VOCAB_SIZE))
    model = _tiny_qwen()
    model.model.embed_tokens = EmbeddingShim(embed)
    model.lm_head = make_tied_head(embed, "product_code", VOCAB_SIZE)
    with tempfile.TemporaryDirectory() as directory:
        trainer = CompositionalTrainer(
            model=model,
            args=TrainingArguments(output_dir=directory, report_to=[]),
            embed_shim=model.model.embed_tokens,
            comp_args=CompositionalArguments(arm="product_code", tie_output=True),
        )
        decay = set(trainer.get_decay_parameter_names(model))
    assert "model.embed_tokens.embed.gate_offsets" not in decay
    assert "model.embed_tokens.embed.E_h" in decay
    assert "model.embed_tokens.embed.C.0" in decay


def _naive_table(embed):
    """Reference effective table built with explicit per-token loops."""
    rows = []
    gates = 1.0 + embed.gate_offsets
    for token in range(embed.vocab_size):
        vector = embed.bias.clone()
        head = int(embed.head_row[token])
        if head >= 0:
            vector = vector + embed.E_h[head]
        else:
            tail = int(embed.tail_row[token])
            for index in range(embed.num_hashes):
                vector = vector + gates[tail, index] * embed.C[index][int(embed.codes[tail, index])]
        rows.append(vector)
    return torch.stack(rows)


def test_materialize_and_head_match_naive_reference_with_nonzero_bias():
    torch.manual_seed(11)
    embed = ProductCodeEmbed(23, 6, 4, 3, 5, importance=_importance(23)).double()
    with torch.no_grad():
        embed.bias.normal_()
        embed.gate_offsets.uniform_(-0.8, 0.8)
    reference = _naive_table(embed)
    torch.testing.assert_close(embed.materialize(), reference, rtol=0, atol=1e-12)
    hidden = torch.randn(3, 6, dtype=torch.float64)
    torch.testing.assert_close(
        make_tied_head(embed, "product_code", 23)(hidden), hidden @ reference.T,
        rtol=0, atol=1e-12,
    )


def test_tied_head_gradients_match_materialized_head():
    torch.manual_seed(12)
    embed = ProductCodeEmbed(19, 5, 3, 2, 4, importance=_importance(19)).double()
    with torch.no_grad():
        embed.bias.normal_()
        embed.gate_offsets.uniform_(-0.5, 0.5)
    hidden = torch.randn(4, 5, dtype=torch.float64)
    upstream = torch.randn(4, 19, dtype=torch.float64)

    (make_tied_head(embed, "product_code", 19)(hidden) * upstream).sum().backward()
    fast = {name: p.grad.clone() for name, p in embed.named_parameters()}
    embed.zero_grad()
    ((hidden @ embed.materialize().T) * upstream).sum().backward()
    for name, p in embed.named_parameters():
        torch.testing.assert_close(fast[name], p.grad, rtol=0, atol=1e-12, msg=name)


def test_gate_offsets_can_move_in_bfloat16_where_unit_gates_cannot():
    step = 3e-4  # the production learning rate
    frozen = torch.ones(4, dtype=torch.bfloat16)
    frozen = frozen - step
    assert torch.equal(frozen, torch.ones(4, dtype=torch.bfloat16))  # 1.0 - 3e-4 rounds back to 1.0
    offsets = torch.zeros(4, dtype=torch.bfloat16) - step
    assert torch.all(offsets != 0)
    embed = ProductCodeEmbed(12, 4, 2, 2, 4, importance=_importance(12)).to(torch.bfloat16)
    assert embed.gate_offsets.dtype == torch.bfloat16
    assert embed.codes.dtype == torch.long
    torch.testing.assert_close(embed.gates.float(), torch.ones(10, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_resident_module_loads_its_state_dict_and_matches_cpu():
    embed = ProductCodeEmbed(41, 8, 6, 3, 7, importance=_importance(41))
    with torch.no_grad():
        embed.bias.normal_()
        embed.gate_offsets.uniform_(-0.5, 0.5)
    state = {k: v.clone() for k, v in embed.state_dict().items()}
    cuda = ProductCodeEmbed(41, 8, 6, 3, 7, importance=_importance(41)).to("cuda")
    cuda.load_state_dict(state, strict=True)          # post-hook validates on CUDA
    hidden = torch.randn(3, 8)
    torch.testing.assert_close(
        make_tied_head(cuda, "product_code", 41)(hidden.cuda()).cpu(),
        make_tied_head(embed, "product_code", 41)(hidden),
        rtol=1e-5, atol=1e-5,
    )


def test_tied_head_saves_at_most_one_table_sized_tensor():
    vocab, d = 64, 8
    embed = ProductCodeEmbed(vocab, d, 4, 3, 5, importance=_importance(vocab))
    hidden = torch.randn(16, d)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.numel()) or t, lambda t: t):
        make_tied_head(embed, "product_code", vocab)(hidden).sum().backward()
    n, n_tail = 16, 60
    assert not any(size == n * n_tail for size in saved), "a full (N, N_t) operand was saved"
    table_sized = [size for size in saved if size == vocab * d]
    assert len(table_sized) <= 1, f"{len(table_sized)} table-sized tensors saved (gathers not recomputed)"


def test_pq_codes_reject_uncoverable_space_up_front():
    with pytest.raises(ValueError, match="cannot cover"):
        pq_codes(torch.randn(280, 8), 2, 5, iters=1)


def test_pq_duplicate_repair_is_guaranteed_on_adversarial_tables():
    # Fully identical rows, duplicate-heavy rows, and random small spaces must
    # all yield unique signatures whenever B^H >= n.
    identical = torch.ones(50, 8)
    codes = pq_codes(identical, 2, 8, iters=2, seed=0)             # 64 >= 50
    assert torch.unique(codes, dim=0).size(0) == 50
    heavy = torch.randn(120, 12, generator=torch.Generator().manual_seed(4))
    heavy[:90] = heavy[0]
    codes = pq_codes(heavy, 3, 5, iters=3, seed=1)                  # 125 >= 120
    assert torch.unique(codes, dim=0).size(0) == 120
    generator = torch.Generator().manual_seed(9)
    for trial in range(40):
        H = int(torch.randint(2, 5, (1,), generator=generator))
        B = int(torch.randint(2, 7, (1,), generator=generator))
        n = int(torch.randint(1, max(2, B ** H), (1,), generator=generator))
        n = min(n, 400)
        d = 4 * H
        table = torch.randn(n, d, generator=generator)
        if trial % 3 == 0:
            table[: n // 2] = table[0]
        if B > n:
            continue
        codes = pq_codes(table, H, B, iters=2, seed=trial, capacity_factor=2.0)
        assert codes.shape == (n, H)
        assert torch.unique(codes, dim=0).size(0) == n, (H, B, n)
        assert torch.all(codes >= 0) and torch.all(codes < B)


def test_runner_resolves_required_inputs_and_requires_nonempty(monkeypatch, tmp_path):
    import run_experiments as runner

    empty = tmp_path / "empty.pt"; empty.touch()
    full = tmp_path / "full.pt"; full.write_bytes(b"x")
    monkeypatch.delenv("PRODUCT_CODE_CODES_PATH", raising=False)
    spec = "${PRODUCT_CODE_CODES_PATH:-" + str(full) + "}"
    assert runner.resolve_input_path(spec) == str(full)
    assert runner.missing_input_files({"required_input_files": [spec]}) == []
    monkeypatch.setenv("PRODUCT_CODE_CODES_PATH", str(empty))
    assert runner.missing_input_files({"required_input_files": [spec]})
    names = [e["name"] for e in runner.EXPERIMENT_COMMANDS]
    assert names[-2:] == ["product_code_hashed_h2048", "product_code_pq_h2048"]
    pq = runner.EXPERIMENT_COMMANDS[-1]
    assert any("PRODUCT_CODE_CODES_PATH" in s for s in pq["required_input_files"])


def test_tied_head_gradients_match_explicit_per_token_reference():
    # Independent of _GatedCodebookSum: the reference is the explicit per-token
    # loop in _naive_table, so a custom-backward regression cannot satisfy
    # both sides of this comparison.
    torch.manual_seed(21)
    embed = ProductCodeEmbed(23, 6, 4, 3, 5, importance=_importance(23)).double()
    with torch.no_grad():
        embed.bias.normal_()
        embed.gate_offsets.uniform_(-0.6, 0.6)
    hidden = torch.randn(5, 6, dtype=torch.float64)
    upstream = torch.randn(5, 23, dtype=torch.float64)

    (make_tied_head(embed, "product_code", 23)(hidden) * upstream).sum().backward()
    fast = {name: p.grad.clone() for name, p in embed.named_parameters()}
    embed.zero_grad()
    ((hidden @ _naive_table(embed).T) * upstream).sum().backward()
    for name, p in embed.named_parameters():
        torch.testing.assert_close(fast[name], p.grad, rtol=0, atol=1e-12, msg=name)
