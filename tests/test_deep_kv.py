"""Four-arm mechanism acceptance tests, on CPU only."""
import unittest
import torch
import transformers
from transformers import Qwen3Config
from deep_kv.model import DeepKV, Context

def parameter_hash(module):
    import hashlib
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def config():
    result = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
                        tie_word_embeddings=True)
    result._attn_implementation = "sdpa"
    return result


def model(arm, checkpoint=False):
    return DeepKV.from_scratch(config(), arm, consumer=2, deep_target=4,
                               checkpoint_layers=checkpoint, lm_chunk=3)


def objective(m, result):
    if m.arm in ("F", "G"):
        aux = result["route_sum"] + (result["msg_sum"] if m.arm == "G" else 0)
        return result["lm_sum"] / result["lm_count"] + .3 * aux / result["route_count"].clamp_min(1)
    return result["lm_sum"] / result["lm_count"] + m.kv_loss_weight * (result["k_sum"] + result["v_sum"]) / (2 * result["kv_count"])


def context():
    ids = torch.tensor([[3, 8, 2, 6, 9, 1], [4, 7, 8, 0, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 0, 0]])
    segments = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    return Context(ids, valid, positions, segments)


@unittest.skipUnless(transformers.__version__ == "5.9.0", "Deep-KV uses the B200-matched Transformers 5.9.0 environment")
class DeepKVAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_base_equivalence_and_identical_initialization(self):
        batch = context()
        base = model("A").eval()
        with torch.no_grad():
            native = base.backbone(input_ids=batch.input_ids, attention_mask=batch.additive_mask(torch.float32),
                                   position_ids=batch.position_ids, use_cache=False).logits
            base_logits = base.backbone.lm_head(base.hidden_states(batch)[0])
        torch.testing.assert_close(base_logits, native)
        branch_hash = None
        for arm in "BCDEFG":
            m = model(arm).eval()
            self.assertEqual(parameter_hash(base.backbone), parameter_hash(m.backbone))
            if branch_hash is None:
                branch_hash = parameter_hash(m.aux)
            self.assertEqual(branch_hash, parameter_hash(m.aux))
            with torch.no_grad():
                logits = m.backbone.lm_head(m.hidden_states(batch)[0])
            torch.testing.assert_close(logits, native, atol=1e-6, rtol=1e-5)

    def test_strict_mask_empty_rows_and_reference_attention(self):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        m = model("B")
        torch.nn.init.normal_(m.aux.out.weight, std=.1)
        batch = context()
        u = torch.randn(2, 6, 32, requires_grad=True)
        q = torch.randn(2, 4, 6, 8)
        rotary = m.backbone.model.rotary_emb(u, batch.position_ids)
        qr, _ = apply_rotary_pos_emb(q, q, *rotary)
        actual, k, v = m.aux(u, qr, rotary, batch.allowed())
        _, kr = apply_rotary_pos_emb(qr, k, *rotary)
        allowed = batch.allowed() & torch.ones(6, 6, dtype=torch.bool).tril(-1)
        nonempty = allowed.any(-1, keepdim=True)
        scores = qr @ repeat_kv(kr, 2).transpose(-1, -2) / 8 ** .5
        scores = scores.masked_fill(~allowed, -torch.inf)
        weights = torch.where(nonempty, scores, torch.zeros_like(scores)).softmax(-1).masked_fill(~allowed, 0)
        expected = m.aux.out((weights @ repeat_kv(v, 2)).transpose(1, 2).reshape(2, 6, -1))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(weights.masked_select(~allowed.expand_as(weights)).count_nonzero().item(), 0)
        self.assertTrue(torch.equal(actual[0, 0], torch.zeros(32)))
        self.assertTrue(torch.equal(actual[0, 3], torch.zeros(32)))
        self.assertTrue(torch.equal(actual[1, 3:], torch.zeros(3, 32)))
        actual[0, 1].sum().backward()
        self.assertEqual(u.grad[0, 1:].count_nonzero().item(), 0)
        changed = u.detach().clone()
        changed[0, 2:] += 100
        after = m.aux(changed, qr, rotary, batch.allowed())[0]
        torch.testing.assert_close(actual[0, :2], after[0, :2])

    def test_native_targets_exact_and_one_projection_per_forward(self):
        for arm, target_index in (("C", 4), ("D", 20), ("E", 20), ("F", 20), ("G", 20)):
            cfg = config()
            cfg.num_hidden_layers = 28
            cfg.layer_types = ["full_attention"] * 28
            m = DeepKV.from_scratch(cfg, arm, checkpoint_layers=False).eval()
            captures, handles = {}, []
            def hook(name):
                def save(module, args, output):
                    captures.setdefault(name, []).append(output)
                return save
            for i, layer in enumerate(m.backbone.model.layers):
                for key in ("q_proj", "k_norm", "v_proj"):
                    handles.append(getattr(layer.self_attn, key).register_forward_hook(hook((i, key))))
            try:
                _, predicted, target = m.hidden_states(context())
            finally:
                for h in handles:
                    h.remove()
            self.assertTrue(all(len(x) == 1 for x in captures.values()))
            native_k = captures[(target_index, "k_norm")][0].transpose(1, 2)
            native_v = captures[(target_index, "v_proj")][0].view(2, 6, 2, 8).transpose(1, 2)
            if arm in ("F", "G"):
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                batch = context()
                rotary = m.backbone.model.rotary_emb(m.backbone.model.embed_tokens(batch.input_ids), batch.position_ids)
                _, native_k = apply_rotary_pos_emb(native_k, native_k, *rotary)
            self.assertTrue(torch.equal(target[0], native_k))
            self.assertTrue(torch.equal(target[1], native_v))
            self.assertEqual(predicted[0].shape, (2, 2, 6, 8))

    def test_target_stopgrad_and_padding_exclusion(self):
        for arm, target_index in (("C", 1), ("D", 3), ("E", 3)):
            m = model(arm)
            batch = context()
            _, predicted, target = m.hidden_states(batch)
            for t in target:
                t.retain_grad()
            k, v = m.alignment(predicted, target, batch.valid)
            expected = [(p - t.detach()).abs().permute(0, 2, 1, 3)[batch.valid].mean() for p, t in zip(predicted, target)]
            torch.testing.assert_close(k / batch.valid.sum(), expected[0])
            torch.testing.assert_close(v / batch.valid.sum(), expected[1])
            ((k + v) / (2 * batch.valid.sum())).backward()
            self.assertTrue(all(t.grad is None for t in target))
            a = m.backbone.model.layers[target_index].self_attn
            self.assertIsNone(a.k_proj.weight.grad)
            self.assertIsNone(a.v_proj.weight.grad)
            self.assertGreater(m.aux.k.weight.grad.abs().sum().item(), 0)
            self.assertGreater(m.aux.v.weight.grad.abs().sum().item(), 0)
            self.assertGreater(m.backbone.model.embed_tokens.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_recomputation_and_lm_gradient_path(self):
        for arm in "ABCDEFG":
            plain, checked = model(arm), model(arm, checkpoint=True)
            if plain.aux is not None:
                torch.nn.init.normal_(plain.aux.out.weight, std=.03)
                checked.load_state_dict(plain.state_dict())
            for m in (plain, checked):
                result = m(context())
                loss = objective(m, result)
                loss.backward()
            for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
                self.assertIsNotNone(p.grad, name)
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)
            if plain.aux is not None:
                plain.zero_grad(set_to_none=True)
                plain(context())["lm_sum"].backward()
                self.assertGreater(plain.aux.k.weight.grad.abs().sum().item(), 0)
                self.assertGreater(plain.backbone.model.layers[-1].self_attn.v_proj.weight.grad.abs().sum().item(), 0)

    def test_bfloat16_forward_backward(self):
        reference = None
        for arm in "ABCDEFG":
            m = model(arm, checkpoint=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = m(context())
                loss = objective(m, result)
            if reference is None:
                reference = result["lm_sum"].detach()
            torch.testing.assert_close(result["lm_sum"], reference, atol=1e-5, rtol=1e-5)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_query_width_can_exceed_hidden_size_as_in_qwen_06b(self):
        cfg = config()
        cfg.head_dim = 16  # four query heads = 64 features, hidden size = 32
        m = DeepKV.from_scratch(cfg, "D", consumer=2, deep_target=4)
        self.assertEqual(m.aux.out.in_features, 64)
        result = m(context())
        loss = objective(m, result)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_modified_arms_have_identical_lm_path_after_branch_learns(self):
        # Zero-output equivalence alone would hide mistakes in the live branch.
        for bf16 in (False, True):
            models = [model(arm, checkpoint=True) for arm in "BCDEFG"]
            torch.nn.init.normal_(models[0].aux.out.weight, std=.03)
            for other in models[1:]:
                other.load_state_dict(models[0].state_dict())
            reference = None
            for current in models:
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                    result = current(context())
                if reference is None:
                    reference = result["lm_sum"].detach()
                torch.testing.assert_close(result["lm_sum"], reference, atol=1e-6, rtol=1e-6)
                result["lm_sum"].backward()
            for other in models[1:]:
                for (name, p), (_, q) in zip(models[0].named_parameters(), other.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-6, msg=name)

    def test_functional_losses_match_independent_dense_reference_and_gradients(self):
        from torch.nn import functional as F
        batch = context()
        torch.manual_seed(61)
        # Uneven padding, reset positions, segments, GQA and nonempty singleton
        # rows exercise the actual loss denominator and KL direction.
        originals = [torch.randn(2, h, 6, 8) for h in (2, 2, 4, 2, 2)]
        for with_message in (False, True):
            ref = [t.clone().requires_grad_() for t in originals]
            pk, pv, q, dk, dv = ref
            expected_r, expected_m, counts = [], [], []
            for b in range(2):
                route, msg, count = pk[b].sum() * 0, pv[b].sum() * 0, 0
                for t in range(6):
                    source = [j for j in range(t) if batch.allowed()[b, 0, t, j]]
                    if not source:
                        continue
                    count += 1
                    for h in range(4):
                        g = h // 2
                        query = q[b, h, t].detach()
                        deep_log = (dk[b, g, source].detach() @ query / 8 ** .5).log_softmax(-1)
                        pred_log = (pk[b, g, source] @ query / 8 ** .5).log_softmax(-1)
                        route = route + F.kl_div(pred_log, deep_log, reduction="sum", log_target=True) / 4
                        if with_message:
                            pm = pred_log.exp() @ pv[b, g, source]
                            dm = deep_log.exp() @ dv[b, g, source].detach()
                            msg = msg + F.smooth_l1_loss(pm, dm, beta=1., reduction="mean") / 4
                expected_r.append(route); expected_m.append(msg); counts.append(count)
            expected_r, expected_m = torch.stack(expected_r), torch.stack(expected_m)
            (expected_r.sum() + expected_m.sum()).backward()
            for size in (1, 2, 128):
                for recompute in (False, True):
                    tensors = [t.clone().requires_grad_() for t in originals]
                    pk, pv, q, dk, dv = tensors
                    actual_r, actual_m, actual_counts = DeepKV.routing_alignment(
                        (pk, pv, q), (dk, dv), batch.allowed(), message=with_message,
                        query_chunk=size, checkpoint_chunks=recompute)
                    torch.testing.assert_close(actual_r, expected_r)
                    torch.testing.assert_close(actual_m, expected_m)
                    self.assertEqual(actual_counts.tolist(), counts)
                    (actual_r.sum() + actual_m.sum()).backward()
                    torch.testing.assert_close(pk.grad, ref[0].grad, atol=2e-6, rtol=2e-5)
                    if with_message:
                        torch.testing.assert_close(pv.grad, ref[1].grad, atol=2e-6, rtol=2e-5)
                    else:
                        self.assertIsNone(pv.grad)
                    for t in (q, dk, dv):
                        self.assertIsNone(t.grad)
                    # Final positions and padding never serve a valid later query.
                    self.assertEqual(pk.grad[0, :, [2, 5]].count_nonzero().item(), 0)
                    self.assertEqual(pk.grad[1, :, 2:].count_nonzero().item(), 0)

    def test_functional_empty_rows_identical_targets_and_fp32_under_autocast(self):
        batch = context()
        tensors = [torch.randn(2, h, 6, 8).bfloat16().requires_grad_() for h in (2, 2, 4)]
        pk, pv, q = tensors
        for allowed in (batch.allowed(), torch.eye(6, dtype=torch.bool)[None, None].expand(2, 1, 6, 6)):
            for t in tensors:
                t.grad = None
            with torch.autocast("cpu", dtype=torch.bfloat16):
                r, m, counts = DeepKV.routing_alignment((pk, pv, q), (pk, pv), allowed,
                                                        message=True, query_chunk=2)
            self.assertEqual(r.dtype, torch.float32)
            self.assertEqual(m.dtype, torch.float32)
            torch.testing.assert_close(r, torch.zeros(2), atol=1e-7, rtol=0)
            torch.testing.assert_close(m, torch.zeros(2), atol=1e-7, rtol=0)
            (r.sum() + m.sum()).backward()
            self.assertTrue(torch.isfinite(pk.grad).all() and torch.isfinite(pv.grad).all())
            torch.testing.assert_close(pk.grad.float(), torch.zeros_like(pk.grad).float(), atol=2e-7, rtol=0)
            self.assertIsNone(q.grad)
        self.assertEqual(counts.sum().item(), 0)

    def test_functional_query_target_detach_and_exact_forward_attention_inputs(self):
        from unittest.mock import patch
        for arm in "FG":
            m = model(arm, checkpoint=True)
            torch.nn.init.normal_(m.aux.out.weight, std=.03)
            batch = context()
            captured = {}
            original = torch.nn.functional.scaled_dot_product_attention
            def capture(q, k, v, *args, **kwargs):
                if kwargs.get("attn_mask") is not None and kwargs["attn_mask"].dtype == torch.bool:
                    captured.update(q=q.detach(), k=k.detach(), v=v.detach())
                return original(q, k, v, *args, **kwargs)
            with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=capture):
                _, predicted, target = m.hidden_states(batch)
            from transformers.models.qwen3.modeling_qwen3 import repeat_kv
            torch.testing.assert_close(predicted[2], captured["q"], rtol=0, atol=0)
            torch.testing.assert_close(repeat_kv(predicted[0], 2), captured["k"], rtol=0, atol=0)
            torch.testing.assert_close(repeat_kv(predicted[1], 2), captured["v"], rtol=0, atol=0)
            self.assertFalse(predicted[2].requires_grad)
            self.assertTrue(all(not t.requires_grad for t in target))
            r, msg, _ = m.routing_alignment(predicted, target, batch.allowed(), message=arm == "G", query_chunk=2)
            (r.sum() + msg.sum()).backward()
            consumer = m.backbone.model.layers[m.consumer - 1].self_attn
            deep = m.backbone.model.layers[m.deep_target - 1].self_attn
            self.assertIsNone(consumer.q_proj.weight.grad)
            self.assertIsNone(consumer.q_norm.weight.grad)
            self.assertIsNone(deep.k_proj.weight.grad)
            self.assertIsNone(deep.v_proj.weight.grad)
            self.assertIsNone(m.aux.out.weight.grad)
            self.assertGreater(m.aux.k.weight.grad.abs().sum().item(), 0)
            if arm == "F":
                self.assertIsNone(m.aux.v.weight.grad)
            else:
                self.assertGreater(m.aux.v.weight.grad.abs().sum().item(), 0)
            m.zero_grad(set_to_none=True)
            with patch.object(m, "alignment", side_effect=AssertionError("No raw K/V loss allowed")):
                result = m(batch)
            result["lm_sum"].backward()
            self.assertGreater(consumer.q_proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(m.aux.v.weight.grad.abs().sum().item(), 0)



if __name__ == "__main__":
    unittest.main()
