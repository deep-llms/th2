"""CPU-default acceptance tests. CUDA driver explicitly opts into GPU execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from ccm.contracts import Budget, PILOT, schedule, write_json, read_json, digest_json
from ccm.data import build_manifest, Corpus, collate, validate_splits, cuts
from ccm.keys import position_keys, Vocabulary, count_vocabulary, safe_gather
from ccm.model import MemoryLM
from ccm.artifacts import Accumulator, save_bundle, load_bundle, load_table, state_hash
from ccm.runtime import optimizer_groups, MasterAdamW, train, load_model, save_checkpoint
from ccm.compiler import coverage, compile_tables
from ccm.evaluation import evaluate, validate_final_access
from ccm.statistics import bootstrap, paired_arrays
from ccm.cli import parser
from ccm.decisions import lock_final
from ccm.statistics import compare

torch.set_num_threads(1)


def tiny_config():
    c = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                   num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                   head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
                   attention_dropout=0.0, pad_token_id=0)
    c._attn_implementation = "sdpa"
    return c


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0]

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return dict(input_ids=[2, 3]*8+[4+len(text) % 4])


class Temporary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def corpus(self, compile_tokens=64, duplicates=False):
        budget = Budget(batch_tokens=32, common_steps=8, adapt_steps=2, continue_steps=2,
                        compile_tokens=compile_tokens, dev_tokens=32, val_tokens=32, segment_length=8, slots=2)
        docs = ((f"doc{i}", f"document {i//2 if duplicates else i}") for i in range(5000))
        build_manifest(docs, Tokenizer(), self.root/"data", budget,
                       dict(tokenizer={"repo_id": "toy", "revision": "toy"}), engineering=True)
        return Corpus(self.root/"data")


class DataTests(Temporary):
    def test_misaligned_compile_boundary_and_duplicate_groups(self):
        # Production 1B compilation is NOT a multiple of the global token batch.
        c = self.corpus(compile_tokens=70, duplicates=True)
        validate_splits(c)
        self.assertGreater(c.meta["duplicate_groups"], 0)
        common = [r for b in c.optimizer_batches("common", 1017) for r in b]
        adapt = [r for b in c.optimizer_batches("stage1", 1017) for r in b]
        in_common = {r["segment_id"]: r["tokens"].tolist() for r in common}
        for row in adapt:
            self.assertEqual(in_common[row["segment_id"]], row["tokens"].tolist())
        for phase in ("common", "stage1", "stage2"):
            self.assertTrue(all(sum(len(r["tokens"]) for r in b) == 32 for b in c.optimizer_batches(phase, 1017)))
        roles_by_content = {}
        for role in c.meta["quotas"]:
            for row in c.segments(role):
                self.assertEqual(roles_by_content.setdefault(row["content_hash"], role), role)

    def test_exact_quotas_and_immutable_phase_cuts(self):
        c = self.corpus()
        validate_splits(c)
        seen = {}
        for role in c.meta["quotas"]:
            rows = list(c.segments(role))
            self.assertEqual(sum(len(r["tokens"]) for r in rows), c.meta["quotas"][role])
            for row in rows:
                self.assertEqual(seen.setdefault(row["doc_id"], role), role)
        for phase, expected in (("common", 8), ("stage1", 2), ("stage2", 2)):
            batches = list(c.optimizer_batches(phase, 1017))
            self.assertEqual(len(batches), expected)
            self.assertTrue(all(sum(len(r["tokens"]) for r in batch) == 32 for batch in batches))
            self.assertEqual([[r["segment_id"] for r in b] for b in batches],
                             [[r["segment_id"] for r in b] for b in c.optimizer_batches(phase, 1017)])

    def test_routing_alignment_boundaries_special_tokens(self):
        ids = [2, 3, 4, 0, 5, 6]
        keys, e = position_keys(ids, [0])
        self.assertEqual(e.tolist(), [False, True, True, False, False, False])
        vocab = Vocabulary([keys[1], keys[2]], [8, 7], {})
        row = dict(tokens=ids)
        b = collate([row, dict(tokens=[2, 3])], [0], vocab)
        self.assertEqual(b["slots"][0].tolist(), [-1, 0, 1, -1, -1, -1])
        self.assertEqual(b["targets"][0].tolist(), [3, 4, 0, 5, 6, -100])
        self.assertTrue((b["slots"][1] == -1).all())
        self.assertEqual(b["position_ids"][1, 0].item(), 0)

    def test_counting_deterministic_and_coverage(self):
        c = self.corpus()
        meta = dict(corpus_hash=c.meta["manifest_hash"])
        v1 = count_vocabulary(c.segments("compile"), [0], 2, self.root/"a.sqlite", meta)
        v2 = count_vocabulary(c.segments("compile"), [0], 2, self.root/"b.sqlite", meta)
        self.assertEqual(v1.hash, v2.hash)
        v1.save(self.root/"v.npz")
        self.assertEqual(v1.hash, Vocabulary.load(self.root/"v.npz").hash)
        r = coverage(c, v1, self.root/"coverage.json")
        self.assertGreaterEqual(r["eligible_hit_rate"], .2)
        self.assertLessEqual(r["overall_memory_active_rate"], r["eligible_hit_rate"])

    def test_corruption_refused(self):
        c = self.corpus()
        with (c.path/"dev.tokens").open("r+b") as f:
            f.write(b"xxxx")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Corpus(c.path)


class ModelTests(Temporary):
    def setUp(self):
        super().setUp()
        torch.manual_seed(17)
        self.base = MemoryLM(tiny_config())
        self.table = torch.randn(3, 16).bfloat16()
        self.batch = collate([dict(tokens=[2, 3, 4, 5, 6]), dict(tokens=[7, 8, 9])], [0])
        self.batch["slots"][0, 1:4] = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.batch["slots"][1, 1] = 0

    def model(self, arm="contextual"):
        m = MemoryLM(tiny_config(), arm, table=self.table, slots=3)
        m.backbone.load_state_dict(self.base.backbone.state_dict())
        return m

    def run_model(self, m, b=None, **kw):
        b = b or self.batch
        return m(**{k: b[k] for k in ("input_ids", "attention_mask", "position_ids", "slots", "targets")}, **kw)

    def test_zero_initialization_fp32_and_bf16(self):
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.bfloat16, 1e-3)):
            base, memory = copy.deepcopy(self.base.backbone), self.model()
            base.to(dtype).eval()
            memory.to(dtype).eval()
            with torch.no_grad():
                a = base(input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"],
                         position_ids=self.batch["position_ids"]).logits
                b = self.run_model(memory, return_logits=True)["logits"]
            self.assertLessEqual(float((a-b).abs().max()), tolerance)

    def test_frozen_trainable_and_misses(self):
        for arm in ("shallow", "contextual", "delta", "isolated", "shuffled", "grad"):
            m = self.model(arm)
            m.set_phase("stage1")
            opt = MasterAdamW(optimizer_groups(m, "stage1"))
            ids = {id(p) for g in optimizer_groups(m, "stage1") for p in g["params"]}
            self.assertEqual(id(m.table) in ids, arm == "grad")
            initial_table = m.table.detach().clone()
            for _ in range(2):
                opt.zero_grad()
                r = self.run_model(m)
                (r["loss_sum"]/r["target_count"]).backward()
                self.assertTrue(m.reader.wv.weight.grad.abs().max() > 0)
                if arm != "grad":
                    self.assertIsNone(m.table.grad)
                    self.assertFalse(m.table.requires_grad)
                for g in opt.inner.param_groups:
                    g["lr"] = .01
                opt.step()
            self.assertEqual(torch.equal(initial_table, m.table), arm != "grad")
            misses = {k: v.clone() for k, v in self.batch.items()}
            misses["slots"].fill_(-1)
            with torch.no_grad():
                a = self.run_model(self.base, misses, return_logits=True)["logits"]
                b = self.run_model(m, misses, return_logits=True)["logits"]
            self.assertTrue(torch.equal(a, b))

    def test_safe_miss_gather_no_dummy_gradient(self):
        t = torch.randn(3, 5, requires_grad=True)
        x = safe_gather(t, torch.tensor([-1, 1, -1]))
        self.assertTrue((x[[0, 2]] == 0).all())
        x.sum().backward()
        self.assertTrue((t.grad[0] == 0).all() and (t.grad[-1] == 0).all())
        with self.assertRaises(ValueError):
            safe_gather(t, torch.tensor([-2]))

    def test_hook_points_and_attention_isolation(self):
        m = self.model()
        with torch.no_grad():
            m.reader.wv.weight.copy_(torch.eye(16)*.1)
        references = {}
        h1 = m.backbone.model.layers[1].register_forward_pre_hook(
            lambda module, args, kwargs: references.update(block2_args=args, block2_kwargs=kwargs), with_kwargs=True)
        h2 = m.backbone.model.norm.register_forward_pre_hook(lambda module, args: references.update(deep=args[0]))
        r = self.run_model(m, capture=True, return_logits=True)
        self.assertTrue(torch.equal(r["r12_pre_final_norm"], references["deep"]))
        direct_shallow = m.backbone.model.layers[1].forward(*references["block2_args"], **references["block2_kwargs"])
        self.assertTrue(torch.equal(r["r2_pre_memory"], direct_shallow))
        self.assertFalse(torch.equal(r["r2_pre_memory"], r["u2_post_memory"]))
        self.assertTrue(torch.allclose(m.backbone.model.norm(r["r12_pre_final_norm"]), r["hidden"]))
        changed = {k: v.clone() for k, v in self.batch.items()}
        changed["input_ids"][0].fill_(12)
        other = self.run_model(m, changed, return_logits=True)
        self.assertTrue(torch.equal(r["logits"][1], other["logits"][1]))
        h1.remove(); h2.remove()

    def test_checkpointing_offline_gradients(self):
        a, b = self.model(), self.model()
        for m in (a, b):
            m.set_phase("stage2"); m.train()
        b.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.run_model(a)["loss_sum"].backward()
        self.run_model(b)["loss_sum"].backward()
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(na, nb)
            self.assertIsNotNone(pa.grad)
            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=1e-5), na)

    def test_paired_reader_and_chunked_loss(self):
        a, b = self.model("contextual"), self.model("grad")
        self.assertEqual(state_hash(a.reader.state_dict()), state_hash(b.reader.state_dict()))
        r = self.run_model(a, return_logits=True, loss_chunk=3)
        ref = torch.nn.functional.cross_entropy(r["logits"].flatten(0, 1).float(), self.batch["targets"].flatten(),
                                                ignore_index=-100, reduction="sum")
        self.assertTrue(torch.allclose(r["loss_sum"], ref))


class ArtifactTests(Temporary):
    def test_mean_variance_and_artifact_contract(self):
        a = Accumulator(2, 3)
        z = torch.tensor([[1., 2., 3.], [3., 4., 5.], [7., 8., 9.]])
        a.add(torch.tensor([0, 0, 1]), z)
        result = a.finish()
        self.assertTrue(torch.equal(result["master_sum"][0]/2, z[:2].mean(0)))
        self.assertAlmostEqual(result["variance"][0].item(), z[:2].var(0, unbiased=False).mean().item())
        save_bundle(self.root/"table", result, dict(constructor="contextual", vocabulary_hash="x"))
        tensors, _ = load_bundle(self.root/"table")
        self.assertTrue(torch.equal(tensors["lookup"], result["lookup"]))
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_table(self.root/"table", dict(vocabulary_hash="wrong"))

    def test_scheduler_boundaries(self):
        for n, w, f, p in ((15259, 306, .02, 3e-4), (977, 49, .05, 5e-4), (3815, 77, .02, 1.5e-4)):
            self.assertAlmostEqual(schedule(1, n, p, f), p/w)
            self.assertAlmostEqual(schedule(w, n, p, f), p)
            self.assertAlmostEqual(schedule(n, n, p, f), p*.1)

    def test_shared_bootstrap_and_zero_population(self):
        ids = ["a", "b", "c"]
        a = np.array([[1., 1.], [2., 1.], [4., 1.]])
        b = a.copy(); b[:, 0] += 1
        r = bootstrap([(ids, a, b)]*3, replicates=100)
        self.assertAlmostEqual(r["upper95"], -1)
        self.assertTrue(r["favorable_all_seeds"])


class EndToEndTests(Temporary):
    device = "cpu"

    def test_offline_pipeline_tiny(self):
        c = self.corpus()
        conf = self.root/"config"
        tiny_config().save_pretrained(conf)
        def train_args(phase, arm, output):
            args = parser().parse_args(["train", "--data", str(c.path), "--output", str(output),
                     "--phase", phase, "--arm", arm, "--device", self.device, "--engineering",
                     "--microbatch-segments", "4", "--loss-chunk", "16"])
            args.source_code_hash = "test"
            args.model_config = str(conf)
            return args
        a = train_args("common", "base", self.root/"common")
        train(a, c)
        theta = self.root/"common/checkpoint-8"
        v = count_vocabulary(c.segments("compile"), [0], 2, self.root/"count.sqlite",
                             dict(corpus_hash=c.meta["manifest_hash"]))
        cov = self.root/"coverage.json"
        self.assertTrue(coverage(c, v, cov)["passed"])
        compile_args = SimpleNamespace(checkpoint=str(theta), device=self.device, output=str(self.root/"tables"),
                        microbatch_segments=4, isolated_batch=2, source_code_hash="test")
        compile_tables(compile_args, c, v)
        context, cm = load_bundle(self.root/"tables/contextual")
        shallow, _ = load_bundle(self.root/"tables/shallow")
        delta, _ = load_bundle(self.root/"tables/delta")
        self.assertTrue(torch.allclose(delta["master_sum"], context["master_sum"]-shallow["master_sum"], atol=1e-5))
        shuffled, _ = load_bundle(self.root/"tables/shuffled")
        self.assertTrue(torch.equal(shuffled["lookup"], context["lookup"][shuffled["permutation"]]))
        iso, _ = load_bundle(self.root/"tables/isolated")
        model, _ = load_model(theta, self.device)
        model.set_phase("compile")
        ids = torch.tensor([[int(v.keys[0]) >> 32, int(v.keys[0]) & 0xffffffff]], device=self.device)
        with torch.no_grad():
            direct = model(ids, torch.ones_like(ids), torch.tensor([[0, 1]], device=self.device), capture=True)
        self.assertTrue(torch.equal(iso["lookup"][0], direct["r12_pre_final_norm"][0, 1].bfloat16().cpu()))
        adapted_hash = None
        stage2 = []
        for phase, arm in (("stage1", "shallow"), ("stage1", "delta"), ("stage1", "isolated"),
                           ("stage1", "shuffled"), ("stage1", "grad"), ("stage1", "contextual"),
                           ("stage2", "contextual"), ("stage2", "grad"),
                           ("stage2", "base"), ("stage2", "isolated"), ("stage2", "shuffled")):
            out = self.root/f"{phase}-{arm}"
            args = train_args(phase, arm, out)
            args.checkpoint, args.coverage = str(theta), str(cov)
            args.table = str(self.root/"tables"/arm) if arm not in ("base", "grad") else None
            train(args, c, v)
            ckpt = out/"checkpoint-2"
            m, meta = load_model(ckpt)
            if phase == "stage1":
                adapted_hash = meta["reader_hash"]
            elif arm == "contextual":
                self.assertNotEqual(meta["paired_initial_reader_hash"], adapted_hash)
            ev = SimpleNamespace(checkpoint=str(ckpt), device=self.device, role="dev", output=str(out/"eval"),
                final_evaluation=False, final_lock=None, diagnostic_table=str(self.root/"tables/contextual"),
                microbatch_segments=4, loss_chunk=16)
            metrics = evaluate(ev, c, v)
            total = metrics["metrics"]
            self.assertEqual(total["overall"]["count"], total["hit"]["count"]+total["miss"]["count"])
            self.assertAlmostEqual(total["overall"]["loss_sum"], total["hit"]["loss_sum"]+total["miss"]["loss_sum"])
            ev.role = "val"
            with self.assertRaisesRegex(ValueError, "Locked D_val"):
                evaluate(ev, c, v)
            if phase == "stage2":
                stage2.append((arm, ckpt))
        locked = self.root/"final-lock.json"
        lock_final(SimpleNamespace(confirm_choices_locked=True, cluster="content_hash", cross_seed_coupling="shared",
                   checkpoints=[str(p) for _, p in stage2], include_delta=False, engineering=True, output=str(locked)))
        final_paths = {}
        for arm, ckpt in stage2:
            out = self.root/f"final-{arm}"
            ev = SimpleNamespace(checkpoint=str(ckpt), device=self.device, role="val", output=str(out),
                    final_evaluation=True, final_lock=str(locked), diagnostic_table=str(self.root/"tables/contextual"),
                    microbatch_segments=4, loss_chunk=16)
            evaluate(ev, c, v)
            final_paths[arm] = str(out)
            ev.output = str(self.root/f"final-repeated-{arm}")
            with self.assertRaises(FileExistsError):
                evaluate(ev, c, v)
        result = compare(SimpleNamespace(left=[final_paths["contextual"]], right=[final_paths["isolated"]],
                            population="overall", cluster="content_hash", cross_seed_coupling="shared",
                            replicates=50, final_report=True, output=str(self.root/"held-out-report.json")))
        self.assertTrue(result["final_report"])


if __name__ == "__main__":
    unittest.main()
