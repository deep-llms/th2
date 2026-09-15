"""28-layer (tiny width) acceptance tests and real local end-to-end paths."""
from dataclasses import replace
import itertools
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
import pytest
from unittest.mock import patch
from test_pilot import Temporary, ModelTests, Tokenizer, tiny_config
from ccm.artifacts import load_bundle, state_hash
from ccm.cli import parser, code_hash
from ccm.contracts import PILOT, schedule, read_json, write_json
from ccm.data import collate
from ccm.studies import SCALEUP, FOLLOWUP, SCALEUP_BUDGET, open_corpus, require_vocabulary
from ccm.scaleup_data import prepare_scaleup, validate_scaleup
from ccm.keys import count_vocabulary, Vocabulary
from ccm.model import MemoryLM
from ccm.compiler import coverage, compile_tables
from ccm.runtime import train, load_model
from ccm.evaluation import evaluate
from ccm.decisions import lock_final
from ccm.distributed_compiler import distributed_compile, compare_accumulators, check_lookup_nll
from ccm.scaleup_protocol import stability_record, validate_stability_inputs
from ccm.scaleup_jobs import make_jobs, validate_run


def config28():
    c = tiny_config()
    c.num_hidden_layers = c.max_window_layers = 28
    c.layer_types = ["full_attention"]*28
    return c


class Model28Tests(ModelTests):
    # Re-run every model acceptance test at 28 blocks, not only an extra hook.
    def setUp(self):
        super().setUp()
        self.base = MemoryLM(config28())

    def model(self, arm="contextual"):
        m = MemoryLM(config28(), arm, table=self.table, slots=3)
        m.backbone.load_state_dict(self.base.backbone.state_dict())
        return m

    def test_explicit_28_endpoint(self):
        m = self.model()
        saved = {}
        handle = m.backbone.model.norm.register_forward_pre_hook(lambda _, args: saved.update(value=args[0]))
        r = self.run_model(m, capture=True)
        self.assertTrue(torch.equal(r["r28_pre_final_norm"], saved["value"]))
        self.assertNotIn("r12_pre_final_norm", r)
        handle.remove()


class ScaleupTests(Temporary):
    def fixture(self):
        historical = self.corpus(compile_tokens=70, duplicates=True)
        vocab = count_vocabulary(historical.segments("compile"), [0], 2, self.root/"counts.sqlite",
                                dict(corpus_hash=historical.meta["manifest_hash"]))
        vocab.save(self.root/"vocabulary.npz")
        b = replace(historical.budget, common_steps=12, continue_steps=4)
        factory = lambda: ((f"doc{i}", f"document {i//2}") for i in range(5000))
        prepare_scaleup(factory, Tokenizer(), historical, vocab, self.root/"extended",
                        historical.meta["provenance"], budget=b, engineering=True)
        return historical, open_corpus(self.root/"extended"), vocab

    def train_args(self, corpus, phase, arm, name):
        args = parser().parse_args(["train", "--data", str(corpus.path), "--output", str(self.root/name),
            "--study", SCALEUP, "--phase", phase, "--arm", arm, "--device", "cpu", "--engineering",
            "--microbatch-segments", "4", "--loss-chunk", "16"])
        args.source_code_hash = code_hash()
        args.model_config = str(self.root/"config")
        return args

    def test_budget_and_schedule(self):
        b = SCALEUP_BUDGET
        self.assertEqual(b.common_steps*b.batch_tokens, 10000007168)
        self.assertEqual(b.continue_steps*b.batch_tokens, 1999896576)
        self.assertEqual((b.common_steps+b.continue_steps)*b.batch_tokens, 11999903744)
        for total, warm, peak in [(38147,763,3e-4),(7629,153,1.5e-4),(977,49,5e-4)]:
            fraction = .05 if total == 977 else .02
            self.assertAlmostEqual(schedule(1,total,peak,fraction),peak/warm)
            self.assertAlmostEqual(schedule(warm,total,peak,fraction),peak)
            self.assertAlmostEqual(schedule(total,total,peak,fraction),peak*.1)
        self.assertEqual(PILOT.common_steps,15259)

    def test_generated_queues_parse_and_stop_before_replication(self):
        from run_experiments import load_jobs
        for queue in ("common","panel","shallow12"):
            args=parser().parse_args(["scaleup-jobs","--queue",queue,"--data","/data","--vocabulary","/vocab.npz",
                "--checkpoint","/common/checkpoint-38147","--tables","/tables","--model-config","/assets",
                "--contextual-eval","/old-eval","--gpus","0","1","--output",str(self.root/f"{queue}.json")])
            config=make_jobs(args)
            write_json(args.output,config)
            jobs=load_jobs(Path(args.output))
            for job in jobs:
                index=job["argv"].index("ccm")
                parser().parse_args(job["argv"][index+1:])
                self.assertNotIn("--final-evaluation",job["argv"])
                self.assertNotIn("delta",job["argv"])
            train_jobs=[j for j in jobs if "train" in j["argv"]]
            def value(job,flag):
                return job["argv"][job["argv"].index(flag)+1]
            if queue=="common":
                # Decisions 1(c)/3a: pipeline smoke, past-warmup stability smoke,
                # sparse full-run checkpoints.
                self.assertEqual([j["name"] for j in train_jobs],["common-pipeline","common-stability","common-base"])
                self.assertEqual([value(j,"--stability-steps") for j in train_jobs[:2]],["32","1024"])
                self.assertIn("--stability-report",train_jobs[2]["argv"])
                self.assertEqual([value(j,"--save-every") for j in train_jobs],["32","1024","5000"])
            if queue=="panel":
                self.assertEqual(len(train_jobs),11)
                self.assertEqual(jobs[-1]["name"],"replication-decision")
                self.assertEqual(jobs[-1]["required_outputs"][0]["json_equals"],dict(automatic_launch=False))
                for j in train_jobs:
                    self.assertEqual(value(j,"--save-every"),"977" if value(j,"--phase")=="stage1" else "2000")
            if queue=="shallow12":
                # Historical 12L convention is unchanged by decision 3a.
                self.assertNotIn("--save-every",train_jobs[0]["argv"])

    def test_two_seed_posthoc_lock_is_explicit(self):
        arms=("base","contextual","isolated","shuffled","shallow","grad")
        metas={}
        for seed in (17,29):
            for arm in arms:
                metas[f"{seed}-{arm}"]=dict(study=FOLLOWUP if arm=="shallow" else "pilot12",
                    phase="stage2",arm=arm,seed=seed,step=3815,total_steps=3815,corpus_hash="old",
                    engineering=False,source_checkpoint_hash=f"common-{seed}",vocabulary_hash="vocab",
                    paired_initial_reader_hash=f"reader-{seed}",model_sha256=f"weights-{seed}-{arm}",
                    metadata_hash=f"meta-{seed}-{arm}",backbone_contract=dict(config=dict(num_hidden_layers=12)))
        args=SimpleNamespace(study=FOLLOWUP,confirm_choices_locked=True,cluster="doc_id",cross_seed_coupling="independent",
            checkpoints=list(metas),include_delta=False,engineering=False,output=str(self.root/"lock.json"))
        with patch("ccm.decisions.checkpoint_meta",side_effect=metas.__getitem__):
            lock=lock_final(args)
            self.assertEqual(lock["seeds"],[17,29])
            self.assertIn("Post-hoc",lock["protocol_amendment"])
            args.study="pilot12"; args.output=str(self.root/"bad.json")
            with self.assertRaises(ValueError): lock_final(args)

    def test_single_seed_scaleup_lock_needs_terminal_declaration(self):
        # Pre-launch decision 2(a): D_val_28 waits for the frozen replication
        # panel unless the study explicitly stops at seed 17.
        arms=("base","contextual","isolated","shuffled","shallow","grad")
        metas={f"17-{a}":dict(study=SCALEUP,phase="stage2",arm=a,seed=17,step=7629,total_steps=7629,
            corpus_hash="ext",engineering=False,source_checkpoint_hash="common-28",vocabulary_hash="vocab",
            paired_initial_reader_hash="reader-17",model_sha256=f"w-{a}",metadata_hash=f"m-{a}",
            backbone_contract=dict(config=dict(num_hidden_layers=28))) for a in arms}
        args=SimpleNamespace(study=SCALEUP,confirm_choices_locked=True,cluster="doc_id",
            cross_seed_coupling="independent",checkpoints=list(metas),include_delta=False,
            engineering=False,single_seed_terminal=False,output=str(self.root/"s17.json"))
        with patch("ccm.decisions.checkpoint_meta",side_effect=metas.__getitem__):
            with self.assertRaisesRegex(ValueError,"single-seed-terminal"): lock_final(args)
            args.single_seed_terminal=True
            self.assertTrue(lock_final(args)["single_seed_terminal"])

    def test_compiler_gate_rejects_wrong_counts_or_mean(self):
        from ccm.artifacts import Accumulator
        a=Accumulator(2,3); b=Accumulator(2,3)
        for x in (a,b): x.add(torch.tensor([0,1]),torch.ones(2,3))
        b.counts[0]+=1
        with self.assertRaisesRegex(ValueError,"counts differ"): compare_accumulators({"contextual":a},{"contextual":b})
        b.counts[0]-=1; b.sums[0,0]+=1
        with self.assertRaisesRegex(ValueError,"means exceed"): compare_accumulators({"contextual":a},{"contextual":b})

    def test_lookup_gate_cannot_pass_on_all_misses(self):
        from ccm.artifacts import Accumulator
        old,new,v=self.fixture()
        writer=MemoryLM(config28())
        acc={"contextual":Accumulator(2,16)}
        with self.assertRaisesRegex(ValueError,"observed memory rows"):
            check_lookup_nll(writer,acc,acc,new,v,list(new.segments("dev")))

    def test_actual_historical_shallow_followup(self):
        old,_,v=self.fixture()
        tiny_config().save_pretrained(self.root/"config")
        args=self.train_args(old,"common","base","common12")
        args.study="pilot12"
        train(args,old)
        theta=self.root/"common12/checkpoint-8"
        ca=SimpleNamespace(checkpoint=str(theta),device="cpu",output=str(self.root/"tables12"),
                           microbatch_segments=4,isolated_batch=2,source_code_hash=code_hash())
        compile_tables(ca,old,v)
        cov=self.root/"coverage12.json"
        coverage(old,v,cov)
        args=self.train_args(old,"stage2","shallow","shallow12")
        args.study=FOLLOWUP; args.checkpoint=str(theta); args.coverage=str(cov)
        args.table=str(self.root/"tables12/shallow")
        train(args,old,v)
        m,meta=load_model(self.root/"shallow12/checkpoint-2")
        self.assertEqual(meta["study"],FOLLOWUP)
        self.assertEqual(meta["source_checkpoint_hash"],read_json(theta/"checkpoint.json")["model_sha256"])
        self.assertEqual(meta["paired_initial_reader_hash"],state_hash(MemoryLM(tiny_config(),"grad",slots=2).reader.state_dict()))
        args.study="pilot12"; args.output=str(self.root/"forbidden-shallow12")
        with self.assertRaisesRegex(ValueError,"diagnostic only"): train(args,old,v)

    def test_prefixes_disjointness_and_mapping(self):
        old, new, vocab = self.fixture()
        validate_scaleup(new)
        require_vocabulary(new, vocab)
        def signature(batch):
            return [(r["segment_id"], r["doc_id"], r["tokens"].tolist()) for r in batch]
        for seed in (1017,1029,1043):
            for phase, count in [("common",12),("stage1",2),("stage2",4)]:
                before = [signature(b) for b in old.optimizer_batches(phase,seed)]
                after = [signature(b) for b in new.optimizer_batches(phase,seed)]
                self.assertEqual(after[:len(before)],before)
                self.assertEqual(len(after),count)
                self.assertEqual(len(new.meta["optimizer_batch_orders"][str(seed)][phase]["historical"]),len(before))
        for role in ("compile","adapt","dev"):
            self.assertEqual(signature(list(old.segments(role))),signature(list(new.segments(role))))
        all_roles = {}
        for role in old.meta["quotas"]:
            for row in old.segments(role):
                all_roles[row["content_hash"]]=role
        for role in ("val28","continue_extension","common_extension"):
            for row in new.segments(role):
                self.assertEqual(all_roles.setdefault(row["content_hash"],role),role)
        self.assertEqual(signature(list(new.segments("val"))),signature(list(new.segments("val28"))))
        changed = Vocabulary(vocab.keys, vocab.counts+1, vocab.metadata)
        with self.assertRaisesRegex(ValueError,"exact historical"):
            require_vocabulary(new,changed)
        # Historical files remain byte/hash-identical after preparation.
        from ccm.data import Corpus
        Corpus(old.path)

    def test_stability_binding_and_no_completion(self):
        _, c, v = self.fixture()
        config28().save_pretrained(self.root/"config")
        args=self.train_args(c,"common","base","stability")
        args.stability_steps=2
        train(args,c)
        self.assertTrue((self.root/"stability/stability.json").exists())
        self.assertFalse((self.root/"stability/complete.json").exists())
        record=stability_record(args,c,True,2)
        args.engineering=False
        args.stability_steps=None
        args.stability_report=str(self.root/"proof.json")
        write_json(args.stability_report,record)
        with self.assertRaisesRegex(ValueError,"does not match"):
            validate_stability_inputs(args,c)

    def test_pipeline_and_distributed(self):
        _, c, v = self.fixture()
        config28().save_pretrained(self.root/"config")
        train(self.train_args(c,"common","base","common"),c)
        theta=self.root/"common/checkpoint-12"
        va=SimpleNamespace(run=str(theta.parent),phase="common",arm="base",seed=17,output=str(self.root/"common-verified.json"))
        self.assertTrue(validate_run(va,c)["success"])
        cov=self.root/"coverage.json"
        self.assertTrue(coverage(c,v,cov)["passed"])
        ca=parser().parse_args(["compile","--data",str(c.path),"--vocabulary",str(self.root/"vocabulary.npz"),
            "--checkpoint",str(theta),"--output",str(self.root/"reference"),"--device","cpu"])
        ca.source_code_hash=code_hash()
        compile_tables(ca,c,v)
        self.assertFalse((self.root/"reference/delta").exists())
        # Real two-process Gloo compilation, including a nonzero-reader NLL gate.
        prefix=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2","-m","ccm","compile",
            "--data",str(c.path),"--vocabulary",str(self.root/"vocabulary.npz"),"--checkpoint",str(theta),
            "--device","cpu","--microbatch-segments","2","--validation-batches","8","--validation-eval-batches","1"]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        for extra in (["--validate-distributed","--output",str(self.root/"gate")],
                      ["--distributed","--compiler-validation",str(self.root/"gate/validation.json"),"--output",str(self.root/"tables")]):
            p=subprocess.run(prefix+extra,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            self.assertEqual(p.returncode,0,p.stdout)
        self.assertTrue(read_json(self.root/"gate/validation.json")["success"])
        self.assertFalse((self.root/"tables/delta").exists())
        for kind in ("shallow","contextual","isolated","shuffled"):
            a,_=load_bundle(self.root/"reference"/kind); b,_=load_bundle(self.root/"tables"/kind)
            torch.testing.assert_close(a["lookup"].float(),b["lookup"].float(),atol=1e-4,rtol=1e-4)
        model,_=load_model(theta)
        model.set_phase("compile")
        key=int(v.keys[0]); ids=torch.tensor([[key>>32,key&0xffffffff]])
        with torch.no_grad():
            direct=model(ids,torch.ones_like(ids),torch.tensor([[0,1]]),capture=True)
        iso,_=load_bundle(self.root/"tables/isolated")
        self.assertTrue(torch.equal(iso["lookup"][0],direct["r28_pre_final_norm"][0,1].bfloat16()))
        panel=[]; initial=[]
        for phase in ("stage1","stage2"):
            arms=("shallow","isolated","contextual","shuffled","grad") if phase=="stage1" else ("base","contextual","isolated","shuffled","shallow","grad")
            for arm in arms:
                args=self.train_args(c,phase,arm,f"{phase}-{arm}")
                args.checkpoint=str(theta); args.coverage=str(cov)
                args.table=str(self.root/"tables"/arm) if arm not in ("base","grad") else None
                train(args,c,v)
                step=2 if phase=="stage1" else 4
                ck=Path(args.output)/f"checkpoint-{step}"
                meta=read_json(ck/"checkpoint.json")
                if arm!="base": initial.append(meta["paired_initial_reader_hash"])
                ev=SimpleNamespace(checkpoint=str(ck),device="cpu",role="dev",output=str(Path(args.output)/"eval"),
                    final_evaluation=False,final_lock=None,diagnostic_table=str(self.root/"tables/contextual"),
                    microbatch_segments=4,loss_chunk=16)
                evaluate(ev,c,v)
                if phase=="stage2": panel.append(str(ck))
        self.assertEqual(len(set(initial)),1)
        lock=self.root/"final.json"
        la=SimpleNamespace(study=SCALEUP,confirm_choices_locked=True,cluster="doc_id",cross_seed_coupling="independent",
             checkpoints=panel,include_delta=False,engineering=True,output=str(lock))
        lock_final(la)
        ev.checkpoint=panel[-1]; ev.role="val"; ev.final_evaluation=True; ev.final_lock=str(lock); ev.output=str(self.root/"val28")
        result=evaluate(ev,c,v)
        self.assertEqual(result["input_tokens"],c.budget.val_tokens)
        self.assertTrue(all(r.startswith('{"doc_id"') for r in (self.root/"val28/segments.jsonl").read_text().splitlines()))
        ev.output=str(self.root/"val28-repeat")
        with self.assertRaises(FileExistsError): evaluate(ev,c,v)
        args=self.train_args(c,"stage2","delta","bad-delta")
        with self.assertRaisesRegex(ValueError,"Delta"): train(args,c,v)
