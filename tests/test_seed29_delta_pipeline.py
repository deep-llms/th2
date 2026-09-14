"""Tiny real CPU trainer/compiler test of the new own-seed Delta guard."""
from pathlib import Path
from types import SimpleNamespace

from test_pilot import Temporary, tiny_config
from ccm.cli import parser
from ccm.contracts import write_json
from ccm.keys import count_vocabulary
from ccm.compiler import coverage, compile_tables
from ccm.runtime import train, load_model


class OwnSeedDeltaPipeline(Temporary):
    def test_seed29_delta_fresh_stage2_and_wrong_seed_refusal(self):
        corpus=self.corpus()
        conf=self.root/'config';tiny_config().save_pretrained(conf)
        def args(phase,arm,name):
            a=parser().parse_args(['train','--data',str(corpus.path),'--output',str(self.root/name),
                '--phase',phase,'--arm',arm,'--seed','29','--device','cpu','--engineering',
                '--microbatch-segments','4','--loss-chunk','16'])
            a.source_code_hash='toy-test';a.model_config=str(conf)
            return a
        train(args('common','base','common'),corpus)
        theta=self.root/'common/checkpoint-8'
        _, common=load_model(theta)
        vocab=count_vocabulary(corpus.segments('compile'),[0],2,self.root/'counts.sqlite',
                              dict(corpus_hash=corpus.meta['manifest_hash']))
        cov=self.root/'coverage.json';self.assertTrue(coverage(corpus,vocab,cov)['passed'])
        compile_tables(SimpleNamespace(checkpoint=str(theta),device='cpu',output=str(self.root/'tables'),
            microbatch_segments=4,isolated_batch=2,source_code_hash='toy-test'),corpus,vocab)
        a=args('stage1','delta','stage1');a.checkpoint=str(theta);a.coverage=str(cov)
        a.table=str(self.root/'tables/delta');train(a,corpus,vocab)
        _, prior=load_model(self.root/'stage1/checkpoint-2')
        # Synthetic acceptance fixture, not a claim of a measured toy Delta win.
        gate=dict(include_delta=True,replication_policy='per_seed',decision_seed=29,
            source_checkpoint_hash=common['model_sha256'],corpus_hash=corpus.meta['manifest_hash'],
            vocabulary_hash=vocab.hash,overall_safeguard=True,
            results={k:dict(upper95=-.001,replicates=10000,bootstrap_seed=20260913)
                     for k in ('hit_vs_contextual','hit_vs_shuffled','miss_vs_contextual')})
        write_json(self.root/'gate.json',gate)
        a=args('stage2','delta','stage2');a.checkpoint=str(theta);a.coverage=str(cov)
        a.table=str(self.root/'tables/delta');a.delta_decision=str(self.root/'gate.json')
        train(a,corpus,vocab)
        _, final=load_model(self.root/'stage2/checkpoint-2')
        self.assertEqual(final['seed'],29)
        self.assertEqual(final['source_checkpoint_hash'],common['model_sha256'])
        self.assertEqual(final['paired_initial_reader_hash'],prior['paired_initial_reader_hash'])
        self.assertNotEqual(final['paired_initial_reader_hash'],prior['reader_hash'])
        gate['decision_seed']=17;write_json(self.root/'wrong_gate.json',gate)
        a.output=str(self.root/'rejected');a.delta_decision=str(self.root/'wrong_gate.json')
        with self.assertRaisesRegex(ValueError,'Invalid Delta inclusion'):
            train(a,corpus,vocab)
        self.assertFalse(Path(a.output).exists())
