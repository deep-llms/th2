import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from eval.a2_harness import make_harness
from eval.a2_likelihood import score_requests
from efficient_llm.a2.model import prune, save_student, reload_student
from test_model import tiny_model


def tokenizer():
    vocab={'<eos>':0, '<unk>':1, **{f'w{i}':i+2 for i in range(30)}}
    backend=Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', unk_token='<unk>')


class NativeHarnessTests(unittest.TestCase):
    def test_native_scorer_matches_harness_without_harness_runtime(self):
        model=tiny_model()
        rows=[dict(context=[2,3],continuation=[4,5]),
              dict(context=[],continuation=[6]),
              dict(context=[2],continuation=[3,4,5]),
              dict(context=[2,3,0,4,5,6],continuation=[7,8])]
        lm=make_harness(model,tokenizer(),batch_size=3,max_length=4)
        reference=lm._loglikelihood_tokens(
            [((str(i),'answer'),r['context'] or [0],r['continuation']) for i,r in enumerate(rows)],
            disable_tqdm=True)
        for batch_size in (1,3):
            got=score_requests(model,rows,batch_size=batch_size,max_length=4)
            self.assertEqual(got['truncated_requests'],1)
            for a,b in zip(got['scores'],reference):
                self.assertAlmostEqual(a[0],b[0],places=5)
                self.assertEqual(a[1],b[1])
        for invalid in ([dict(context=[-1],continuation=[2])],
                        [dict(context=[2],continuation=[])],
                        [dict(context=[2],continuation=[3]*5)]):
            with self.assertRaises(ValueError):score_requests(model,invalid,max_length=4)

    def test_public_text_requests_with_pinned_neox_tokenizer(self):
        from lm_eval.api.instance import Instance
        location=os.environ.get('A2_NEOX_TOKENIZER')
        if not location:
            self.skipTest('Set A2_NEOX_TOKENIZER to the existing pinned local tokenizer')
        manifest=json.loads(Path('resources/a2_inputs_20261028.json').read_text())
        for name,digest in manifest['tokenizer_provenance']['gpt_neox']['files'].items():
            self.assertEqual(hashlib.sha256(Path(location,name).read_bytes()).hexdigest(),digest)
        tok=AutoTokenizer.from_pretrained(location,local_files_only=True)
        model=tiny_model(vocab_size=50304)
        lm=make_harness(model,tok,max_length=64,batch_size=3)
        pairs=[('The capital of Singapore is',' Singapore.'),('', 'Hello.'),
               ('A small test ', 'has two parts.')]
        requests=[Instance(request_type='loglikelihood',doc={},arguments=pair,idx=i)
                  for i,pair in enumerate(pairs)]
        got=lm.loglikelihood(requests,disable_tqdm=True)
        for (context,answer),(value,_) in zip(pairs,got):
            # Harness's canonical convention moves trailing context whitespace
            # into the answer. Score independent native logits for those tokens.
            clean=context.rstrip();answer=context[len(clean):]+answer
            context_ids=tok.encode(clean,add_special_tokens=False)
            joined=tok.encode(clean+answer,add_special_tokens=False)
            target=joined[len(context_ids):]
            context_ids=context_ids or [0]
            ids=torch.tensor([context_ids+target])
            with torch.no_grad():
                logp=model(ids[:,:-1],use_cache=False).logits[0,len(context_ids)-1:].float().log_softmax(-1)
            wanted=logp.gather(1,torch.tensor(target)[:,None]).sum().item()
            self.assertAlmostEqual(value,wanted,places=4)
        self.assertEqual(lm.prefix_token_id,0)
        self.assertFalse(lm.add_bos_token)

    def test_likelihood_matches_manual_and_padding(self):
        model=tiny_model(); lm=make_harness(model,tokenizer(),batch_size=3,max_length=16)
        requests=[(('a','b'),[2,3],[4,5]), (('c','d'),[0],[6]), (('e','f'),[2],[3,4,5])]
        got=lm._loglikelihood_tokens(requests,disable_tqdm=True)
        for (_,context,target),(score,greedy) in zip(requests,got):
            ids=torch.tensor([context+target])
            with torch.no_grad():logits=model(ids[:,:-1],use_cache=False).logits[0].float()
            chosen=logits[len(context)-1:].log_softmax(-1)
            wanted=chosen.gather(1,torch.tensor(target)[:,None]).sum().item()
            self.assertAlmostEqual(score,wanted,places=5)
            self.assertEqual(greedy,bool(torch.equal(chosen.argmax(-1),torch.tensor(target))))
        single=make_harness(model,tokenizer(),batch_size=1,max_length=16)
        for a,b in zip(got,single._loglikelihood_tokens(requests,disable_tqdm=True)):
            self.assertAlmostEqual(a[0],b[0],places=5)
        self.assertEqual(lm.truncated_requests,0)

    def test_context_truncation_matches_manual(self):
        model=tiny_model();lm=make_harness(model,tokenizer(),max_length=4)
        context=[2,3,4,5,6]; target=[7,8]
        score,_=lm._loglikelihood_tokens([(('long','answer'),context,target)],disable_tqdm=True)[0]
        ids=torch.tensor([(context+target)[-5:][:-1]])
        with torch.no_grad():logits=model(ids,use_cache=False).logits[0,-2:].float()
        wanted=logits.log_softmax(-1).gather(1,torch.tensor(target)[:,None]).sum().item()
        self.assertAlmostEqual(score,wanted,places=5)
        self.assertEqual(lm.truncated_requests,1)

    def test_pruned_export_matches_reload(self):
        model=tiny_model();prune(model,[1,3])
        ids=torch.tensor([[2,3,0,4]])
        with tempfile.TemporaryDirectory() as tmp:
            save_student(model,tmp+'/final'); restored=reload_student(tmp+'/final')
            lm=make_harness(restored,tokenizer(),max_length=16)
            with torch.no_grad():expected=restored(ids,use_cache=False).logits
            torch.testing.assert_close(lm._model_call(ids),expected,rtol=0,atol=0)
            self.assertEqual(restored.config.a2_kept_layers,[0,2])

    def test_reject_wrong_precision_eos_and_generation(self):
        model=tiny_model()
        with self.assertRaises(ValueError):make_harness(copy.deepcopy(model).bfloat16(),tokenizer())
        bad=tokenizer();bad.eos_token='<unk>'
        with self.assertRaises(ValueError):make_harness(model,bad)
        lm=make_harness(model,tokenizer(),max_length=16)
        with self.assertRaises(ValueError):lm.generate_until([])
        with self.assertRaises(ValueError):lm._model_call(torch.tensor([[2]]),labels=torch.tensor([[2]]))


if __name__=='__main__':unittest.main()
