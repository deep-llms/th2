import copy
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from eval.a2_harness import make_harness
from efficient_llm.a2.model import prune, save_student, reload_student
from test_model import tiny_model


def tokenizer():
    vocab={'<eos>':0, '<unk>':1, **{f'w{i}':i+2 for i in range(30)}}
    backend=Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', unk_token='<unk>')


class NativeHarnessTests(unittest.TestCase):
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
