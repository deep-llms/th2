import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3ForCausalLM
from test_kv_mlp import tiny
from efficient_llm.kv_mlp import convert
from eval.kv_harness import make_harness


def tokenizer_assets(path):
    tok = Tokenizer(WordLevel({'[UNK]':0,'[EOS]':1,'a':2,'b':3,'c':4,'d':5}, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', eos_token='[EOS]',
                                 pad_token='[EOS]', model_max_length=128)
    tok.save_pretrained(path)
    return tok


class HarnessTests(unittest.TestCase):
    def test_dense_matches_native_harness(self):
        from lm_eval.api.instance import Instance
        from lm_eval.models.huggingface import HFLM
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); tok = tokenizer_assets(p); model = tiny(); model.save_pretrained(p)
            ref = Qwen3ForCausalLM._from_config(deepcopy(model.config), attn_implementation='sdpa').eval()
            ref.load_state_dict(model.state_dict(), strict=True)
            custom = make_harness(p, batch_size=2, max_length=16)
            native = HFLM(pretrained=ref, tokenizer=tok, backend='causal', device='cpu',
                          batch_size=2, max_length=16, softmax_dtype='float32')
            requests = [Instance(request_type='loglikelihood', doc={}, arguments=(c,t), idx=i)
                for i,(c,t) in enumerate([('a b',' c d'),('a',' b'),('',' a b'),('a '*24,' b c')])]
            for (a,ag),(b,bg) in zip(custom.loglikelihood(requests), native.loglikelihood(requests)):
                self.assertAlmostEqual(a,b,places=6); self.assertEqual(ag,bg)
            self.assertEqual(custom.truncated_requests, 1)

    def test_shared_reload_padding_and_manual_scores(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); tokenizer_assets(p); model=convert(tiny(),share=True); model.save_pretrained(p)
            h=make_harness(p,batch_size=2,max_length=16)
            before={k:v.clone() for k,v in h.model.state_dict().items()}
            ids=torch.tensor([[2,1,4,5],[3,4,2,1]])
            with torch.inference_mode(): expected=model(ids,use_cache=False).logits
            torch.testing.assert_close(h._model_call(ids),expected,atol=0,rtol=0)
            padded=torch.cat((ids,torch.zeros(2,3,dtype=torch.long)),dim=1)
            torch.testing.assert_close(h._model_call(padded)[:,:4],expected,atol=1e-6,rtol=1e-6)
            rows=ids.tolist(); observed=h._loglikelihood_tokens([(None,t[:2],t[2:]) for t in rows])
            logp=h._model_call(ids[:,:-1]).float().log_softmax(-1)
            for i,t in enumerate(rows):
                score=logp[i,1:].gather(-1,ids[i,2:,None]).sum().item()
                self.assertAlmostEqual(score,observed[i][0],places=6)
            for k,v in before.items():torch.testing.assert_close(h.model.state_dict()[k],v,atol=0,rtol=0)
            self.assertEqual(h.model.config.kv_sources,[0,0,2,2])
            with self.assertRaises(ValueError):h.generate_until([])


if __name__ == '__main__': unittest.main()
