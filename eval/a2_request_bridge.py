"""CPU-only bridge between canonical lm-eval tasks and native A2 scoring.

Capture stops before task metrics are computed: no placeholder evaluation result
is published. Replay requires identical text/token requests and complete finite
scores in their original order. GPU scoring lives in a2_likelihood, in a2_env.
"""
import hashlib
import json
import math
from importlib.metadata import version


class RequestsCaptured(Exception):
    """Intentional end of request preparation, before benchmark aggregation."""


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def request_rows(requests):
    return [dict(text=list(text),context=list(context),continuation=list(continuation))
            for text,context,continuation in requests]


def make_bridge(tokenizer, *, prepared=None, scored=None):
    """No model is loaded. Caller runs the same fixed task configuration twice."""
    from lm_eval.api.model import TemplateLM

    if version('lm-eval')!='0.4.13' or tokenizer.eos_token_id!=0:
        raise ValueError('Require pinned harness and verified GPT-NeoX EOS0 tokenizer')
    if (prepared is None)!=(scored is None):
        raise ValueError('Replay needs both prepared requests and scores')
    if prepared is not None:
        if (prepared.get('schema')!=1 or prepared.get('request_sha256')!=digest(prepared['requests'])
                or scored.get('request_sha256')!=prepared['request_sha256']
                or len(scored['scores'])!=len(prepared['requests'])):
            raise ValueError('Score/request identity or coverage mismatch')
        for score in scored['scores']:
            if (len(score)!=2 or type(score[0]) not in (int,float)
                    or not math.isfinite(score[0]) or score[0]>0
                    or type(score[1]) is not bool):
                raise ValueError('Invalid likelihood/greedy result')

    class RequestBridge(TemplateLM):
        backend='causal'

        def __init__(self):
            super().__init__()
            self.tokenizer=tokenizer
            self.captured=None
            self.calls=0

        @property
        def eot_token_id(self): return 0

        @property
        def tokenizer_name(self): return 'a2-pinned-gpt-neox'

        def tok_encode(self,string,**kwargs):
            if kwargs.get('add_special_tokens',False):
                raise ValueError('A2 likelihood does not add BOS/chat tokens')
            return tokenizer.encode(string,add_special_tokens=False,truncation=False)

        def _loglikelihood_tokens(self,requests,**kwargs):
            self.calls+=1
            if self.calls!=1:raise ValueError('Expected one canonical likelihood request collection')
            rows=request_rows(requests)
            if prepared is None:
                if not rows or any(not row['context'] or not row['continuation'] for row in rows):
                    raise ValueError('Empty likelihood request collection or token sequence')
                self.captured=dict(schema=1,requests=rows,request_sha256=digest(rows))
                raise RequestsCaptured()
            if rows!=prepared['requests']:
                raise ValueError('Task text, tokenization or request order changed before replay')
            return [tuple(value) for value in scored['scores']]

        def loglikelihood_rolling(self,*args,**kwargs):
            raise ValueError('Only the fixed conditional-likelihood benchmarks are supported')

        def generate_until(self,*args,**kwargs):
            raise ValueError('Generation is outside this benchmark screen')

    return RequestBridge()
