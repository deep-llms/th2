"""Variable-shape evaluation keeps the trained functions without Dynamo limits."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from deep_kv.proxy import ProxyModel, ProxySettings
from eval.models import EvaluationModel, load_checkpoint
from tests.test_proxy_memory import config
from tests.test_p7_speed_trial import reference_kernel


def model(compile_execution=True):
    return ProxyModel.from_scratch(config(), 'P7-simple-targetgrad', consumer=2, deep_target=8,
        proxy_settings=ProxySettings(width=8, features=9, chunk_size=2, compile_entries=True,
            compile_auxiliary=True, lse_attention=True, fused_targets=True),
        compile_execution=compile_execution, checkpoint_layers=False, checkpoint_lm=False,
        checkpoint_aux=False, lm_chunk=3).eval()


class CompilePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_loader_never_compiles_and_accepts_many_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ck=root/'checkpoint-7';ck.mkdir()
            m=model(False)
            recipe=dict(model_config=config().to_dict(),model=dict(tokenizer_name='local'),
                data=dict(block_size=32),training=dict(seed=42,bf16=False),
                pilot=dict(arm=m.arm,proxy_screen=True,consumer=2,deep_target=8,lm_chunk=3,
                    causal_attention=False,attention_backend='sdpa',
                    **{'proxy_'+k:v for k,v in asdict(m.settings).items()}))
            (root/'train_config.json').write_text(json.dumps(recipe))
            (ck/'trainer_state.json').write_text('{"global_step":7}')
            save_file({k:v.clone() for k,v in m.state_dict().items()},ck/'model.safetensors')
            with patch('torch.compile',side_effect=AssertionError('Evaluation must not compile')):
                loaded,_,metadata=load_checkpoint(ck)
                self.assertFalse(metadata['torch_compile'])
                self.assertEqual(loaded.wrapped.settings,m.settings)
                for n in range(1,15):
                    ids=torch.arange(n).remainder(20).unsqueeze(0)
                    expected=EvaluationModel(m,32)(ids).logits
                    actual=loaded(ids).logits
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for key,value in m.state_dict().items():
                torch.testing.assert_close(loaded.wrapped.state_dict()[key],value,rtol=0,atol=0)

    def test_eager_lse_matches_compiled_functions(self):
        torch._dynamo.reset()
        real_compile=torch.compile
        def compile_aot(fn,**kwargs):return real_compile(fn,backend='aot_eager',**kwargs)
        with patch('torch.compile',side_effect=compile_aot) as compiler:
            compiled=model()  # The training default still creates its wrappers.
        self.assertGreater(compiler.call_count,1)
        eager=model(False);eager.load_state_dict(compiled.state_dict())
        for m in (compiled,eager):
            m.attention_backend='fa4';m.fa4_kernel=reference_kernel
        with torch.no_grad():
            for n in (3,7,13):
                ids=torch.randint(0,20,(2,n))
                a=EvaluationModel(compiled,32)(ids).logits
                b=EvaluationModel(eager,32)(ids).logits
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        torch._dynamo.reset()


if __name__=='__main__':unittest.main()
