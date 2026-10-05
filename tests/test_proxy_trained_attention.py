"""CPU checks for the production-checkpoint numerical diagnostic."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from deep_kv.proxy import ProxyModel
from deep_kv.packing import isolated_data_collator
from scripts.check_proxy_trained_attention import capture, model_from_run
from scripts.check_trained_attention import acceptable, fingerprint
from scripts.benchmark_document_training import compare
from tests.test_deep_kv import config


class ProxyTrainedAttentionTests(unittest.TestCase):
    def test_production_restore_and_capture_preserve_weights(self):
        torch.set_num_threads(1)
        cfg = config()
        cfg.num_hidden_layers = 28
        cfg.layer_types = ['full_attention'] * 28
        cfg.num_attention_heads = 8
        cfg.num_key_value_heads = 4
        model = ProxyModel.from_scratch(cfg, 'A', seed=42, checkpoint_layers=False,
            checkpoint_lm=False, checkpoint_aux=False, lm_chunk=3)
        model.backbone.lm_head.weight = model.backbone.model.embed_tokens.weight
        with torch.no_grad():
            model.backbone.model.embed_tokens.weight.add_(.01)
        before = fingerprint(model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'checkpoint-2500').mkdir()
            (root / 'train_config.json').write_text(json.dumps(dict(model_config=cfg.to_dict(),
                training=dict(seed=42), pilot=dict(lm_chunk=3))))
            save_file({k: v.clone() for k, v in model.state_dict().items()},
                      str(root / 'checkpoint-2500/model.safetensors'))
            restored = model_from_run(root)
        self.assertEqual(fingerprint(restored), before)
        rows = [dict(input_ids=[3,4,31,8,9,10,11,31], labels=[3,4,31,8,9,10,11,31],
                     attention_mask=[1]*8, segments=[0]*3+[1]*5)]
        inputs = isolated_data_collator(rows)
        a = capture(restored, inputs, 'sdpa', fp32=True)
        b = capture(restored, inputs, 'sdpa', fp32=True)
        self.assertEqual(a['targets'], 6)
        self.assertTrue(acceptable(compare(a, b, True)))
        self.assertEqual(fingerprint(restored), before)
        self.assertGreater(sum(float(g.square().sum()) for g in a['grads'].values()), 0.)


if __name__ == '__main__':
    unittest.main()
