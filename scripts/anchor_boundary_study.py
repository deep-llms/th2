"""Frozen CPU evaluation at the 1->2 boundary of path 0123 on original anchor fits."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from efficient_llm.warm_order.embedding_anchor import EmbeddingAnchor
from efficient_llm.warm_order.embedding_anchor_data import BoundRows
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.model import Config

FIT = Path('/mnt/local/_outputs/deep-llms_th2/embedding_anchor_fit_20261122_a01')
MANIFESTS = ('3408b58aa1befdb875f4fd44caf1d4aac88f68a9247b3cd9526fe45543718402',
             'ae345108c0237d2ebf4bbcb87256a22b0828ae66b7f8ca19c6d07c387d0d3bd9')
CASES = {'native': (.5, .5), 'remove_embedding': (.5, 0.), 'restore_hidden': (1., .5)}
PATH = (0, 1, 2, 3)


@torch.inference_mode()
def measure(model, tokens, case):
    h_scale, e_scale = CASES[case]
    saved = {}

    def capture_embedding(module, args, output):
        saved['embedding'] = output

    def capture_hidden(module, args, output):
        saved['hidden'] = output

    def replace(module, args):
        h, e = saved['hidden'], saved['embedding']
        boundary = args[0] if case == 'native' else h_scale * h + e_scale * e
        saved['rms'] = boundary.square().mean(-1).sqrt().mean(-1)
        return (boundary, *args[1:])

    with ExitStack() as hooks:
        hooks.enter_context(model.embed.register_forward_hook(capture_embedding))
        hooks.enter_context(model.blocks[1].register_forward_hook(capture_hidden))
        hooks.enter_context(model.blocks[2].register_forward_pre_hook(replace))
        loss = model.token_losses(tokens, PATH, alpha=.5)
    return loss.double().mean(-1).numpy(), saved['rms'].numpy()


def run(seed, output):
    directory = FIT / f's{seed}' / 'evaluation'
    manifest = directory / 'manifest.json'
    if sha(manifest) != MANIFESTS[seed]:
        raise ValueError('Prior verified manifest mismatch')
    meta = json.loads(manifest.read_text())
    proof = meta['checkpoint_proofs']
    pin = proof['arms']['A']['checkpoint']
    if sha(pin['path']) != pin['sha256'] or sha(directory/'scores.npz') != meta['files']['scores.npz']:
        raise ValueError('Checkpoint or scores changed')
    data = BoundRows(meta['data_binding']['prior_recipe_pin'], seed=seed)
    if data.binding != meta['data_binding'] or data.measurement_ids.tolist() != meta['row_ids']:
        raise ValueError('Measurement binding changed')
    state = torch.load(pin['path'], map_location='cpu', weights_only=False)
    if state['arm'] != 'A' or state['step'] != 1920:
        raise ValueError('Unexpected endpoint')
    model = EmbeddingAnchor(Config(**proof['model_config'])).eval()
    model.load_state_dict(state['model'])
    del state
    ids = data.measurement_ids
    with np.load(directory/'scores.npz') as prior:
        np.testing.assert_array_equal(ids, prior['row_ids'])
        reference = prior['A/path0123'] / prior['token_count']
    score = {k: [] for k in CASES}
    rms = {k: [] for k in CASES}
    for pos in range(0, len(ids), 2):
        tokens = data.batch(ids[pos:pos+2])
        for name in CASES:
            nll, norm = measure(model, tokens, name)
            if not np.isfinite(nll).all() or not np.isfinite(norm).all():
                raise ValueError('Nonfinite measurement')
            if name == 'native':
                np.testing.assert_allclose(nll, reference[pos:pos+2], atol=1e-4, rtol=0)
            score[name].extend(nll.tolist())
            rms[name].extend(norm.tolist())
    raw = output / f's{seed}.npz'
    with raw.open('xb') as f:
        np.savez_compressed(f, row_ids=ids, **score, **{'rms_'+k:v for k,v in rms.items()})
    return dict(seed=seed, checkpoint=pin['sha256'], rows=len(ids), path=PATH,
                raw_sha256=sha(raw), means={k:float(np.mean(v)) for k,v in score.items()},
                delta_nll={k:float(np.mean(np.subtract(v,score['native']))) for k,v in score.items()},
                mean_token_rms={k:float(np.mean(v)) for k,v in rms.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    panels = [run(seed,args.out) for seed in range(2)]
    result = json.dumps(panels,allow_nan=False)
    (args.out/'report.json').write_text(result+'\n')
    print('ANCHOR_BOUNDARY_COMPLETE', result, flush=True)
