"""Three fixed two-way weight averages of existing own-parent fine-tunes.

No training, ingredient selection, BN recalibration, TTA or logit ensembling.
The supervisor waits for the staged queue's verified burn handoff first.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from deadline_study import run_child, save
from followup_study import PARENT_ROOT, PARENTS, EXTERNAL, verify_parents
from staged_study import average_states, digest

SRC = Path(__file__).resolve().parent
DIRECT = Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a02/supervised/run/study')
MODELS = ('grip_nodown', 'srresnet', 'srresnet_stats')
FINISH_BY = '2026-10-17T07:00:00+00:00'
# Filled from the previously downloaded and verified fixed-final receipts.
EXPECTED = {'grip_nodown': ['29090be74fddeb5a77e43795d8c798aef61f26332ff971883631e829dc58fb8c', '66debf7e7f22bef034be371f702ad1dc7eb5514233f0618f0f20d2aa6d710af4'], 'srresnet': ['46da5542ff9e7e9ff103db9cb8e4e8e3e7b24ed1506d778d62495b86b43b0571', 'f3face904542159ef64f7f42b7115227fffc2adf4d93fca4edd319add2881f6e'], 'srresnet_stats': ['e16f3c6c23b58e49cd7a4335144ca18a81faf203acbeba98a9a189f9bdfe93b6', '47a494f3cf3f5191a84c022000f4b956cedd4a8fe18aded2a0011ca2ab469ce3']}


def validate_args(ck, model, parent, seed, lr):
    expected = dict(model=model, train_split='train', val_frac=.1, split_seed=42,
                    crop=512, rrc_size=512, norm='imagenet', epochs=40,
                    init_from=str(parent), seed=seed, data_seed=42+seed,
                    lr=lr, optim='adam', sched='cosine', views=2, kd_teacher=None)
    if ck['epoch'] != 40 or any(ck['args'].get(k) != v for k, v in expected.items()):
        raise ValueError('Ingredient architecture, parent or recipe mismatch')


def verify_inputs():
    import torch
    torch.set_num_threads(1)
    parents = verify_parents()
    records = {}
    for model in MODELS:
        parent = Path(parents[model]['path'])
        paths = [PARENT_ROOT/(model+'_full512')/'last.pt', DIRECT/(model+'_s1')/'last.pt']
        for i, path in enumerate(paths):
            if digest(path) != EXPECTED[model][i]:
                raise ValueError('Fixed final ingredient hash mismatch: '+str(path))
            ck = torch.load(path, map_location='cpu', weights_only=False)
            validate_args(ck, model, parent, i, 1e-4 if i == 0 else 3e-5)
        records[model] = {'parent': parents[model], 'sources': list(map(str, paths)),
                          'sha256': EXPECTED[model], 'weights': [.5, .5]}
    ready = json.loads((EXTERNAL/'ready.json').read_text())
    if ready['status'] != 'passed' or ready['purpose'] != 'evaluation_only':
        raise ValueError('External evaluation inputs not verified')
    return records


def make_soup(model, record, dest):
    import torch
    from models import build_model, count_trainable
    if model not in MODELS:
        raise ValueError('Unapproved soup architecture')
    paths = list(map(Path, record['sources']))
    if len(paths) != 2 or record['weights'] != [.5, .5]:
        raise ValueError('Exactly two fixed equally weighted ingredients required')
    cks = []
    for i, path in enumerate(paths):
        if digest(path) != record['sha256'][i]:
            raise ValueError('Ingredient changed')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        validate_args(ck, model, Path(record['parent']['path']), i, 1e-4 if i == 0 else 3e-5)
        cks.append(ck)
    net = build_model(model).eval()
    for ck in cks:
        net.load_state_dict(ck['model'], strict=True)
    state = average_states(cks[0]['model'], cks[1]['model'])
    net.load_state_dict(state, strict=True)
    n = count_trainable(net)
    if n > 12_300_000 or any(ck['n_params_training'] != n for ck in cks):
        raise ValueError('Parameter count mismatch')
    info = {**record, 'bn_recalibrated': False, 'method': 'uniform weight average',
            'note': 'Inference-only. No inherited validation score or optimizer state.'}
    out = {'model': state, 'args': cks[0]['args'], 'epoch': 40, 'n_params': n,
           'n_params_training': n, 'weight_soup': info}
    with Path(dest).open('xb') as f:
        torch.save(out, f)
    restored = torch.load(dest, map_location='cpu', weights_only=False)
    net.load_state_dict(restored['model'], strict=True)
    for key in state:
        torch.testing.assert_close(state[key], restored['model'][key], rtol=0, atol=0)
    return {**info, 'parameters': n, 'checkpoint_sha256': digest(dest)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--cache', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    inputs = verify_inputs()
    if args.preflight_only:
        print('SOUP_INPUTS_VERIFIED', json.dumps(inputs), flush=True)
        return
    if not args.cache or not args.output or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('Expected supervisor allocation and explicit paths')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'manifest.json', {'inputs': inputs, 'fixed_models': MODELS,
         'weights': [.5, .5], 'finish_by': FINISH_BY, 'training': False,
         'selection': 'All three predefined compatible pairs; no dev/test selection'})
    # CPU averaging first; GPU jobs only perform the existing evaluator.
    import torch
    torch.set_num_threads(1)
    soups = {m: make_soup(m, inputs[m], args.output/(m+'_soup.pt')) for m in MODELS}
    deadline = datetime.fromisoformat(FINISH_BY).timestamp()
    def evaluate(item):
        gpu, model = item
        ck = args.output/(model+'_soup.pt')
        dest = args.output/(model+'_soup_evaluation')
        run_child([sys.executable, '-u', str(SRC/'evaluate_deadline.py'),
                   '--checkpoint', str(ck), '--cache', str(args.cache),
                   '--external-cache', str(EXTERNAL), '--output', str(dest)],
                  gpu, args.output/(model+'_evaluation.log'), deadline)
        r = json.loads((dest/'results.json').read_text())
        if r['status'] != 'passed' or r['checkpoint_sha256'] != soups[model]['checkpoint_sha256']:
            raise ValueError('Evaluation/checkpoint mismatch')
        ext = r['external_predictions']
        if ext['status'] != 'passed' or any(ext['splits'][s]['n'] != n for s,n in [('dev',900),('test_official',1500)]):
            raise ValueError('Incomplete external evaluation')
        return {'model':model, 'soup':soups[model], 'evaluation':r}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(evaluate, enumerate(MODELS)))
    if inputs != verify_inputs():
        raise ValueError('Source ingredients changed')
    save(args.output/'complete.json', {'status':'passed', 'results':results,
         'finished_utc':datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
