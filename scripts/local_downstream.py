"""Zero-shot downstream for local runs (B200 outage, 26 Nov 2026), same protocol as loop_screen_downstream_study.

Inputs: the lm-eval requests rebuilt locally (bit-identical to the pinned B200 file, sha 6282100c...), and a raw
sample template made once by lm-eval replay (eval.a2_requests replay, train_env with lm-eval 0.4.13) from any
local model's scores; items() reads only the template's sample structure (docs, arguments, gold), never its scores.
Scoring: score_requests in the FP32 measurement runtime, as on the B200.
  python scripts/local_downstream.py score --base <run dir> --seed 0 --arm <arm> --requests <requests.json>
  python scripts/local_downstream.py template --scored <scores.json> --requests <requests.json> --out <raw.json>
  python scripts/local_downstream.py items --base <run dir> --seed 0 --arm <arm> --requests ... --raw <raw.json>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUESTS_SHA = '6282100c39545c6d14b219a4a6538e4ecf13b94a12b97c19c68c18c61a450091'
EVAL_PY = '/home/users/thien/miniconda3/envs/train_env/bin/python'
DATA = '/disk/thuat/dot_data_20261126'


def load_requests(path):
    from efficient_llm.warm_order.data import sha
    if sha(path) != REQUESTS_SHA:
        raise ValueError('Requests differ from the pinned B200 file')
    return json.loads(Path(path).read_text())


def score(base, seed, arm, requests):
    import torch
    from efficient_llm.warm_order.adaptation_runtime import measurement_runtime
    from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
    from efficient_llm.warm_order.loop_screen import SPECS, make_model, native_path
    from efficient_llm.warm_order.loop_screen_downstream import digest, score_requests
    from efficient_llm.warm_order.model import Config
    prepared = load_requests(requests)
    run = base/f's{seed}'/arm
    state = torch.load(run/'terminal.pt', map_location='cpu', weights_only=False)
    if state['arm'] != arm:
        raise ValueError('Checkpoint identity mismatch')
    model = make_model(Config(), arm)
    model.load_state_dict(state['model'])
    model = model.to('cuda').eval()
    model.passes = SPECS[arm].get('passes', 3)
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(4)
    with measurement_runtime('cuda'), anchor_runtime('cuda'):
        scores, meta = score_requests(model, native_path(arm), prepared['requests'], 'cuda')
    out = dict(prepared_sha256=digest(prepared), request_sha256=prepared['request_sha256'],
               max_length=prepared['protocol']['max_length'], precision='fp32_no_autocast_no_tf32',
               likelihood_requests=len(prepared['requests']), scores=scores, meta=meta)
    (run/'downstream_scores.json').write_text(json.dumps(out) + '\n')
    print('SCORED', seed, arm, json.dumps(meta), flush=True)


def template(scored, requests, out):
    load_requests(requests)
    cmd = [EVAL_PY, '-m', 'eval.a2_requests', 'replay', '--tokenizer', f'{DATA}/tokenizer', '--dataset-root',
           f'{DATA}/downstream', '--manifest', 'resources/warm_downstream_data_20261118.json', '--tasks',
           'hellaswag', 'piqa', 'arc_easy', 'lambada_openai', '--max-length', '512', '--prepared', str(requests),
           '--scored', str(scored), '--score-precision', 'fp32_no_autocast_no_tf32', '--output', str(out)]
    env = dict(__import__('os').environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL)
    print('TEMPLATE', out, flush=True)


def accuracy(base, seed, arm, requests, raw):
    from efficient_llm.warm_order.data import sha
    from efficient_llm.warm_order.loop_screen_downstream import items, save_items
    prepared = load_requests(requests)
    run = base/f's{seed}'/arm
    scores = json.loads((run/'downstream_scores.json').read_text())['scores']
    summary = save_items(items(json.loads(Path(raw).read_text()), prepared, scores), run/'downstream_items.npz')
    (run/'downstream_summary.json').write_text(json.dumps(dict(arm=arm, seed=seed, tasks=summary, raw_template=str(raw),
                                                               raw_template_sha256=sha(raw)), indent=1) + '\n')
    print('DOWNSTREAM', seed, arm, json.dumps({t: round(v['accuracy'], 4) for t, v in summary.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('score', 'template', 'items'))
    p.add_argument('--base', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--arm')
    p.add_argument('--requests', type=Path, default=Path(f'{DATA}/downstream/requests.json'))
    p.add_argument('--scored', type=Path)
    p.add_argument('--raw', type=Path, default=Path(f'{DATA}/downstream/raw_template.json'))
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.action == 'score':
        score(a.base, a.seed, a.arm, a.requests)
    elif a.action == 'template':
        template(a.scored, a.requests, a.out)
    else:
        accuracy(a.base, a.seed, a.arm, a.requests, a.raw)
