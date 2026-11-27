"""Eval-only depth extension probe for trained local DOT runs (L21 addendum, 27 Nov 2026).

Loads <base>/s<seed>/<arm>/terminal.pt and scores the pinned local measurement rows at the requested depths with the
same score() call fit() uses; the native depth is re-scored first and must reproduce summary.json's native NLL.
  python scripts/local_depth_probe.py --rows <rows dir> --base <run dir> --seed 0 --arm <arm> --depths 7 8 10 12 14
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--depths', type=int, nargs='+', required=True)
    a = p.parse_args()
    import torch
    from efficient_llm.warm_order.loop_screen import SPECS, make_model, native_path
    from efficient_llm.warm_order.model import Config
    from scripts.anchor_long_study import score
    from scripts.local_screen import load
    rows, ids, _ = load(a.rows)
    run = a.base/f's{a.seed}'/a.arm
    state = torch.load(run/'terminal.pt', map_location='cpu', weights_only=False)
    if state['arm'] != a.arm:
        raise ValueError('Checkpoint identity mismatch')
    model = make_model(Config(), a.arm)
    model.load_state_dict(state['model'])
    model = model.to('cuda').eval()
    model.passes = SPECS[a.arm].get('passes', 3)
    path = native_path(a.arm)
    want = json.loads((run/'summary.json').read_text())['nll']['native']
    got = float(score(model, rows, ids['measurement'], {'v': dict(path=path, skip=None)}, 'cuda')['v'].mean())
    if abs(got - want) > 1e-4:
        raise ValueError(f'Native NLL not reproduced: {got} vs {want}')
    out = dict(arm=a.arm, seed=a.seed, native=got, native_depth=SPECS[a.arm]['sandwich']['depth'], depths={})
    for d in a.depths:
        model.test_depth = d
        try:
            out['depths'][d] = float(score(model, rows, ids['measurement'], {'v': dict(path=path, skip=None)}, 'cuda')['v'].mean())
        finally:
            model.test_depth = None
    (run/'depth_probe.json').write_text(json.dumps(out, indent=1) + '\n')
    print('DEPTH_PROBE', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)


if __name__ == '__main__':
    main()
