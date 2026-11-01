"""One parent per worker; all twelve physical six-layer removals, no training."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from efficient_llm.a2.data import PackedRows
from efficient_llm.a2.model import load_parent, round_to_saved_precision
from efficient_llm.a2.measure import score
from efficient_llm.a2_depth.common import read, write, Checks, removed_layers, layer_sets, SETS
from scripts.a2_remaining import read_locked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--parent', required=True)
    a = p.parse_args()
    cfg = read(a.config)
    output = Path(cfg['results_root'])/'part_a'/a.dataset/a.parent
    output.mkdir(parents=True, exist_ok=False)
    checks, started = Checks(output/'tests.json'), time.monotonic()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = checks.attempt('final_input', lambda: PackedRows(cfg['data_root'], 'final', verify=False))
    assets = checks.attempt('assets_identity', lambda: read_locked(cfg['assets']))
    actual_sets = checks.attempt('phase0_sets_identity', lambda: read_locked(cfg['phase0_sets']))
    if actual_sets is not None:
        checks.record('DT3', layer_sets(actual_sets['sets'])[a.dataset] == cfg['layer_sets'][a.dataset],
                      dict(recorded=cfg['phase0_sets'], actual=actual_sets['sets'][a.dataset]))
    model = checks.attempt('load_parent', lambda: load_parent(assets, a.dataset, a.parent, 'cuda:0').eval().requires_grad_(False))
    completed = []
    if model is not None and rows is not None:
        checks.attempt('bf16_saved_precision', lambda: round_to_saved_precision(model))
        def evaluate(name, kept):
            result = score(model, rows, distributed=False)
            result.update(loss=float(np.mean(result['per_sequence_nll'])), dataset=a.dataset,
                          parent=a.parent, layer_set=name, kept_original_indices=kept,
                          data_manifest=cfg['data_manifest'], assets=cfg['assets'],
                          code_commit=cfg['code_commit'], source_manifest=cfg['source_manifest'])
            write(output/(name+'.json'), result)
            completed.append(name)
            write(output/'progress.json', dict(completed=completed, elapsed_seconds=time.monotonic()-started))
            print('DEPTH_A_VAL', a.dataset, a.parent, name, result['loss'], flush=True)
            return result
        baseline = checks.attempt('parent_evaluation', lambda: evaluate('parent', list(range(24))))
        reference = checks.attempt('Q_parent_identity', lambda: read_locked(cfg['reference_nll'][a.dataset+'/'+a.parent+'/parent']))
        if baseline is not None and reference is not None:
            diff = baseline['loss'] - float(np.mean(reference['per_sequence_nll']))
            checks.record('parent_vs_Q', abs(diff) <= 1e-6, dict(difference=diff))
        for name in SETS:
            def evaluate_set(name=name):
                removed = cfg['layer_sets'][a.dataset][name]
                with removed_layers(model, removed) as kept:
                    actual = [l.self_attn.layer_idx for l in model.model.layers]
                    checks.record('DT2:'+name, actual == kept == [i for i in range(24) if i not in removed]
                                  and len(actual) == 18 and 0 in actual and 1 in actual,
                                  dict(removed=removed, kept_original_indices=actual))
                    result = evaluate(name, actual)
                if name == 'W8' and baseline is not None and reference is not None:
                    ref = read_locked(cfg['reference_nll'][a.dataset+'/'+a.parent+'/pre-S1'])
                    delta = result['loss']-baseline['loss']
                    ref_delta = float(np.mean(ref['per_sequence_nll']))-float(np.mean(reference['per_sequence_nll']))
                    checks.record('DT1', abs(delta-ref_delta) <= 1e-6,
                                  dict(damage=delta, Q_damage=ref_delta, difference=delta-ref_delta))
            checks.attempt('evaluation:'+name, evaluate_set)
    flags = [x for x in checks.items if not x['passed']]
    for name in completed:
        path = output/(name+'.json')
        write(path, dict(read(path), flags=flags, unverified=bool(flags)))
    write(output/'complete.json', dict(execution_complete=True, completed=completed,
          coverage_complete=set(completed) == set(SETS)|{'parent'}, flags=flags,
          elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
