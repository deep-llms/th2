"""Final-step diagnostic queue and strict stage validation; local inputs only."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from eval.runtime import offline
offline()
from capacity_allocation.data import sha256, write_json
from eval.diagnostic_data import load_bundle
from eval.diagnostics_checkpoint import training_provenance
from eval.eval_parallel import build_jobs
from eval.parallel import run
from eval.runtime import checkpoint_identity

ARMS = ('B0', 'A128', 'A256', 'A512', 'C', 'D')


def diagnostic_jobs(training, bundle, output, tokenizer):
    args = SimpleNamespace(
        checkpoints=[f'{arm}_step5000={training/arm/"checkpoint-5000"}' for arm in ARMS],
        languages='en', task_groups=['hellaswag'], dataset_root=str(training),
        tokenizer_name=str(tokenizer), precision='bf16', benchmark_batch_size=8,
        output_dir=str(output), eval_dir=str(bundle/'eval'), batch_size=1,
        preprocessing_num_workers=160, preprocessing_batch_size=1000,
        preprocessing_cache_dir=None, diagnostic_bundle=str(bundle),
        diagnostics=['frequency', 'spectra', 'gradients'])
    # Reuse the tested job builder without rerunning completed PPL/benchmarks.
    jobs = [job for job in build_jobs(args) if job['stage'] == 'diagnostics']
    assert len(jobs) == 18 and len({job['result'] for job in jobs}) == 18
    return jobs


def verify(stage, output, reference, manifest=None):
    complete = json.loads((output/'complete.json').read_text())
    expected = 18 if stage == 'diagnostics' else 54
    assert complete['success'] is True and len(complete['completed']) == expected
    identities = {item['path']: item for item in reference['checkpoints'].values()}
    seen = set()
    for record in complete['completed']:
        assert record['exit_code'] == 0 and record['error'] is None
        path = Path(record['result'])
        assert path.resolve().is_relative_to(output.resolve())
        item = json.loads(path.read_text())
        assert item['success'] is True and item['languages'] == ['en']
        assert item['checkpoint'] == identities[item['checkpoint']['path']]
        checkpoint = item['checkpoint']['path']
        assert Path(checkpoint).name == 'checkpoint-5000'
        if stage == 'diagnostics':
            assert item['diagnostic_manifest_sha256'] == reference['bundle_sha256']
            assert item['training']['status'] == 'verified' and item['training']['global_step'] == 5000
            assert item['training']['consumed_input_tokens'] == 5242880000
            assert item['settings']['precision'] == 'bf16'
            assert len(item['diagnostics']) == 1
            kind, result = next(iter(item['diagnostics'].items()))
            assert kind in ('frequency', 'spectra', 'gradients')
            key = (checkpoint, kind)
            if kind == 'frequency':
                row = result['by_language']['en']['overall']
                assert row['scored_targets'] == 9829694 and math.isfinite(row['nll'])
                assert sum(b['scored_targets'] for b in result['by_language']['en']['buckets'].values()) == row['scored_targets']
            elif kind == 'gradients':
                assert result['optimizer_steps'] == 0 and result['precision'] == 'fp32'
                assert result['probe_ids'] == manifest['probe_ids']
                assert result['scored_targets'] == 8*2047
                if Path(checkpoint).parent.name == 'B0':
                    paths = result['tied_paths']
                    assert paths['overall']['additivity_relative_l2'] <= paths['tolerance']
            else:
                assert set(result['interfaces']) == {'input', 'output'}
                for interface in result['interfaces'].values():
                    for view in ('raw', 'effective'):
                        assert set(interface[view]) == {'seen', 'all'}
                        assert all(r['status'] == 'ok' for r in interface[view].values())
        else:
            key = (checkpoint, item['task'], item['seed'])
            assert item['task'] in ('hellaswag', 'arc_easy', 'xnli')
            assert item['seed'] in (42, 123, 456) and item['train_language'] == 'en'
            assert item['precision'] == 'bf16' and item['master_weights'] == 'fp32'
            from finetune.tasks import TASK_CONFIGS
            from eval.benchmarks import task_plan
            plan, _ = task_plan('en', [item['task']])
            assert set(item['benchmarks']) == {row['task'] for row in plan}
            assert item['config'] == TASK_CONFIGS[item['task']]
            config = item['config']
            assert item['training']['steps'] == ((item['used_examples']+config['batch_size']-1)//config['batch_size'])*3
            assert item['used_examples'] + item['skipped_examples'] == item['source_examples']
            for name, result in item['benchmarks'].items():
                assert result['samples']['effective'] == reference['benchmark_coverage'][name]['eval_examples']
        assert key not in seen
        seen.add(key)
    print('FINAL_CHECKPOINT_STAGE_VERIFIED', stage, expected, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--verify-finetune', action='store_true')
    args = parser.parse_args()
    manifest, _, _, _ = load_bundle(args.bundle)
    if args.verify_finetune:
        reference = json.loads((args.run_root/'inputs_verified.json').read_text())
        verify('finetune', args.run_root/'finetune', reference)
        return
    old = json.loads((args.old_run/'inputs_verified.json').read_text())
    old_done = json.loads((args.old_run/'eval/complete.json').read_text())
    assert old_done['success'] is True and len(old_done['completed']) == 84
    reference = dict(checkpoints={}, benchmark_coverage=old['benchmark_coverage'],
                     bundle_sha256=sha256(args.bundle/'manifest.json'))
    assert manifest['validation']['en']['scored_targets'] == 9829694
    assert manifest['validation']['en']['packed_fingerprint'] == old['ppl_fingerprint']
    for arm in ARMS:
        name = f'{arm}_step5000'
        checkpoint = args.training_root/arm/'checkpoint-5000'
        ident = checkpoint_identity(checkpoint)
        assert ident == old['checkpoints'][name]
        provenance = training_provenance(checkpoint, manifest)
        assert provenance['global_step'] == 5000
        reference['checkpoints'][name] = ident
    write_json(args.run_root/'inputs_verified.json', reference)
    jobs = diagnostic_jobs(args.training_root, args.bundle, args.run_root/'diagnostics', args.tokenizer)
    write_json(args.run_root/'diagnostic_plan.json', jobs)
    run(jobs, list(range(8)), args.run_root/'diagnostics', Path(__file__).resolve().parents[1])
    verify('diagnostics', args.run_root/'diagnostics', reference, manifest)
    from eval.compare_diagnostics import compare
    reports = {arm: json.loads((args.run_root/'diagnostics'/f'{arm}_step5000'/'diagnostic_frequency.json').read_text()) for arm in ARMS}
    write_json(args.run_root/'frequency_comparison.json', dict(comparisons=[compare(reports['B0'], reports[arm]) for arm in ARMS[1:]]))


if __name__ == '__main__':
    main()
