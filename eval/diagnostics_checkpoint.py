"""Offline diagnostics on a frozen bundle and one checkpoint; never train/update."""
import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import offline
offline()
from capacity_allocation.data import sha256, write_json
from eval.runtime import checkpoint_identity, languages, load_checkpoint
from eval.diagnostic_data import load_bundle
from eval.diagnostic_metrics import frequency_nll, embedding_spectra
from eval.diagnostic_gradients import probe_gradients

DIAGNOSTICS = ('frequency', 'spectra', 'gradients')


def training_provenance(checkpoint, manifest, tiny=False):
    path = Path(checkpoint)
    state_path, config_path = path/'trainer_state.json', path.parent/'train_config.json'
    if not state_path.is_file() or not config_path.is_file():
        if tiny:
            return dict(status='tiny_test_no_training_metadata')
        raise ValueError('Production diagnostics require checkpoint trainer_state.json and parent train_config.json')
    state, config = json.loads(state_path.read_text()), json.loads(config_path.read_text())
    settings = manifest['preprocessing']
    expected = dict(block_size=settings['block_size'],
        preprocessing_num_workers=settings['workers'], preprocessing_batch_size=settings['map_batch_size'])
    if (config['train_fingerprint'] != manifest['training']['shuffled_fingerprint'] or
            any(config['data'][k] != v for k, v in expected.items()) or
            set(config['data']['languages'].split(',')) != set(manifest['languages'])):
        raise ValueError('Frozen frequency pool does not match checkpoint training data/preprocessing')
    # Current English training and diagnostics pack the identical eval stream.
    if len(manifest['languages']) == 1:
        lang = manifest['languages'][0]
        if config['eval_fingerprint'] != manifest['validation'][lang]['packed_fingerprint']:
            raise ValueError('Frozen evaluation stream does not match checkpoint evaluation')
    step, batch = state['global_step'], config['execution']['effective_batch_size']
    # Exact only for the full-batch first epoch used by the current screening.
    if type(step) is not int or step < 0 or type(batch) is not int or batch < 1:
        raise ValueError('Invalid training budget metadata')
    full_batches = step*batch <= manifest['training']['blocks']
    return dict(status='verified', global_step=step, effective_batch_size=batch,
        consumed_input_tokens=step*batch*settings['block_size'] if full_batches else None,
        consumed_scored_targets=step*batch*(settings['block_size']-1) if full_batches else None,
        count_policy='exact_full_batches_first_epoch_else_unknown',
        trainer_state_sha256=sha256(state_path), train_config_sha256=sha256(config_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer-name')
    parser.add_argument('--diagnostic-bundle', required=True)
    parser.add_argument('--languages', default='en')
    parser.add_argument('--diagnostics', choices=DIAGNOSTICS, nargs='+', default=list(DIAGNOSTICS))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--gradient-precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=4, help='Bound CPU covariance/BLAS threads per worker')
    parser.add_argument('--spectrum-buckets', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(set(args.diagnostics)) != len(args.diagnostics) or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error('Unique diagnostics and positive batch required')
    import torch
    torch.set_num_threads(args.cpu_threads)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Run one unsharded diagnostic worker, not DDP')
    output = Path(args.output)
    if output.exists():
        parser.error('Fresh output required')
    manifest, counts, mapping, datasets = load_bundle(args.diagnostic_bundle)
    selected = languages(args.languages)
    if any(lang not in datasets for lang in selected):
        parser.error('Requested language absent from diagnostic bundle')
    model, tokenizer = load_checkpoint(args.checkpoint, args.tokenizer_name, args.device, args.precision)
    from eval.diagnostic_data import tokenizer_identity
    if tokenizer_identity(tokenizer) != manifest['tokenizer_sha256'] or model.config.vocab_size != len(counts):
        raise ValueError('Checkpoint tokenizer/vocabulary differs from diagnostic bundle')
    identity = checkpoint_identity(args.checkpoint)
    provenance = training_provenance(args.checkpoint, manifest, getattr(model.config, 'tiny_test', False))
    datasets = {lang: datasets[lang] for lang in selected}
    results = {}
    if 'frequency' in args.diagnostics:
        results['frequency'] = frequency_nll(model, datasets, mapping, device=args.device,
            precision=args.precision, batch_size=args.batch_size)
        for lang in selected:
            if results['frequency']['by_language'][lang]['overall']['scored_targets'] != manifest['validation'][lang]['scored_targets']:
                raise ValueError('Frequency NLL did not score the complete frozen eval set')
    if 'spectra' in args.diagnostics:
        results['spectra'] = embedding_spectra(model, counts, mapping, include_buckets=args.spectrum_buckets)
    if 'gradients' in args.diagnostics:
        results['gradients'] = probe_gradients(model, datasets,
            {lang: manifest['probe_ids'][lang] for lang in selected}, mapping,
            device=args.device, precision=args.gradient_precision, batch_size=args.batch_size)
    if checkpoint_identity(args.checkpoint) != identity:
        raise ValueError('Checkpoint files changed during diagnostics')
    code_root = Path(__file__).resolve().parents[1]
    result = dict(success=True, checkpoint=identity, languages=selected,
        diagnostic_manifest_sha256=sha256(Path(args.diagnostic_bundle)/'manifest.json'),
        training=provenance, settings=vars(args), diagnostics=results,
        code={str(p.relative_to(code_root)): sha256(p) for p in
              sorted((code_root/'eval').glob('*.py')) + [code_root/'capacity_allocation/modeling.py']})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print('CHECKPOINT_DIAGNOSTICS_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
