"""Bounded real-checkpoint CUDA diagnostic gate; no weight updates or downloads."""
import argparse
import gc
from pathlib import Path

from eval.runtime import offline
offline()
import torch
from capacity_allocation.data import write_json
from eval.runtime import load_checkpoint, checkpoint_identity
from eval.diagnostic_data import load_bundle, tokenizer_identity
from eval.diagnostics_checkpoint import training_provenance
from eval.diagnostic_metrics import frequency_nll, embedding_spectra
from eval.diagnostic_gradients import probe_gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    manifest, counts, mapping, datasets = load_bundle(args.bundle)
    subset = {lang: data.select(manifest['probe_ids'][lang]) for lang, data in datasets.items()}
    reports = {}
    for arm in ('B0', 'C'):
        path = args.training_root/arm/'checkpoint-5000'
        identity = checkpoint_identity(path)
        model, tokenizer = load_checkpoint(path, args.tokenizer, 'cuda', 'bf16')
        assert tokenizer_identity(tokenizer) == manifest['tokenizer_sha256']
        assert training_provenance(path, manifest)['global_step'] == 5000
        frequency = frequency_nll(model, subset, mapping, device='cuda', precision='bf16')
        assert frequency['by_language']['en']['overall']['scored_targets'] == 8*2047
        spectra = embedding_spectra(model, counts, mapping)
        gradients = probe_gradients(model, datasets, manifest['probe_ids'], mapping, device='cuda', precision='fp32')
        assert gradients['scored_targets'] == 8*2047 and gradients['optimizer_steps'] == 0
        assert checkpoint_identity(path) == identity
        reports[arm] = dict(frequency=frequency, spectra=spectra, gradients=gradients)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('REAL_CHECKPOINT_DIAGNOSTIC_SMOKE_PASSED', arm, flush=True)
    write_json(args.output, dict(success=True, smoke_only=True, reports=reports))


if __name__ == '__main__':
    main()
