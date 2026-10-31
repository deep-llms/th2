"""CPU-only canonical readout revision after registered GPU probes finish.

Preserves the originally frozen run and reports; reads saved token arrays only.
Frequency counts include ALL final-set token positions, including position 0.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from scripts.a2_quant_io import lock,write
from scripts.a2_remaining import read_locked
from scripts.a2_mech_report import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code-commit',required=True)
    a=p.parse_args();started=time.monotonic();base=a.root/'results/a2_mech_20261031'
    cfg=json.loads((base/'config/plan.json').read_text())
    # Invocation is made after terminal evidence; do not publish partial canonical results.
    complete=json.loads((base/'complete.json').read_text()) if (base/'complete.json').exists() else None
    native=json.loads((a.root/'supervised/supervisor.json').read_text())
    if not native.get('finished_at'):raise ValueError('GPU supervisor has not reached terminal handoff')
    revision=base/'readout_frequency_revision';revision.mkdir(exist_ok=False)
    (revision/'workers').symlink_to(base/'workers',target_is_directory=True)
    if (base/'preparation_checks.json').exists():shutil.copyfile(base/'preparation_checks.json',revision/'preparation_checks.json')
    sources={str(Path(__file__).resolve()):lock(__file__),str(Path(summarize.__code__.co_filename).resolve()):lock(summarize.__code__.co_filename)}
    import efficient_llm.a2.mech as mech
    sources[str(Path(mech.__file__).resolve())]=lock(mech.__file__)
    provenance=dict(spec_sha256=cfg['spec_sha256'],code_commit=a.code_commit,source_files=sources,
        prior_complete=lock(base/'complete.json') if complete else None,native_terminal=lock(a.root/'supervised/supervisor.json'),prior_decision=lock(base/'decision_mech.json') if (base/'decision_mech.json').exists() else None,
        correction='Count token types over all 4096 positions per final sequence; buckets contain scored positions 1..4095. No forward passes, training or compression.',
        original_run_preserved=True)
    write(revision/'config/plan.json',dict(cfg,readout_revision=provenance))
    checks=json.loads((revision/'preparation_checks.json').read_text()) if (revision/'preparation_checks.json').exists() else []
    if native.get('training_status')!='ok':
        checks.append(dict(test='original_coordinator_exit',passed=False,spec_sha256=cfg['spec_sha256'],detail=dict(native_status=native.get('training_status'),action='CPU completion recovers registered weights/readout; original error retained')))
    for path in sorted((base/'workers/tokens').glob('*/*/*.npy')):
        ds,parent=path.parts[-3:-1];name=path.stem
        try:
            metadata=json.loads(path.with_suffix('.json').read_text())
            actual_lock=lock(path)
            means=np.asarray(np.load(path,mmap_mode='r'),dtype=np.float64).mean(-1)
            storage_error=float(np.max(np.abs(means-metadata['per_sequence_nll'])))
            if name.startswith('student-'):
                identity=cfg['students'][ds+'/'+name.removeprefix('student-')]
                reference=read_locked(identity['training_receipts']['result'])['scores']['final']
            else:
                key=ds+'/'+parent+('/pre-S1' if name.startswith('pre-') else '/parent')
                reference=read_locked(cfg['reference_nll'][key])
            reference_error=float(np.max(np.abs(means-reference['per_sequence_nll'])))
            passed=actual_lock==metadata['tokens_file'] and storage_error<=1e-6 and reference_error<=1e-6
            detail=dict(file=actual_lock,storage_max_abs=storage_error,prior_per_sequence_max_abs=reference_error)
        except Exception as error:
            passed=False;detail=dict(error=repr(error),action='retain token arrays and continue readout with flag')
        checks.append(dict(test='MT6_external_replay:'+ds+'/'+name,passed=passed,detail=detail,spec_sha256=cfg['spec_sha256']))
    write(revision/'preparation_checks.json',checks)
    result=summarize(cfg,revision);fresh=summarize(cfg,revision)
    verified=result==fresh
    write(revision/'verification.json',dict(spec_sha256=cfg['spec_sha256'],independent_readout_exact=verified,
        elapsed_seconds=time.monotonic()-started,provenance=provenance,coverage_complete=result['coverage_complete']))
    with (revision/'REPORT.md').open('a') as f:
        f.write('\nCanonical CPU readout revision: frequency counts all final-set positions, including position 0. Original readout preserved in parent directory. Independent recomputation exact: '+str(verified)+'.\n')
    print('MECH_CANONICAL_READOUT',json.dumps(dict(root=str(revision),verified=verified,coverage_complete=result['coverage_complete'])),flush=True)


if __name__=='__main__':main()
