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
from scripts.a2_quant_io import lock,write
from scripts.a2_remaining import read_locked
from scripts.a2_mech_report import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code-commit',required=True)
    a=p.parse_args();started=time.monotonic();base=a.root/'results/a2_mech_20261031'
    cfg=json.loads((base/'config/plan.json').read_text())
    # Invocation is made after terminal evidence; do not publish partial canonical results.
    complete=json.loads((base/'complete.json').read_text())
    if not complete['execution_complete']:raise ValueError('GPU probes have not finished')
    revision=base/'readout_frequency_revision';revision.mkdir(exist_ok=False)
    (revision/'workers').symlink_to(base/'workers',target_is_directory=True)
    if (base/'preparation_checks.json').exists():shutil.copyfile(base/'preparation_checks.json',revision/'preparation_checks.json')
    sources={str(Path(__file__).resolve()):lock(__file__),str(Path(summarize.__code__.co_filename).resolve()):lock(summarize.__code__.co_filename)}
    import efficient_llm.a2.mech as mech
    sources[str(Path(mech.__file__).resolve())]=lock(mech.__file__)
    provenance=dict(spec_sha256=cfg['spec_sha256'],code_commit=a.code_commit,source_files=sources,
        prior_complete=lock(base/'complete.json'),prior_decision=lock(base/'decision_mech.json') if (base/'decision_mech.json').exists() else None,
        correction='Count token types over all 4096 positions per final sequence; buckets contain scored positions 1..4095. No forward passes, training or compression.',
        original_run_preserved=True)
    write(revision/'config/plan.json',dict(cfg,readout_revision=provenance))
    result=summarize(cfg,revision);fresh=summarize(cfg,revision)
    verified=result==fresh
    write(revision/'verification.json',dict(spec_sha256=cfg['spec_sha256'],independent_readout_exact=verified,
        elapsed_seconds=time.monotonic()-started,provenance=provenance,coverage_complete=result['coverage_complete']))
    with (revision/'REPORT.md').open('a') as f:
        f.write('\nCanonical CPU readout revision: frequency counts all final-set positions, including position 0. Original readout preserved in parent directory. Independent recomputation exact: '+str(verified)+'.\n')
    print('MECH_CANONICAL_READOUT',json.dumps(dict(root=str(revision),verified=verified,coverage_complete=result['coverage_complete'])),flush=True)


if __name__=='__main__':main()
