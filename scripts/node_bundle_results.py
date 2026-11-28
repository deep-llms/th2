"""Bundle the small result files of a node_dot_queue batch for a single #2 export (no checkpoints).

Per run: summary.json, calibration.jsonl, terminal.json, scores.npz, downstream_scores.json; plus the batch's
jobs.json, terminal.json, native-verified.json, failure files and the job logs' last lines. Writes
<out>.tar.gz and <out>.sha256 (the manifest of every bundled file).
  python scripts/node_bundle_results.py --base <batch run dir> --out <path without suffix>
"""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile

PER_RUN = ('summary.json', 'calibration.jsonl', 'terminal.json', 'scores.npz', 'downstream_scores.json')
BATCH = ('jobs.json', 'terminal.json', 'native-verified.json', 'failure.json', 'handoff-failure.json',
         'admission.json', 'reclaim.json', 'gpus-free-before-training.json')


def files(base):
    out = [base/n for n in BATCH if (base/n).is_file()]
    for run in sorted(base.glob('s*/*')):
        if run.is_dir() and not run.name.startswith('qualification-'):
            out += [run/n for n in PER_RUN if (run/n).is_file()]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    tar_path, manifest = Path(str(a.out) + '.tar.gz'), Path(str(a.out) + '.sha256')
    if tar_path.exists() or manifest.exists():
        raise SystemExit('bundle exists')
    picked = files(a.base)
    lines = [f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(a.base)}' for f in picked]
    with tarfile.open(tar_path, 'w:gz') as tar:
        for f in picked:
            tar.add(f, arcname=str(f.relative_to(a.base)))
        for log in sorted((a.base/'logs').glob('*.log')):
            tail = b''.join(log.read_bytes().splitlines(keepends=True)[-40:])
            info = tarfile.TarInfo(f'logs/{log.name}.tail')
            info.size = len(tail)
            tar.addfile(info, io.BytesIO(tail))
    manifest.write_text('\n'.join(lines) + '\n')
    print('BUNDLE', tar_path, tar_path.stat().st_size, 'bytes', len(picked), 'files',
          hashlib.sha256(tar_path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
