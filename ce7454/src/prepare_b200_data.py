"""Verify and decode the uploaded train-only archive into the existing cache format."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from prepare_cache import build_split

EXPECTED_SHA256 = 'ef57baacee12b92038a845308b0911dd71c5b564bdda43b516ed2f90fbb08ce3'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    if args.cache.exists():
        raise FileExistsError('Refusing to overwrite an existing cache')
    manifest = json.loads((args.raw / 'manifest.json').read_text())
    with (args.raw / 'train.zip').open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != EXPECTED_SHA256 or sha != manifest['sha256']:
        raise ValueError('Training archive checksum mismatch')
    rows = manifest['images']
    if len(rows) != 15000 or len({r['filename'] for r in rows}) != 15000:
        raise ValueError('Expected 15000 unique images')
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='ce7454-decode-', dir=args.cache.parent))
    try:
        images = stage / 'images'
        images.mkdir()
        cache = stage / 'cache'
        cache.mkdir()
        with zipfile.ZipFile(args.raw / 'train.zip') as z:
            for row in rows:
                name = row['filename']
                if Path(name).name != name or name in ('.', '..'):
                    raise ValueError('Unsafe filename')
                with z.open(row['path']) as source, (images / name).open('xb') as target:
                    shutil.copyfileobj(source, target)
        meta = {'filenames': [r['filename'] for r in rows], 'labels': [r['label'] for r in rows]}
        build_split('train', str(images), meta['filenames'], str(cache), args.workers)
        (cache / 'train_meta.json').write_text(json.dumps(meta))
        (cache / 'ready.json').write_text(json.dumps({'archive_sha256': sha, 'images': 15000,
            'split': 'train_only', 'shape': [15000, 512, 512, 3], 'dtype': 'uint8'}) + '\n')
        cache.rename(args.cache)
    finally:
        shutil.rmtree(stage)
    print('CE7454_DATA_READY', args.cache, sha, flush=True)


if __name__ == '__main__':
    main()
