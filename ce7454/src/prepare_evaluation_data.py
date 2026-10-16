"""Verify/decode dev and official test only, into a separate evaluation cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare(raw, cache, expected_manifest):
    manifest_path=raw/'evaluation_manifest.json'
    if sha(manifest_path)!=expected_manifest:
        raise ValueError('Evaluation manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    if set(manifest['splits'])!={'dev','test_official'}:
        raise ValueError('Expected only dev and official test')
    # Check all archives before writing cache; never touch training/cache paths.
    for name,count in [('dev',900),('test_official',1500)]:
        spec=manifest['splits'][name]; archive=raw/spec['archive']
        if archive.resolve().parent!=raw.resolve():raise ValueError('Invalid archive path')
        if len(spec['images'])!=count or len({r['filename'] for r in spec['images']})!=count:
            raise ValueError('Wrong image count or duplicate filenames')
        if archive.stat().st_size!=spec['archive_bytes'] or sha(archive)!=spec['archive_sha256']:
            raise ValueError('Archive hash/size mismatch')
    cache.mkdir(parents=True,exist_ok=False)
    receipt={'status':'passed','purpose':'evaluation_only','manifest_sha256':expected_manifest,'splits':{}}
    for name,spec in manifest['splits'].items():
        rows=spec['images']; path=cache/(name+'_images.npy')
        arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(len(rows),512,512,3))
        with zipfile.ZipFile(raw/spec['archive']) as z:
            def decode(row):
                info=z.getinfo(row['archive_path'])
                if info.file_size!=row['bytes'] or info.CRC!=row['crc32']:
                    raise ValueError('Archive member differs from manifest')
                with Image.open(io.BytesIO(z.read(info))) as im:a=np.asarray(im.convert('RGB'))
                if a.shape!=(512,512,3) or a.dtype!=np.uint8:raise ValueError('Wrong image shape')
                return a
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i,a in enumerate(pool.map(decode,rows)):arr[i]=a
        arr.flush();del arr
        meta=cache/(name+'_meta.json')
        meta.write_text(json.dumps({'filenames':[r['filename'] for r in rows]})+'\n')
        receipt['splits'][name]={'images':len(rows),'archive_sha256':spec['archive_sha256'],
             'images_sha256':sha(path),'meta_sha256':sha(meta)}
        print('VERIFIED',name,receipt['splits'][name],flush=True)
    (cache/'ready.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.raw,a.cache,a.manifest_sha256)),flush=True)
