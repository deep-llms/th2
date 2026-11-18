"""Restore exact archived historical results locally; never overwrite changed files."""
import argparse,gzip,hashlib,json
from pathlib import Path

def restore(root,manifest):
    root=Path(root).resolve()
    for item in json.loads(Path(manifest).read_text())['files']:
        src=(root/item['archive']).resolve();dst=(root/item['path']).resolve()
        if not src.is_relative_to(root) or not dst.is_relative_to(root):raise ValueError('Escaping path')
        raw=src.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['archive_sha256']:raise ValueError('Archive hash')
        data=gzip.decompress(raw)
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Restored hash')
        if dst.exists():
            if dst.read_bytes()!=data:raise ValueError('Existing result changed')
        else:
            dst.parent.mkdir(parents=True,exist_ok=True)
            with dst.open('xb') as f:f.write(data)
        print(item['path'],'PASS')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--manifest',default='resources/research_source_archives_20261119.json');a=p.parse_args();restore(a.root,a.manifest)
