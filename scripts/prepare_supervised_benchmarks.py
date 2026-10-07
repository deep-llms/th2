"""Dev-machine download of pinned STS-B/BoolQ train and labeled validation files.

GPU nodes acquire these same immutable files via controller #d directives.
No official hidden-label test files are used by this study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCES = {
    'stsb': ('nyu-mll/glue', 'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c'),
    'boolq': ('aps/super_glue', '3de24cf8022e94f4ee4b9d55a6f539891524d646'),
}


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    root = Path(args.output); repositories = []
    for task, (repo, revision) in SOURCES.items():
        files = []; data_files = {}
        for split in ('train', 'validation'):
            name = f'{task}/{split}-00000-of-00001.parquet'
            cached = hf_hub_download(repo, name, repo_type='dataset', revision=revision)
            dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.read_bytes() != Path(cached).read_bytes():
                raise ValueError('Refusing to overwrite different benchmark data')
            shutil.copyfile(cached, dest)
            files.append(dict(path=name, bytes=dest.stat().st_size,
                              sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
            data_files[split] = [name]
        repositories.append(dict(repo_id=repo, revision=revision, path=task, tasks=[task], files=files,
                                 load_configs={task: dict(loader='parquet', data_files=data_files)}))
    manifest = Path(args.manifest); manifest.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(schema_version=1, repositories=repositories), indent=2)+'\n'
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError('Refusing to replace a different manifest')
    manifest.write_text(content)
    print('Pinned four raw files:', manifest)


if __name__ == '__main__':
    main()
