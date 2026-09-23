"""Read-only inventory for the authorized th2 deployment; no GPU allocation."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    print(json.dumps({"host": platform.node(), "python": sys.executable,
                      "disk": shutil.disk_usage('/mnt/local')._asdict()}), flush=True)
    for package in ('torch', 'transformers', 'datasets', 'numpy', 'matplotlib', 'accelerate'):
        try:
            print(package, importlib.metadata.version(package), flush=True)
        except importlib.metadata.PackageNotFoundError:
            print(package, 'MISSING', flush=True)
    subprocess.run(['nvidia-smi'], check=True)
    subprocess.run([sys.executable, 'scripts/gpu_status.py'], check=True)
    import torch
    print('torch CUDA', torch.version.cuda, 'compiled arches', torch.cuda.get_arch_list(), flush=True)
    for proc in sorted(Path('/proc').glob('[0-9]*')):
        try:
            args = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if any(name in args for name in ('prepare_data.py', 'llm_pretrain_burn.py', 'pcc.joint')):
                # Exclude the container entrypoint, whose embedded script is huge.
                if int(proc.name) > 1:
                    print('relevant_process', proc.name, args[:600], flush=True)
        except (OSError, ValueError):
            pass
    root = Path('/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B-Base')
    for split in ('train/en', 'eval/en'):
        directory = root / split
        print('dataset', directory, 'exists', directory.is_dir(), flush=True)
        for info in sorted(directory.rglob('dataset_info.json')):
            print('dataset_info', info, info.read_text()[:2000], flush=True)
        files = list(directory.rglob('*.arrow'))
        print('arrow_files', len(files), 'bytes', sum(p.stat().st_size for p in files), flush=True)
    for p in sorted(Path('/mnt/local/_models/deep-llms_th2').glob('*/*')):
        if p.is_file():
            print('model_file', p, p.stat().st_size, flush=True)
    completion = Path('/mnt/local/_outputs/deep-llms_th2/data_preparation/culturax_qwen_tpbw_20260923_a01/sampling_complete.json')
    print('sampling_complete', completion.read_text(), flush=True)
    print('INSPECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
