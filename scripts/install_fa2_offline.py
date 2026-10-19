#!/usr/bin/env python3
"""Build the verified FA2 sdist offline, exclusively inside train_fa2.

Fetch inputs through the controller before running this. No system packages,
driver changes, GPU allocations, or existing-workload signals are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

PREFIX = Path('/mnt/local/conda-py311/envs/train_fa2')
OUTPUTS = Path('/mnt/local/_outputs/deep-llms_th2')
SOURCE_SHA256 = 'd0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b'
GIB = 1024 ** 3


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot(python: Path) -> dict:
    code = '''
import hashlib, importlib.metadata as md, json
items = {}
for dist in md.distributions():
    name = dist.metadata.get('Name')
    if name:
        files = '\\n'.join(dist.read_text(f) or '' for f in ('METADATA', 'RECORD', 'direct_url.json'))
        items[name.lower().replace('_', '-')] = [dist.version, hashlib.sha256(files.encode()).hexdigest()]
print(json.dumps(items, sort_keys=True))
'''
    return json.loads(subprocess.check_output([str(python), '-c', code], text=True, timeout=60))


def stop_owned_group(process: subprocess.Popen) -> None:
    # This group belongs to the child we created with start_new_session=True.
    # It is never inferred from a GPU PID or a process-name match.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != PREFIX or Path(sys.executable).parent.resolve() != PREFIX / 'bin':
        raise RuntimeError('Refuse: use only the isolated train_fa2 interpreter')
    work = args.work_dir.resolve()
    if work.parent != OUTPUTS or not work.name.startswith('6y3o-fa2-build-') or work.exists():
        raise RuntimeError('Refuse: expected a fresh, dedicated FA2 build directory')
    if digest(args.archive) != SOURCE_SHA256:
        raise RuntimeError('Official source archive checksum mismatch')

    import importlib.metadata as md
    import psutil
    import torch

    if md.version('torch') != '2.14.1' or torch.version.cuda != '13.0' or not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError('Unexpected PyTorch/CUDA/ABI; do not change the shared runtime')
    try:
        md.version('flash-attn')
    except md.PackageNotFoundError:
        pass
    else:
        raise RuntimeError('FA2 is already installed; refuse an unreviewed overwrite')
    # CUDA 13 pip distributions use a common environment-local toolkit root.
    cuda = PREFIX / 'lib/python3.11/site-packages/nvidia/cu13'
    if not (cuda / 'bin/nvcc').is_file():
        raise RuntimeError(f'Environment-local CUDA compiler missing: {cuda}')
    nvcc = subprocess.check_output([str(cuda / 'bin/nvcc'), '--version'], text=True, timeout=20)
    if not re.search(r'release 13\.0\b', nvcc):
        raise RuntimeError('Expected CUDA 13.0 nvcc')
    for relative in ('include/cuda_runtime.h', 'include/crt/host_config.h', 'include/cccl/cuda/std/type_traits', 'nvvm/libdevice/libdevice.10.bc'):
        if not (cuda / relative).is_file():
            raise RuntimeError(f'Missing build input: {cuda / relative}')
    for tool in ('nice', 'prlimit'):
        if not shutil.which(tool):
            raise RuntimeError(f'Missing resource control: {tool}')
    if psutil.virtual_memory().available < 128 * GIB or shutil.disk_usage('/mnt/local').free < 20 * GIB:
        raise RuntimeError('Insufficient free resources for a bounded build')

    protected = {name: PREFIX.parent / name / 'bin/python3.11' for name in ('train', 'eval')}
    before = {name: snapshot(python) for name, python in protected.items()}
    isolated_before = snapshot(Path(sys.executable))
    work.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'wheels', 'source'):
        (work / name).mkdir()
    (work / 'production_before.json').write_text(json.dumps(before, indent=2) + '\n')
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        if sum(m.size for m in members) > GIB:
            raise RuntimeError('Unexpected source archive size')
        for member in members:
            parts = Path(member.name).parts
            if not parts or parts[0] != 'flash_attn-2.8.3.post2' or '..' in parts or '.git' in parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f'Unexpected archive member: {member.name}')
        archive.extractall(work / 'source', filter='data')
    source = work / 'source/flash_attn-2.8.3.post2'
    if not (source / 'csrc/cutlass/include/cutlass/cutlass.h').is_file():
        raise RuntimeError('Bundled CUTLASS header missing')

    env = dict(os.environ)
    env.update(CUDA_HOME=str(cuda), PATH=f'{PREFIX}/bin:{cuda}/bin:' + env['PATH'],
               CPATH=str(cuda / 'include/cccl'),
               CUDA_VISIBLE_DEVICES='', FLASH_ATTENTION_FORCE_BUILD='TRUE',
               FLASH_ATTENTION_SKIP_CUDA_BUILD='FALSE', BUILD_TARGET='cuda',
               FLASH_ATTN_CUDA_ARCHS='100', MAX_JOBS='4', NVCC_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1',
               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(work / 'tmp'))
    command = ['nice', '-n', '10', 'prlimit', f'--as={64 * GIB}', '--', sys.executable,
               '-m', 'pip', 'wheel', '--no-index', '--no-build-isolation', '--no-deps',
               '--no-cache-dir', '--verbose', '--wheel-dir', str(work / 'wheels'), str(source)]
    report = dict(source_sha256=SOURCE_SHA256, torch=torch.__version__, cuda_home=str(cuda),
                  nvcc=nvcc, max_jobs=4, nvcc_threads=1, arch='100', per_process_as_limit=64 * GIB,
                  tree_rss_limit=96 * GIB, timeout_seconds=5400, stage='building')
    (work / 'build_config.json').write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    print('FA2_BUILD_START ' + json.dumps(report), flush=True)
    try:
        smoke_source = work / 'compiler_check.cu'
        smoke_source.write_text('#include <cuda_runtime.h>\n#include <cuda/std/type_traits>\n'
                                '__global__ void compiler_check(float* p) { p[0] = 1.0f; }\n')
        compiler = subprocess.Popen(
            ['nice', '-n', '10', 'prlimit', f'--as={64 * GIB}', '--',
             str(cuda / 'bin/nvcc'), '-arch=sm_100', '-c', str(smoke_source),
             '-o', str(work / 'compiler_check.o')], env=env, start_new_session=True)
        try:
            if compiler.wait(timeout=120):
                raise RuntimeError('Environment-local CUDA compiler check failed')
        except BaseException:
            stop_owned_group(compiler)
            raise
        print('FA2_COMPILER_CHECK_PASS (compile only; no GPU execution)', flush=True)
        with (work / 'build.log').open('w') as log:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        tree = [parent, *parent.children(recursive=True)]
                        rss = sum(p.memory_info().rss for p in tree if p.is_running())
                    except psutil.NoSuchProcess:
                        rss = 0
                    elapsed = time.monotonic() - started
                    if rss > 96 * GIB or elapsed > 5400 or psutil.virtual_memory().available < 128 * GIB or shutil.disk_usage(work).free < 20 * GIB:
                        raise RuntimeError('Build resource/time limit reached; stopping only the owned build group')
                    print('FA2_BUILD_PROGRESS ' + json.dumps(dict(seconds=round(elapsed), tree_rss_gib=round(rss / GIB, 2), object_files=sum(1 for _ in source.glob('build/**/*.o')))), flush=True)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise RuntimeError(f'Wheel build exited {process.returncode}; see {work / "build.log"}')
            except BaseException:
                stop_owned_group(process)
                raise
        wheels = list((work / 'wheels').glob('flash_attn-*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Expected exactly one built FA2 wheel')
        report.update(stage='installing', wheel=str(wheels[0]), wheel_sha256=digest(wheels[0]))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', str(wheels[0])],
                       env=env, check=True, timeout=300)
        verify = "import torch, flash_attn, flash_attn_2_cuda; from flash_attn import flash_attn_varlen_func; print(flash_attn.__version__, flash_attn_2_cuda.__file__)"
        report['import_check'] = subprocess.check_output([sys.executable, '-c', verify], env=env, text=True, timeout=60).strip()
        after = {name: snapshot(python) for name, python in protected.items()}
        isolated_after = snapshot(Path(sys.executable))
        added = isolated_after.pop('flash-attn')
        if before != after or isolated_before != isolated_after:
            raise RuntimeError('Unexpected package metadata changes; preserve results for inspection')
        report.update(stage='complete', installed_version=added[0], production_metadata_unchanged=True,
                      isolated_existing_metadata_unchanged=True, elapsed_seconds=time.monotonic() - started)
        (work / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print('FA2_INSTALL_COMPLETE ' + json.dumps(report), flush=True)
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
        (work / 'failure.json').write_text(json.dumps(report, indent=2) + '\n')
        print('FA2_INSTALL_FAILED ' + json.dumps(report), flush=True)
        raise


if __name__ == '__main__':
    main()
