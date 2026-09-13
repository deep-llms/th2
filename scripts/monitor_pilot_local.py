"""DEV ONLY: hourly #2 snapshots and Dropbox retrieval, never GPU commands.

Stops on Git ownership changes or runner/access errors. Run in local tmux.
Credentials and resulting reports stay in ignored local directories.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from dropbox_access import access_token, shared_link, list_folder, download

ROOT = '/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01'
PREP = '/mnt/local/_outputs/deep-llms_th2/ccm_prepare_pilot_v1_20260913_a01'
# These three files exist as soon as the handoff reports ARMED. Avoid requests
# for future artifacts: missing-file reports must not look like runner failure.
EXPORTS = [ROOT+'.handoff.log', ROOT+'/status.json', PREP+'/prepare.log']


def git(checkout, *args):
    return subprocess.check_output(['git', '-C', str(checkout), *args], text=True, timeout=120).strip()


def submit_snapshot(checkout, expected, tag, patch_binary):
    if git(checkout, 'status', '--porcelain'):
        raise RuntimeError('Execution checkout changed; monitoring relinquishes Git control')
    if git(checkout, 'remote', 'get-url', 'origin') != 'git@github-share:deep-llms/th2.git':
        raise RuntimeError('Unexpected execution remote')
    git(checkout, 'fetch', 'origin', 'main')
    if git(checkout, 'rev-parse', 'HEAD') != expected or git(checkout, 'rev-parse', 'origin/main') != expected:
        raise RuntimeError('Execution HEAD changed; monitoring relinquishes Git control')
    old = (checkout/'commands.sh').read_text()
    new = '#2 +a -f-'+','.join(EXPORTS)+'\n#th2-ccm-hourly-'+tag+'\n'
    patch = '*** Begin Patch\n*** Update File: '+str(checkout/'commands.sh')+'\n@@\n'
    patch += '\n'.join('-'+x for x in old.splitlines())+'\n'
    patch += '\n'.join('+'+x for x in new.splitlines())+'\n*** End Patch\n'
    subprocess.run([patch_binary], input=patch, text=True, check=True, capture_output=True)
    git(checkout, 'add', 'commands.sh')
    git(checkout, 'diff', '--cached', '--check')
    git(checkout, 'commit', '-m', 'Retrieve overnight pilot status '+tag)
    git(checkout, 'push', 'origin', 'HEAD:main')
    result = git(checkout, 'rev-parse', 'HEAD')
    if git(checkout, 'ls-remote', 'origin', 'refs/heads/main').split()[0] != result:
        raise RuntimeError('Pushed head not confirmed')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--credentials', type=Path, required=True)
    p.add_argument('--folders', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--apply-patch', required=True)
    p.add_argument('--hours', type=int, default=24)
    a = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.expected_head) or not 1 <= a.hours <= 48:
        raise SystemExit('Invalid monitor head/duration')
    a.output.mkdir()  # Fresh local output, never overwrite previous evidence.
    expected = a.expected_head
    previous_status_lines = None
    for iteration in range(a.hours):
        if (a.output/'STOP').exists():
            break
        started = time.monotonic()
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        here = a.output/tag
        here.mkdir()
        # Read baseline status BEFORE any Git mutation, so a reported runner
        # error blocks even a harmless snapshot submission.
        token = access_token(a.credentials)
        url = shared_link(a.folders, 'th2')
        download(token, url, '/_RUN_STATUS_.log', here/'status_before.log')
        lines = (here/'status_before.log').read_text().splitlines()
        recent = lines[-12:] if previous_status_lines is None else [x for x in lines if x not in previous_status_lines]
        if any(re.search(r'\b(ERROR|FAILED)\b', x, re.I) for x in recent):
            raise RuntimeError('Runner reports an error; no new submission. Inspect status_before.log')
        expected = submit_snapshot(a.checkout, expected, tag, a.apply_patch)
        (a.output/'expected_head.txt').write_text(expected+'\n')
        print(json.dumps(dict(time=tag, event='snapshot_requested', commit=expected)), flush=True)
        time.sleep(150)  # Controller export is asynchronous; never resend #1.
        token = access_token(a.credentials)
        entries = list(list_folder(token, url))
        candidates = []
        for item in entries:
            relevant = ('ccm_pilot_seed17_20260913_a01' in item['path'] or
                        'ccm_prepare_pilot_v1_20260913_a01' in item['path'] or
                        item['path'] == '/_RUN_STATUS_.log')
            if relevant and item['type'] == 'file':
                candidates.append(item)
            elif relevant and item['type'] == 'folder':
                candidates.extend(x for x in list_folder(token, url, item['path']) if x['type'] == 'file')
        report = dict(time=tag, commit=expected, files=[], workflow=None)
        for item in candidates:
            if not item['bytes'] or item['bytes'] > 25*1024*1024:
                continue
            path = item['path']
            name = hashlib.sha256(path.encode()).hexdigest()[:12]+'_'+Path(path).name
            record = download(token, url, path, here/name)
            report['files'].append(dict(remote=path, **record))
            content = (here/name).read_text(errors='replace')
            if path == '/_RUN_STATUS_.log':
                current = content.splitlines()
                new_lines = [x for x in current if x not in lines]
                if any(re.search(r'\b(ERROR|FAILED)\b', x, re.I) for x in new_lines):
                    raise RuntimeError('Runner error after export; inspect the downloaded status')
                previous_status_lines = set(current)
            if 'ccm_pilot_seed17_20260913_a01' in path and path.endswith('status.json'):
                report['workflow'] = json.loads(content)
        (here/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(event='snapshot_pulled', directory=str(here),
                              workflow=report['workflow'], files=len(report['files']))), flush=True)
        state = report['workflow']
        if state and state.get('stage') in ('complete', 'failed'):
            break
        # Short sleeps allow STOP to relinquish control promptly.
        while time.monotonic()-started < 3600 and not (a.output/'STOP').exists():
            time.sleep(30)
    print('LOCAL_PILOT_MONITOR_EXITED', flush=True)


if __name__ == '__main__':
    main()
