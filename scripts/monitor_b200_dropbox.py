"""Read-only hourly Dropbox receipt monitor for the current B200 runner.

This does not submit runner commands, touch GPUs, or infer GPU availability from
Dropbox activity. State and downloaded status logs stay in ignored temp/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts.dropbox_access import AccessError, access_token, download, list_folder, shared_link


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.state-',
                                     suffix='.tmp', delete=False) as handle:
        temporary=Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
    temporary.replace(path)


def poll(credentials, folders, label, state_dir):
    state_dir=Path(state_dir)
    os.umask(0o077)
    state_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    state_dir.chmod(0o700)
    state_file=state_dir/'status.json'
    previous=json.loads(state_file.read_text()) if state_file.is_file() else {}
    checked=datetime.now(timezone.utc).isoformat()
    result={**previous, 'last_checked_utc':checked, 'last_error':None}
    try:
        url=shared_link(Path(folders),label)
        token=access_token(Path(credentials))
        entries=list(list_folder(token,url))
        files=[entry for entry in entries if entry['type']=='file']
        status=next((entry for entry in files if entry['path']=='/_RUN_STATUS_.log'),None)
        result['dropbox_reachable']=True
        result['latest_file']=max(files,key=lambda item:item.get('modified') or '') if files else None
        if status is None:
            result['last_error']='Runner status file is absent from the share'
        elif status.get('modified')!=previous.get('runner_status_modified'):
            stamp=status['modified'].replace(':','').replace('-','')
            destination=state_dir/f'runner-status-{stamp}.log'
            if destination.is_file():
                raw=destination.read_bytes()
                if len(raw)!=status['bytes']:
                    raise ValueError('Existing runner status snapshot has a different size')
                receipt={'sha256':hashlib.sha256(raw).hexdigest()}
            else:
                receipt=download(token,url,status['path'],destination)
            lines=destination.read_text(errors='replace').splitlines()
            failed=max((i for i,line in enumerate(lines)
                        if 'FAILED(rc=255)' in line and 'th2-tjx3' in line),default=-1)
            later_success=any(i>failed and 'th2-tjx3' in line and '#1' in line
                              and '| OK | synced to bx' in line
                              for i,line in enumerate(lines)) if failed>=0 else False
            result.update(runner_status_modified=status['modified'],
                          runner_status_sha256=receipt['sha256'],
                          runner_status_local=str(destination),
                          latest_runner_event=lines[-1] if lines else None,
                          candidate_controller_recovery=later_success,
                          latest_failed_event=lines[failed] if failed>=0 else None)
    except (AccessError,OSError,KeyError,ValueError) as error:
        result['dropbox_reachable']=False
        result['last_error']=str(error) if isinstance(error,AccessError) else type(error).__name__
    write_json(state_file,result)
    print(json.dumps({key:result.get(key) for key in
                      ('last_checked_utc','dropbox_reachable','runner_status_modified',
                       'candidate_controller_recovery','latest_runner_event','last_error')}))
    return 0 if result['last_error'] is None else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credentials',type=Path,default=Path('temp/dropbox_credentials.txt'))
    p.add_argument('--folders',type=Path,default=Path('temp/dropbox_tjx3_folders.txt'))
    p.add_argument('--label',default='th2-tjx3')
    p.add_argument('--state-dir',type=Path,default=Path('temp/b200_hourly_monitor'))
    a=p.parse_args()
    raise SystemExit(poll(a.credentials,a.folders,a.label,a.state_dir))


if __name__=='__main__':
    main()
