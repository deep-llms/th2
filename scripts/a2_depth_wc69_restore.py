"""Restore exact registered inputs from backup; never overwrite existing bytes."""
import argparse
import json
from pathlib import Path
import socket
from datetime import datetime, timezone

from scripts.a2_prepare_corpora import sha, write_json
from scripts.a2_rebuild_wc69_data import copy_verified, require
from efficient_llm.a2_depth.common import SPEC_SHA


def restore_files(records):
    # Check every source/destination before writing any new target.
    destinations=set()
    for item in records:
        src,dst=Path(item['source']),Path(item['destination'])
        require(str(dst) not in destinations, 'Duplicate restore destination')
        destinations.add(str(dst))
        require(src.is_file() and not src.is_symlink(), 'Missing backup/source: '+str(src))
        require(sha(src)==item['sha256'], 'Source hash mismatch: '+str(src))
        require(not dst.is_symlink(), 'Destination symlink: '+str(dst))
        if dst.exists():
            require(dst.is_file() and sha(dst)==item['sha256'], 'Existing destination differs: '+str(dst))
    results=[]
    for item in records:
        dst=Path(item['destination'])
        if dst.exists():
            # Recheck immediately before reuse; retain changed/partial bytes on failure.
            require(dst.is_file() and not dst.is_symlink() and sha(dst)==item['sha256'], 'Destination changed')
            result=dict(item,bytes=dst.stat().st_size,action='reuse exact existing bytes')
        else:
            result=copy_verified(item['source'],dst,item['sha256'])
            result['action']='copied and verified'
        results.append(result)
        print('RESTORED',json.dumps(result),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True)
    args=parser.parse_args();recipe=json.loads(Path(args.recipe).read_text())
    require(recipe['spec_sha256']==SPEC_SHA,'Different Depth spec')
    require(recipe['host']==socket.gethostname()=='thiennh-p6-wc69-worker-0','Wrong node')
    output=Path(recipe['output'])
    require(output.parent==Path('/mnt/local/_outputs/deep-llms_th2') and not output.exists(),'Wrong/existing output')
    roots=[Path('/mnt/local/_outputs/deep-llms_th2'),Path('/mnt/local/_models/deep-llms_th2')]
    for item in recipe['files']:
        target=Path(item['destination'])
        require(any(target.resolve().is_relative_to(root) for root in roots),'Destination outside project assets')
    output.mkdir()
    try:
        files=restore_files(recipe['files'])
        receipt=dict(status='verified',spec_sha256=SPEC_SHA,utc=datetime.now(timezone.utc).isoformat(),
                     recipe_sha256=sha(args.recipe),files=files,backup_modified=False,gpu_actions=False)
        write_json(output/'complete.json',receipt)
        print('RESTORE_COMPLETE',json.dumps(receipt),flush=True)
    except Exception as error:
        write_json(output/'failure.json',dict(error=repr(error),spec_sha256=SPEC_SHA))
        raise


if __name__=='__main__':main()
