"""Bind a wc69 rerun to completed reconstruction and fresh physical inspection."""
import argparse
import copy
from pathlib import Path
import socket

from efficient_llm.a2_depth.common import SPEC_SHA, read, write
from efficient_llm.a2_depth.reconstruction import rebuilt_manifest
from scripts.a2_quant_io import lock
from scripts.a2_remaining import read_locked
from scripts.gpu_process import inspect, process
from scripts.a2_prepare_corpora import sha

HOST='thiennh-p6-wc69-worker-0'


def assemble(base, reconstruction, restoration, cpu_proof, inspection, identities):
    cfg=copy.deepcopy(base)
    if cfg['spec_sha256']!=SPEC_SHA or inspection['host']!=HOST:
        raise ValueError('Wrong spec/replacement node')
    restored=read_locked(restoration)
    if restored['status']!='verified' or restored['spec_sha256']!=SPEC_SHA or not restored['files']:
        raise ValueError('Required assets restoration not verified')
    for item in restored['files']:
        if sha(item['destination'])!=item['sha256']:
            raise ValueError('Restored asset changed: '+item['destination'])
    receipt=read_locked(reconstruction)
    stream=Path(receipt['stream'])
    manifest_sha=rebuilt_manifest(stream,cfg['stream_history'],reconstruction)
    cfg['heal_streams']['dclm']=dict(path=str(stream),sha256=manifest_sha)
    cfg.update(stream_reconstruction=reconstruction,assets_restoration=restoration,
        arm_order=['X72','P300','P1T'],arm_order_authorization='User requested X72 first on replacement wc69, 03Nov2026, same thread',
        cpu_validation=cpu_proof,cpu_validation_resource='resources/a2_depth_wc69_cpu_20261103.json',
        node_restart=dict(host=HOST,inspection=inspection,known_burn_identities=identities))
    cfg['deviations']+=['Original C weights/per-sequence artifacts were not found in supplied backup; rerun on wc69 with X72 first per user. Historical P300/P1T scalar results retained separately.']
    return cfg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reconstruction',required=True)
    parser.add_argument('--restoration',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if socket.gethostname()!=HOST or Path(args.output).exists():raise ValueError('Wrong node or existing plan')
    live=inspect();identities={}
    for gpu in live['gpus']:
        for pid in gpu['pids']:
            worker=process(pid);parent=process(worker['ppid'])
            identities[str(pid)]=worker;identities[str(parent['pid'])]=parent
    cfg=assemble(read('resources/a2_depth_partc_plan_20261101.json'),lock(args.reconstruction),
        lock(args.restoration),lock('resources/a2_depth_wc69_cpu_20261103.json'),live,identities)
    write(args.output,cfg)
    print('PLAN_PREPARED',args.output,flush=True)


if __name__=='__main__':main()
