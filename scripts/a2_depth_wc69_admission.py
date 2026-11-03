"""Read-only admission for the user-authorized new-node Depth rerun."""
from pathlib import Path
import socket

from scripts.gpu_process import inspect, process
from scripts.train_then_burn import APPROVED_BURNS, GUARD_HASH, argv, digest


def admission(cfg, output):
    restart = cfg['node_restart']
    host = 'thiennh-p6-wc69-worker-0'
    if restart.get('host') != host or socket.gethostname() != host:
        raise ValueError('Different replacement node')
    if Path(output).parent != Path('/mnt/local/_outputs/deep-llms_th2'):
        raise ValueError('Wrong replacement output root')
    script = '/tmp/llm_pretrain_burn.py'
    if digest(script) != APPROVED_BURNS[script]:
        raise ValueError('Unknown replacement-node burn code')
    if digest('/mnt/local/_gpu_guard/gpu_guard.sh') != GUARD_HASH:
        raise ValueError('Unknown native guard protocol')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():
        raise ValueError('Native guard already owned')
    current = inspect()
    expected = restart['inspection']
    topology = lambda snapshot: [(g['index'],g['uuid'],tuple(g['pids'])) for g in snapshot['gpus']]
    if (current['host'] != host or expected['host'] != host
            or [g['index'] for g in current['gpus']] != list(range(8))
            or any('B200' not in g['name'] or len(g['pids']) != 1 for g in current['gpus'])
            or topology(current) != topology(expected)):
        raise ValueError('Replacement GPU topology/workers changed')
    identities = {}
    for gpu in current['gpus']:
        worker = process(gpu['pids'][0]); parent = process(worker['ppid'])
        if parent['pid'] <= 1 or script not in argv(parent['pid']):
            raise ValueError('Unrecognized replacement burn launcher')
        identities[str(worker['pid'])] = worker
        identities[str(parent['pid'])] = parent
    if identities != restart['known_burn_identities']:
        raise ValueError('Replacement process start/command identity changed')
    return dict(inspection=current,identities=identities,
                predecessor_kind='verified operator burn on replacement node',
                historical_predecessor=cfg['predecessor'])
