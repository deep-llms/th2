"""Persistent, spawn-only workers: one visible GPU and one resident model each."""
import contextlib
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
import traceback


def visible_devices(workers=None):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = [x.strip() for x in visible.split(',') if x.strip()] if visible else []
    if len(set(devices)) != len(devices) or any(x == '-1' for x in devices):
        raise ValueError('Invalid CUDA_VISIBLE_DEVICES allocation')
    if workers is None:
        workers = len(devices) or 1
    if type(workers) is not int or workers < 1:
        raise ValueError('Worker count must be positive')
    if workers > 1 and workers > len(devices):
        raise ValueError('Parallel evaluation requires an explicit visible GPU allocation')
    return devices[:workers] if devices else [None]


def options(cmd):
    if cmd[1:3] != ['-m', 'eval.paired'] or cmd[3] not in ('evaluate','position') or len(cmd[4:]) % 2:
        raise ValueError('Unexpected paired worker command')
    values = dict(zip(cmd[4::2], cmd[5::2]))
    if len(values)*2 != len(cmd[4:]) or any(not key.startswith('--') for key in values):
        raise ValueError('Invalid paired worker options')
    return cmd[3], values


def cache_key(cmd):
    _, opts = options(cmd)
    return opts['--checkpoint'], opts['--path']


class ModelCache:
    def __init__(self):
        self.key = None
        self.bundle = None

    def get(self, checkpoint, device, path, fresh=False):
        from .run import load_model
        import torch
        key = (checkpoint, device, path)
        loaded = fresh or self.key != key
        if loaded:
            self.bundle = None
            gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
            # A preceding FP32 job disables TF32; restore the process defaults
            # before selecting the next path, as a fresh interpreter would.
            if not hasattr(self, 'defaults'):
                self.defaults = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                                 torch.backends.cudnn.allow_tf32)
            precision, matmul, cudnn = self.defaults
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
            self.bundle = load_model(checkpoint, device, path)
            self.key = key
        return self.bundle, loaded


def perform(cmd, cache):
    from .run import evaluate
    from .position import evaluate as position
    from .artifacts import load_json
    command, opts = options(cmd)
    device = 'cuda:0' if opts['--device'].startswith('cuda') else opts['--device']
    # Precision/repeat measurements always reload, independently of cache state.
    fresh = '--selection' in opts
    bundle, loaded = cache.get(opts['--checkpoint'], device, opts['--path'], fresh=fresh)
    if command == 'evaluate':
        evaluate(opts['--checkpoint'], opts['--task'], opts['--suite'], opts['--output'], opts['--protocol'],
                 opts['--dataset-root'], opts['--manifest'], device, int(opts['--batch-size']), opts['--path'],
                 load_json(opts['--selection']) if '--selection' in opts else None, model_bundle=bundle)
    else:
        position(opts['--checkpoint'], opts['--sample'], opts['--output'], opts['--protocol'],
                 device, opts['--path'], model_bundle=bundle)
    return dict(model_loaded=loaded)


def worker(slot, gpu, inbox, outbox, handler=perform):
    # Spawn imports this module without importing torch; isolate before CUDA use.
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):
        os.environ.pop(key, None)
    cache = ModelCache()
    while True:
        task = inbox.get()
        if task is None:return
        ident, cmd, log = task
        start = time.monotonic()
        try:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, 'a', buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                extra = handler(cmd, cache)
            result = dict(id=ident, slot=slot, gpu=gpu, status='ok', elapsed_seconds=time.monotonic()-start, **(extra or {}))
        except BaseException:
            result = dict(id=ident, slot=slot, gpu=gpu, status='failed', elapsed_seconds=time.monotonic()-start,
                          error=traceback.format_exc())
        outbox.put(result)


class Pool:
    """Bounded scheduling, cache affinity, and cleanup of only owned children."""
    def __init__(self, devices, root, handler=perform):
        self.devices, self.root, self.handler = list(devices), Path(root), handler
        self.processes, self.inboxes, self.records = [], [], []
        self.cached = [None]*len(devices)
        self.context = mp.get_context('spawn')
        self.outbox = self.context.Queue()
        self.previous_signal = None

    def __enter__(self):
        self.previous_signal = signal.getsignal(signal.SIGTERM)
        def cancelled(*_):raise KeyboardInterrupt('Paired worker pool cancelled')
        signal.signal(signal.SIGTERM, cancelled)
        try:
            for slot, gpu in enumerate(self.devices):
                inbox = self.context.Queue()
                proc = self.context.Process(target=worker,args=(slot,gpu,inbox,self.outbox,self.handler))
                proc.start();self.inboxes.append(inbox);self.processes.append(proc)
        except BaseException:
            self.__exit__(None,None,None);raise
        return self

    def save(self):
        self.root.mkdir(parents=True,exist_ok=True)
        path=self.root/'parallel-status.json';temporary=path.with_suffix('.pending')
        temporary.write_text(json.dumps(dict(devices=self.devices,jobs=self.records),indent=2))
        os.replace(temporary,path)

    def run(self, commands):
        # Long groups first; prefer an already resident checkpoint on later dispatch.
        weight={'mmlu_continuation':100,'blimp':90,'hellaswag':80}
        pending=sorted(commands,key=lambda c:-weight.get(options(c)[1].get('--task'),0))
        outputs=[options(c)[1]['--output'] for c in pending]
        if len(set(outputs))!=len(outputs):raise ValueError('Duplicate parallel output')
        active={}
        while pending or active:
            for slot in range(len(self.devices)):
                if slot in active or not pending:continue
                index=next((i for i,c in enumerate(pending) if cache_key(c)==self.cached[slot]),0)
                cmd=pending.pop(index);ident=len(self.records);_,opts=options(cmd)
                record=dict(id=ident,slot=slot,gpu=self.devices[slot],status='running',output=opts['--output'],
                            checkpoint=opts['--checkpoint'],path=opts['--path'],task=opts.get('--task','position'))
                self.records.append(record);active[slot]=(ident,cmd)
                self.inboxes[slot].put((ident,cmd,str(Path(opts['--output']).with_suffix('.worker.log'))))
            self.save()
            try:result=self.outbox.get(timeout=1)
            except queue.Empty:
                if any(not p.is_alive() for p in self.processes):raise RuntimeError('Paired worker exited unexpectedly')
                continue
            slot=result['slot']
            if slot not in active or result['id']!=active[slot][0]:raise RuntimeError('Unexpected worker result')
            ident,cmd=active.pop(slot);self.records[ident].update(result);self.save()
            if result['status']!='ok':raise RuntimeError(result['error'])
            self.cached[slot]=cache_key(cmd)

    def __exit__(self, *exc):
        # Do not signal groups/unknown GPU PIDs; these Process objects are ours.
        for inbox in self.inboxes:inbox.put(None)
        if not exc or exc[0] is None:
            deadline=time.monotonic()+5
            for proc in self.processes:proc.join(timeout=max(0,deadline-time.monotonic()))
        for proc in self.processes:
            if proc.is_alive():proc.terminate()
        for proc in self.processes:
            proc.join(timeout=10)
            if proc.is_alive():proc.kill();proc.join(timeout=10)
        for inbox in self.inboxes:inbox.close();inbox.cancel_join_thread()
        self.outbox.close();self.outbox.cancel_join_thread()
        if self.previous_signal is not None:signal.signal(signal.SIGTERM,self.previous_signal)
