"""Eight/any-GPU systems smoke on verified real text, NOT a scientific experiment.

Uses official pinned Base dimensions and real immutable Corpus batches. Memory
capacity is the full 262144 x 1024, but frozen values are deterministic random
fixtures and routing uses the small smoke vocabulary. Do not report these
losses as contextual-memory quality, or extrapolate to full-corpus hit rates.
The separate `ccm train` invocation tests the actual trainer/checkpoint path.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import itertools
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ccm.artifacts import state_hash
from ccm.cli import asset_identity, code_hash
from ccm.contracts import require, fresh_dir, write_json
from ccm.data import Corpus, collate, microbatches
from ccm.keys import Vocabulary
from ccm.model import MemoryLM, pilot_config
from ccm.runtime import MasterAdamW, model_inputs, optimizer_groups, to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'vocabulary', 'model-config', 'asset-manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--microbatch-segments', type=int, default=8)
    parser.add_argument('--loss-chunk', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    require(args.steps >= 3 and args.microbatch_segments > 0 and args.loss_chunk > 0, 'Invalid smoke settings')
    corpus = Corpus(args.data)
    require(corpus.meta['engineering'], 'Only a separately marked engineering corpus may enter this smoke')
    assets = asset_identity(args.model_config, args.asset_manifest)
    require(corpus.meta['provenance']['tokenizer'] == assets, 'Smoke tokenizer provenance mismatch')
    vocab = Vocabulary.load(args.vocabulary)
    require(vocab.metadata['corpus_hash'] == corpus.meta['manifest_hash'], 'Smoke vocabulary/data mismatch')
    config = pilot_config(args.model_config)
    rows = list(itertools.islice(corpus.optimizer_batches('common', 1017), args.steps))
    require(len(rows) == args.steps, 'Insufficient real smoke batches')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        world = dist.get_world_size()
        if rank == 0:
            fresh_dir(args.output)
        dist.barrier()
        results = []
        for phase, arm in (('common', 'base'), ('stage1', 'contextual'), ('stage1', 'grad'),
                           ('stage2', 'contextual'), ('stage2', 'grad')):
            table = None
            if arm == 'contextual':
                generator = torch.Generator().manual_seed(90210)
                table = torch.randn(262144, 1024, dtype=torch.bfloat16, generator=generator)
            torch.manual_seed(17)
            model = MemoryLM(config, arm, table=table, slots=262144).to(device=device, dtype=torch.bfloat16)
            del table
            model.set_phase(phase)
            model.train()
            require(sum(p.numel() for p in model.backbone.parameters()) == 344354816, 'Wrong backbone shape')
            require(model.backbone.lm_head.weight is model.backbone.model.embed_tokens.weight, 'Tying broken')
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            wrapped = DDP(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)
            opt = MasterAdamW(optimizer_groups(model, phase))
            for group in opt.inner.param_groups:
                group['lr'] = (5e-4 if phase == 'stage1' else 1.5e-4) * group['multiplier']
            if arm != 'base':
                initial_table_hash = state_hash({'table': model.table})
            torch.cuda.reset_peak_memory_stats()
            records = []
            for step, global_rows in enumerate(rows, 1):
                require(len(global_rows) >= world, 'Too few real segments for all ranks')
                target_count = sum(len(r['tokens'])-1 for r in global_rows)
                input_count = sum(len(r['tokens']) for r in global_rows)
                require(input_count == 262144, 'Smoke must use the actual pilot global input-token batch')
                local = list(microbatches(global_rows[rank::world], args.microbatch_segments))
                dist.barrier()
                torch.cuda.synchronize()
                start = time.monotonic()
                opt.zero_grad()
                totals = torch.zeros(3, device=device, dtype=torch.float64)
                for j, small in enumerate(local):
                    batch = to_device(collate(small, corpus.meta['special_ids'], vocab), device)
                    with wrapped.no_sync() if j < len(local)-1 else nullcontext():
                        out = wrapped(**model_inputs(batch), loss_chunk=args.loss_chunk)
                        (out['loss_sum'] * world / target_count).backward()
                        totals[0] += out['loss_sum'].detach().double()
                        totals[1] += (batch['slots'] >= 0).sum()
                        totals[2] += batch['input_ids'].numel()
                        del out, batch
                require(all(p.grad is not None for p in model.parameters() if p.requires_grad), 'Missing trainable gradient')
                norm = opt.step()  # also rejects nonfinite gradients
                dist.all_reduce(totals)
                torch.cuda.synchronize()
                elapsed = torch.tensor(time.monotonic()-start, device=device, dtype=torch.float64)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                records.append(dict(step=step, input_tokens=input_count, target_tokens=target_count,
                                    nll=float(totals[0])/target_count, grad_norm=norm,
                                    seconds=float(elapsed), input_tokens_per_second=input_count/float(elapsed),
                                    hit_rate=float(totals[1])/target_count,
                                    padding_fraction=1-input_count/float(totals[2])))
            if arm != 'base':
                changed = initial_table_hash != state_hash({'table': model.table})
                require(changed == (arm == 'grad'), 'Frozen/Grad table update semantics failed')
            hashes = [None]*world
            dist.all_gather_object(hashes, state_hash(model.state_dict()))
            require(len(set(hashes)) == 1, 'NCCL replicas diverged')
            peak = torch.tensor(torch.cuda.max_memory_allocated(), device=device, dtype=torch.int64)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            result = dict(phase=phase, arm=arm, success=True, steps=records,
                          peak_allocated_bytes_all_ranks=int(peak),
                          post_warmup_input_tokens_per_second=sum(r['input_tokens'] for r in records[1:]) /
                              sum(r['seconds'] for r in records[1:]))
            results.append(result)
            if rank == 0:
                write_json(Path(args.output)/f'{phase}_{arm}.json', result)
                print('REAL_DATA_PERFORMANCE_PASS', phase, arm, result['post_warmup_input_tokens_per_second'], flush=True)
            del wrapped, opt, model, group
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        if rank == 0:
            write_json(Path(args.output)/'complete.json', dict(success=True, engineering=True, world_size=world,
                       torch=torch.__version__, config=vars(args), source_code_hash=code_hash(),
                       corpus_hash=corpus.meta['manifest_hash'], assets=assets, cases=results,
                       caveat='Real text and full memory dimensions; random frozen values and smoke-vocabulary routing. Not scientific arm results.'))
            print('REAL_DATA_PERFORMANCE_SMOKE_COMPLETE', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
