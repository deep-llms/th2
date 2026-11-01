"""Full native Trainer checkpoints plus per-rank Depth audit continuity.

No numerical optimizer/scheduler override. A completion marker is written only
after all ranks saved their RNG and audit state. Incomplete checkpoints remain
on disk and are reported when selecting the latest usable state.
"""
from pathlib import Path
from transformers import TrainerCallback
from transformers.trainer_utils import SaveStrategy
from .common import read, write, SPEC_SHA
from efficient_llm.a2.protocol import lr_for_update


def enable_checkpoints(args, every=1000):
    args.save_strategy = SaveStrategy.STEPS
    args.save_steps = every
    args.save_total_limit = 2
    # KDModel is a plain nn.Module with tied embeddings. Native torch serialization
    # preserves ties; final scientific students still use the original BF16 save.
    args.save_safetensors = False
    args.save_only_model = False
    args.ignore_data_skip = False
    return args


def checkpoint_files(world):
    return ['pytorch_model.bin', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json'] + [
        'rng_state.pth' if world == 1 else f'rng_state_{rank}.pth' for rank in range(world)] + [
        f'depth-audit-rank{rank}.json' for rank in range(world)]


class CheckpointAudit(TrainerCallback):
    def __init__(self, trainer, trace, identity):
        self.trainer, self.trace, self.identity = trainer, trace, identity

    def on_save(self, args, state, control, **kwargs):
        root = Path(args.output_dir)/f'checkpoint-{state.global_step}'
        write(root/f'depth-audit-rank{args.process_index}.json', dict(
            identity=self.identity, global_step=state.global_step, world_size=args.world_size,
            rank=args.process_index, seen=self.trainer.seen, lrs=self.trace.lrs,
            nonfinite_microbatches=getattr(self.trainer, 'nonfinite_steps', 0)))
        self.trainer.accelerator.wait_for_everyone()
        if args.process_index == 0:
            required = checkpoint_files(args.world_size)
            present = all((root/name).is_file() and (root/name).stat().st_size for name in required)
            write(root/'depth-checkpoint.json', dict(complete=bool(present), identity=self.identity,
                global_step=state.global_step, world_size=args.world_size,
                files={name:(root/name).stat().st_size for name in required if (root/name).is_file()}))
        self.trainer.accelerator.wait_for_everyone()


def select_checkpoint(output, identity, world):
    """Return (latest complete matching checkpoint, rejected candidates).

    Callers must record rejections. This does not launch or restart a GPU job.
    """
    rejected=[]
    candidates=[p for p in Path(output).glob('checkpoint-*') if p.is_dir() and p.name[11:].isdigit()]
    for root in sorted(candidates,key=lambda p:int(p.name[11:]),reverse=True):
        try:
            marker=read(root/'depth-checkpoint.json')
            if marker['spec_sha256']!=SPEC_SHA or not marker['complete'] or marker['identity']!=identity or marker['world_size']!=world:
                raise ValueError('Incomplete checkpoint or changed recipe/world/spec')
            step=marker['global_step']
            if step!=int(root.name[11:]) or read(root/'trainer_state.json')['global_step']!=step:
                raise ValueError('Checkpoint step mismatch')
            for name in checkpoint_files(world):
                if (root/name).stat().st_size!=marker['files'][name]:raise ValueError('Checkpoint member size mismatch: '+name)
            for rank in range(world):
                audit=read(root/f'depth-audit-rank{rank}.json')
                if audit['identity']!=identity or audit['global_step']!=step or audit['world_size']!=world or audit['rank']!=rank:
                    raise ValueError('Rank audit identity mismatch')
                if len(audit['lrs'])!=step:raise ValueError('Incomplete LR trace')
                expected_lrs=[lr_for_update(i,identity['steps'],identity['lr']) for i in range(1,step+1)]
                if audit['lrs']!=expected_lrs:raise ValueError('Saved LR trace differs from full schedule')
                expected_batches=step*identity['global_sequences']//(world*identity['microbatch'])
                if len(audit['seen'])!=expected_batches or any(len(b)!=identity['microbatch'] for b in audit['seen']):
                    raise ValueError('Incomplete rank sample trace')
            return root,rejected
        except Exception as error:
            rejected.append(dict(path=str(root),error=repr(error),action='try earlier complete state'))
    return None,rejected


def restore_audit(trainer, trace, checkpoint):
    audit=read(Path(checkpoint)/f'depth-audit-rank{trainer.args.process_index}.json')
    trainer.seen=audit['seen']
    trainer.nonfinite_steps=audit['nonfinite_microbatches']
    trace.lrs=audit['lrs']
    return audit
