"""CPU artifact identity for A2 evaluation, before any native model loading.

The caller supplies reviewed locks from verified training receipts. This records
actual exported bytes for a later scorer; it does not claim a tensor/GPU test.
"""
import json
from pathlib import Path

from efficient_llm.a2.protocol import Run, layers_for, parent_ref
from efficient_llm.diagnostic_data import sha256
from scripts.a2_remaining import read_locked
from scripts.a2_remaining_heals import verify_result


def checkpoint_identity(*, assets_lock, dataset, parent, checkpoint=None,
                        result_lock=None, reference_lock=None, sets_lock=None):
    assets=read_locked(assets_lock)
    entry=assets[f'{dataset}/{parent}']
    if any(entry.get(k)!=v for k,v in parent_ref(dataset,parent).items()):
        raise ValueError('Unexpected native parent identity')
    required={'configuration_opensci.py','modeling_opensci.py'}
    if not required <= entry['code_files'].keys():raise ValueError('Missing pinned native code')
    for name,value in entry['code_files'].items():
        if sha256(Path(entry['code_dir'])/name)!=value:raise ValueError('Native source changed')
    if checkpoint is None:
        if any(x is not None for x in (result_lock,reference_lock,sets_lock)):
            raise ValueError('Original parent cannot consume a healed-fit receipt')
        root=Path(entry['weights_dir']).resolve();files=dict(entry['files'])
        if not {'config.json','model.safetensors'}<=files.keys():raise ValueError('Parent weights/config missing')
        kept=list(range(24));receipt=None;kind='parent'
    else:
        if any(x is None for x in (result_lock,reference_lock,sets_lock)):
            raise ValueError('Healed checkpoint requires locked result, provenance and frozen sets')
        row=read_locked(result_lock);ref=read_locked(reference_lock);sets=read_locked(sets_lock)
        run=Run(**{k:row[k] for k in ('dataset','arm','set','budget','lr','suffix')})
        verify_result(row,run,ref,row.get('rerun_of'),0)
        if (row['status']!='complete' or row['diverged'] or run.dataset!=dataset
                or run.parent!=parent or ref['assets_sha256']!=assets_lock['sha256']):
            raise ValueError('Wrong or unsuccessful training receipt')
        root=Path(checkpoint).resolve()
        if root.name!='final' or root.parent.name!=run.run_id:
            raise ValueError('Checkpoint is not the declared completed fit')
        if Path(result_lock['path']).resolve()!=root.parent/'result.json':
            raise ValueError('Result receipt is not next to the exported checkpoint')
        kept=[i for i in range(24) if i not in layers_for(sets['sets'][dataset][run.set])]
        config=json.loads((root/'config.json').read_text())
        if config.get('a2_kept_layers')!=kept or config.get('num_hidden_layers')!=len(kept):
            raise ValueError('Saved layer map differs from frozen removal set')
        files={name:sha256(root/name) for name in ('config.json','model.safetensors')}
        for name,value in entry['code_files'].items():
            if sha256(root/name)!=value:raise ValueError('Exported model source differs from native source')
            files[name]=value
        receipt=dict(result=result_lock,reference=reference_lock,sets=sets_lock,run_id=run.run_id)
        kind='healed'
    for name,value in files.items():
        if sha256(root/name)!=value:raise ValueError('Checkpoint file changed: '+name)
    return dict(schema=1,kind=kind,dataset=dataset,parent=parent,checkpoint=str(root),
        files=files,assets=assets_lock,code_dir=str(Path(entry['code_dir']).resolve()),
        code_files=entry['code_files'],kept_layers=kept,training_receipts=receipt)


def verify_identity(identity):
    """Reconstruct from original reviewed locks and require identical bytes."""
    if identity.get('schema') == 2:
        from eval.a2_stage1_checkpoint import verify_identity as verify_stage1
        return verify_stage1(identity)
    if identity.get('schema')!=1 or identity.get('kind') not in ('parent','healed'):
        raise ValueError('Unknown checkpoint identity')
    args=dict(assets_lock=identity['assets'],dataset=identity['dataset'],parent=identity['parent'])
    if identity['kind']=='healed':
        receipts=identity['training_receipts']
        args.update(checkpoint=identity['checkpoint'],result_lock=receipts['result'],
                    reference_lock=receipts['reference'],sets_lock=receipts['sets'])
    actual=checkpoint_identity(**args)
    if actual!=identity:raise ValueError('Evaluation checkpoint identity changed')
    return actual


def load_verified(identity, device='cpu'):
    """Use the same native parent/pruned reload paths as A2 validation."""
    import inspect
    import torch
    import transformers
    from efficient_llm.a2.model import assert_native, load_parent, reload_student
    if transformers.__version__!='4.57.1':raise ValueError('Native A2 runtime required')
    verify_identity(identity)
    if identity['kind']=='parent':
        model=load_parent(read_locked(identity['assets']),identity['dataset'],identity['parent'],device)
    else:
        model=reload_student(identity['checkpoint'],device)
        if model.config.a2_kept_layers!=identity['kept_layers']:raise ValueError('Loaded layer identity differs')
    assert_native(model)
    for cls,name in ((type(model),'modeling_opensci.py'),(type(model.config),'configuration_opensci.py')):
        if sha256(inspect.getfile(cls))!=identity['code_files'][name]:raise ValueError('Imported stale native code')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FP32 parameters required')
    verify_identity(identity)
    return model.eval().requires_grad_(False)
