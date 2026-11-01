"""Score one registered full-schedule Depth C student with the unchanged scorer."""
import argparse
from importlib.metadata import version
from pathlib import Path
from efficient_llm.a2_depth.common import Checks, read, write
from efficient_llm.diagnostic_data import sha256
from scripts.a2_quant_io import lock
from scripts.a2_remaining import read_locked
from eval.a2_request_bridge import digest


def identity(cfg, arm, checks):
    folder=Path(cfg['results_root'])/'heals'/f'ST1-dclm-{arm}-S1-12.8B-lr1e-3'
    result=read(folder/'result.json');recipe=read(folder/'recipe.json')
    expected=dict(dataset='dclm',arm=arm,block='S1')
    checks.record('registered_training',all(result.get(k)==v for k,v in expected.items())
        and recipe['steps']==48828 and len(result['lr_updates'])==48828
        and recipe['removed']==list(range(16,22)),dict(recipe=lock(folder/'recipe.json')))
    checks.record('training_flags',not result.get('unverified',True),result.get('flags'))
    checkpoint=folder/'final'
    # Bind all saved model/config/native-code bytes, including sharded weights.
    files={str(p.relative_to(checkpoint)):sha256(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()}
    if not files or not any(n.endswith('.safetensors') for n in files):
        raise ValueError('Student weights unavailable; no substitute checkpoint')
    return dict(kind='a2_depth_12p8B',spec_sha256=cfg['spec_sha256'],arm=arm,
        result=lock(folder/'result.json'),recipe=lock(folder/'recipe.json'),
        checkpoint=str(checkpoint),files=files,training_flags=result.get('flags',[]),
        source_manifest=cfg['source_manifest'])


def score(cfg,arm,output,device='cuda:0'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    checks=Checks(output/'checks.json')
    def work():
        from efficient_llm.a2.model import reload_student
        from eval.a2_likelihood import score_requests
        from eval.a2_check import select_requests
        from eval.a2_stage1b_check import check_model
        ident=identity(cfg,arm,checks);write(output/'identity.json',ident)
        prepared=read_locked(cfg['prepared'])
        model=reload_student(ident['checkpoint'],device)
        checks.record('kept_layers',model.config.a2_kept_layers==[i for i in range(24) if i not in range(16,22)])
        def gate():
            indices,requests=select_requests(prepared);value=check_model(model,requests,4096)
            write(output/'gate.json',dict(value,selected_request_indices=indices))
            checks.record('representative_gate',value['passed'],value)
        checks.attempt('representative_gate_process',gate)
        value=score_requests(model,prepared['requests'],batch_size=2,max_length=4096)
        value.update(batch_size=2,prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
            checkpoint_identity=ident,checkpoint_identity_sha256=digest(ident),
            packages={k:version(k) for k in ('torch','transformers','tokenizers')})
        return write(output/'scores.json',value)
    item=checks.attempt('score',work)
    write(output/'complete.json',dict(execution_complete=True,scores=item,
          coverage_complete=item is not None,checks=checks.items))
    return item


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=('P300','P1T','X72'),required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(read(a.config),a.arm,a.output)
