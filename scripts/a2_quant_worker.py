"""One allocated GPU, one pinned parent; scientific failures never cancel peers."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

from scripts.a2_quant_io import lock, write
from scripts.a2_remaining import read_locked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--config-sha256', required=True)
    p.add_argument('--dataset', required=True); p.add_argument('--parent', required=True)
    p.add_argument('--phase', choices=('baseline','rtn','gptq','cal1','downstream'), required=True)
    p.add_argument('--bits', type=int); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); cfg = read_locked(dict(path=a.config,sha256=a.config_sha256))
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import numpy as np
    from safetensors.torch import save_file, load_file
    from efficient_llm.a2.model import load_parent, bypass, round_to_saved_precision
    from efficient_llm.a2.data import PackedRows
    from efficient_llm.a2.measure import score
    from efficient_llm.a2.quant import quantize_rtn, quantize_gptq, matrix_manifest, weight_digest, gptq, rtn
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True, warn_only=True)
    checks=[]; runs=[]; start=time.monotonic()
    def record(name, passed, detail):
        event=dict(test=name,passed=bool(passed),detail=detail,spec_sha256=cfg['spec_sha256'],
                   dataset=a.dataset,parent=a.parent,phase=a.phase)
        checks.append(event)
        write(a.output/'tests.json',checks)
        print('QUANT_CHECK',json.dumps(event),flush=True)
    def attempt(name, fn):
        try: return fn()
        except Exception as error:
            import traceback
            traceback.print_exc()
            record(name,False,dict(error=repr(error),action='continue other registered models/sets'))
            return None
    assets=read_locked(cfg['assets'])
    model=None
    def load():
        value=load_parent(assets,a.dataset,a.parent,'cuda:0').eval().requires_grad_(False)
        round_to_saved_precision(value)
        return value
    def evaluate(compression):
        for setname, ds in cfg['sets'][a.dataset].items():
            def evaluate_set():
                rows=PackedRows(ds['root'],ds['stream'],verify=False)
                result=score(model,rows,distributed=False)
                arr=np.asarray(result['per_sequence_nll'],dtype=np.float64)
                target=a.output/(compression+'-'+setname+'.json')
                result.update(run_id=f'Q-{a.dataset}-{a.parent}-{compression}', dataset=a.dataset,
                    parent=a.parent,compression=compression,set=setname,spec_sha256=cfg['spec_sha256'],
                    mean_from_sequences=float(arr.mean()),data_manifest=ds['manifest'],
                    code_commit=cfg['code_commit'],unverified=any(not x['passed'] for x in checks),
                    flags=[x for x in checks if not x['passed']])
                result['loss']=float(arr.mean())
                item=write(target,result);runs.append(dict(run_id=result['run_id'],set=setname,
                    nll=result['loss'],per_sequence_file=item,spec_sha256=cfg['spec_sha256']))
                with (a.output/'runs.jsonl').open('a') as f:f.write(json.dumps(runs[-1])+'\n')
                reference=cfg['references'].get(a.dataset+'/'+a.parent,{})
                if setname=='final' and compression=='parent' and 'parent' in reference:
                    delta=result['loss']-reference['parent']
                    record('QT1',abs(delta)<=1e-6,dict(difference=delta,reference=reference))
                if setname=='final' and compression=='pre-S1' and 'pre' in reference:
                    baseline=json.loads((a.output/'parent-final.json').read_text())['loss']
                    delta=(result['loss']-baseline)-(reference['pre']-reference['parent'])
                    record('QT2',abs(delta)<=1e-6,dict(damage_difference=delta,reference=reference))
                print('QUANT_VAL',json.dumps(runs[-1]),flush=True)
            attempt('evaluation:'+compression+':'+setname,evaluate_set)
    def save_quant(compression):
        names={row['name'] for row in matrix_manifest(model)}
        target=a.output/(compression+'.safetensors')
        save_file({n:v.detach().cpu().bfloat16().contiguous() for n,v in model.state_dict().items() if n in names},str(target))
        return write(a.output/(compression+'-weights.json'),dict(weights=lock(target),matrices=matrix_manifest(model),
            spec_sha256=cfg['spec_sha256'],dataset=a.dataset,parent=a.parent,compression=compression,
            full_state_sha256=weight_digest(model),flags=[x for x in checks if not x['passed']]))
    def compression_work(compression):
        nonlocal model
        model=load()
        matrices=matrix_manifest(model)
        record('QT7',len(matrices)==168 and sum(x['weights'] for x in matrices)==1610612736,
               dict(matrices=matrices,count=len(matrices),weights=sum(x['weights'] for x in matrices)))
        if compression.startswith('rtn'):
            quantize_rtn(model,int(compression[-1]),record)
        else:
            bits=3 if compression.endswith('cal1') else int(compression[-1])
            record('QT6_inherited',cfg['calibration_passed'],dict(manifest=cfg['calibration_manifest'],
                action='retain exact specified prefixes; flag affected calibration-dependent results'))
            offset=128 if compression.endswith('cal1') else 0
            rows=PackedRows(cfg['calibration_root'],'heal',count=128,order=list(range(offset,offset+128)),verify=False)
            if a.dataset=='dclm' and a.parent=='S72' and compression=='gptq4':
                w=model.model.layers[0].self_attn.q_proj.weight
                q=gptq(w,torch.eye(w.shape[1],device=w.device),4)
                expected=rtn(w,4)
                record('QT4',torch.equal(q,expected),dict(max_abs=float((q-expected).abs().max())))
            quantize_gptq(model,rows,bits,record)
        artifact=save_quant(compression)
        evaluate(compression)
        if a.dataset=='dclm' and a.parent=='S72' and compression=='gptq3':
            expected=read_locked(artifact)['full_state_sha256']
            del model;model=load()
            rows=PackedRows(cfg['calibration_root'],'heal',count=128,verify=False)
            quantize_gptq(model,rows,3,record)
            actual=weight_digest(model)
            record('QT5',expected==actual,dict(first=expected,second=actual))
        del model;model=None;torch.cuda.empty_cache()
    if a.phase=='baseline':
        model=attempt('load_parent',load)
        if model is not None:
            matrices=matrix_manifest(model)
            record('QT7',len(matrices)==168 and sum(r['weights'] for r in matrices)==1610612736,
                   dict(matrices=matrices))
            if a.dataset=='dclm' and a.parent=='S72':
                def initial_tests():
                    from efficient_llm.a2.quant import grid_test
                    w=model.model.layers[0].self_attn.q_proj.weight
                    for bits in (3,4):
                        test=grid_test(w,bits)
                        record('QT3',test['cardinality'] and test['idempotent'],dict(bits=bits,**test))
                        q=gptq(w,torch.eye(w.shape[1],device=w.device),bits)
                        expected=rtn(w,bits)
                        record('QT4',torch.equal(q,expected),dict(bits=bits,max_abs=float((q-expected).abs().max())))
                attempt('initial_quantizer_tests',initial_tests)
            evaluate('parent')
            with bypass(model,list(range(16,22))): evaluate('pre-S1')
    elif a.phase in ('rtn','gptq','cal1'):
        for compression in (('rtn4','rtn3') if a.phase=='rtn' else
                            ('gptq4','gptq3') if a.phase=='gptq' else ('gptq3-cal1',)):
            attempt(compression,lambda c=compression:compression_work(c))
            model=None;torch.cuda.empty_cache()
    else:
        def downstream():
            nonlocal model
            from eval.a2_likelihood import score_requests
            from eval.a2_request_bridge import digest
            from eval.a2_check import select_requests
            from eval.a2_stage1b_check import check_model
            model=load(); identity=dict(kind='a2_quant',dataset=a.dataset,parent=a.parent,
                assets=cfg['assets'],spec_sha256=cfg['spec_sha256'],compression='parent')
            if a.bits:
                folder=Path(cfg['results_root'])/'workers'/'gptq'/a.dataset/a.parent
                weight_record=lock(folder/f'gptq{a.bits}-weights.json');saved=read_locked(weight_record)
                tensor_path=read_locked_tensor(saved['weights'])
                tensors=load_file(str(tensor_path),device='cpu')
                state=model.state_dict()
                for name,value in tensors.items():state[name].copy_(value.to(state[name].device).float())
                record('quantized_reload',weight_digest(model)==saved['full_state_sha256'],saved['weights'])
                identity.update(compression=f'gptq{a.bits}',quantized=weight_record,full_state_sha256=weight_digest(model))
            prepared=read_locked(cfg['prepared'])
            def gate():
                ix, requests=select_requests(prepared)
                value=check_model(model,requests,4096)
                write(a.output/'gate.json',dict(value,selected_request_indices=ix,spec_sha256=cfg['spec_sha256']))
                record('representative_gate',value['passed'],value)
            attempt('representative_gate_process',gate)
            result=score_requests(model,prepared['requests'],batch_size=2,max_length=4096)
            result.update(batch_size=2,prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
                checkpoint_identity=identity,checkpoint_identity_sha256=digest(identity),
                packages={k:version(k) for k in ('torch','transformers','tokenizers')},spec_sha256=cfg['spec_sha256'])
            write(a.output/'identity.json',identity);write(a.output/'scores.json',result)
        attempt('downstream',downstream)
    # A late QT1/QT2/QT5 flag must also annotate already-written sequence results.
    flags=[x for x in checks if not x['passed']]
    for item in runs:
        row=read_locked(item['per_sequence_file'])
        row.update(unverified=bool(flags),flags=flags)
        item['per_sequence_file']=write(item['per_sequence_file']['path'],row)
    (a.output/'runs.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in runs))
    write(a.output/'complete.json',dict(execution_complete=True,checks=lock(a.output/'tests.json') if checks else None,
          runs=runs,elapsed_seconds=time.monotonic()-start,spec_sha256=cfg['spec_sha256']))


def read_locked_tensor(item):
    from efficient_llm.diagnostic_data import sha256
    if sha256(item['path'])!=item['sha256']:raise ValueError('Saved quantized weights changed')
    return Path(item['path'])


if __name__=='__main__': main()
