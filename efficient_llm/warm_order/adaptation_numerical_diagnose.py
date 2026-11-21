"""Bounded actual-parent inference diagnosis; no fitting or tolerance change."""
import argparse,json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from .adaptation_train import load_parent
from .adaptation_reservation import validate_binding
from .adaptation_analysis import native_trace,CANONICAL,SWAPPED
from .adaptation_runtime import measurement_runtime,training_runtime
from .data import Rows,sha
from .horizon_train import write_json


def difference(x,y):
    x=x.detach().cpu();y=y.detach().cpu();delta=(x-y).abs()
    tolerance=1e-4+1e-4*y.abs();bad=delta>tolerance
    return dict(max_abs=float(delta.max()),rms=float(delta.double().square().mean().sqrt()),
                mismatched=int(bad.sum()),elements=x.numel(),reference_max=float(y.abs().max()),
                exact=torch.equal(x,y))


@torch.no_grad()
def diagnose(config,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh diagnosis output required')
    validate_binding(config)
    torch.set_num_threads(config['threads'])
    model,optimizer,parent,names=load_parent(config['parent'],config['parent_hashes'])
    del optimizer,parent
    model.eval().requires_grad_(False)
    ids=np.load(config['calibration_indices'],allow_pickle=False)[:1]
    x=Rows(config['data'],model.c.length).batch(ids,'cpu')
    references={};report=dict(seed=config['row_reservation']['seed'],row_ids=ids.tolist(),cases={},
        torch=torch.__version__,device=torch.cuda.get_device_name(),tf32_at_start=torch.backends.cuda.matmul.allow_tf32,
        scope='One calibration row, inference only. Diagnose numerical discrepancy; no training or gate override.')
    for label,path in [('canonical',CANONICAL),('swap',SWAPPED)]:
        with measurement_runtime('cpu'):
            plain=model.hidden(x,path).clone()
            for backend in ('native','math'):
                with sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext():
                    for repeat in range(2):
                        h,trace=native_trace(model,x,path);nll=model.nll(x,path)
                        key=label+'/cpu_'+backend+'/'+str(repeat)
                        references[key]=(h,trace,nll)
                        report['cases'][key]=dict(hidden_vs_cpu_plain=difference(h,plain),nll=nll.tolist())
    model.to('cuda');x=x.to('cuda')
    gpu_previous={}
    for label,path in [('canonical',CANONICAL),('swap',SWAPPED)]:
        cpu_h,cpu_trace,cpu_nll=references[label+'/cpu_native/0']
        math_h,math_trace,math_nll=references[label+'/cpu_math/0']
        for deterministic in (False,True):
            for repeat in range(3):
                with training_runtime('cuda') if deterministic else nullcontext():
                    with measurement_runtime('cuda'):
                        h,trace=native_trace(model,x,path);nll=model.nll(x,path)
                        layers=[]
                        for ref,observed in zip(cpu_trace,trace):
                            layers.append(dict(block=ref['block'],fields={k:difference(observed[k],ref[k]) for k in
                                ('input','attention_input_normalized','attention_update','ffn_input_normalized','ffn_update','output')}))
                        key=label+'/cuda_math/deterministic'+str(deterministic)+'/'+str(repeat)
                        report['cases'][key]=dict(hidden_vs_cpu_native=difference(h,cpu_h),hidden_vs_cpu_math=difference(h,math_h),
                            nll_vs_cpu_native=difference(nll,cpu_nll),nll=nll.cpu().tolist(),layers=layers,
                            tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision())
                        if label in gpu_previous:report['cases'][key]['vs_previous_cuda']=difference(h,gpu_previous[label])
                        gpu_previous[label]=h.detach().cpu()
    out.mkdir(parents=True);write_json(out/'REPORT.json',report)
    write_json(out/'complete.json',dict(report_sha256=sha(out/'REPORT.json'),inference_only=True,training_updates=0))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print('ADAPTATION_NUMERICAL_DIAGNOSIS',json.dumps(diagnose(json.loads(Path(a.config).read_text()),a.out)),flush=True)
if __name__=='__main__':main()
