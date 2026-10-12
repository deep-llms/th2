"""Separate entry point; existing eval commands and reporting stay unchanged."""
import argparse
from pathlib import Path
from . import artifacts, protocol


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--output',required=True)
    f.add_argument('--position-documents',type=int,default=2000,help='Freeze an explicit smaller count when the original pool is insufficient')
    plan=sub.add_parser('plan')
    for key in ('protocol','a10','a20','p20','dataset-root','manifest','output','position-sample','calibration'):plan.add_argument('--'+key,required=True)
    plan.add_argument('--batch-size',type=int,default=16);plan.add_argument('--device',default='cuda')
    plan.add_argument('--legacy-f',help='Optional historical Suite F directory mapping')
    plan.add_argument('--defer-paloma',action='store_true',help='Record unavailable Paloma; keep Z and probes complete')
    run=sub.add_parser('run');run.add_argument('--plan',required=True)
    ev=sub.add_parser('evaluate')
    for key in ('checkpoint','task','output','protocol','dataset-root','manifest'):ev.add_argument('--'+key,required=True)
    ev.add_argument('--suite',choices=['Z','F','probes','paloma'],required=True)
    ev.add_argument('--device',default='cuda');ev.add_argument('--batch-size',type=int,default=16)
    ev.add_argument('--path',choices=['production','reference','sdpa-bf16'],default='production')
    ev.add_argument('--selection')
    an=sub.add_parser('analyze')
    for key in ('root','protocol','output'):an.add_argument('--'+key,required=True)
    an.add_argument('--suite',choices=['Z','F','probes','paloma'],required=True)
    an.add_argument('--legacy-f',help='Explicit A20/P20 -> task -> directory JSON; F only')
    pf=sub.add_parser('freeze-position')
    for key in ('data','tokenizer','protocol','output'):pf.add_argument('--'+key,required=True)
    pe=sub.add_parser('position')
    for key in ('checkpoint','sample','protocol','output'):pe.add_argument('--'+key,required=True)
    pe.add_argument('--device',default='cuda');pe.add_argument('--batch-size',type=int,default=1)
    pe.add_argument('--path',choices=['production','reference','sdpa-bf16'],default='production')
    pa=sub.add_parser('analyze-position')
    for key in ('a10','a20','p20','protocol','output'):pa.add_argument('--'+key,required=True)
    audit=sub.add_parser('audit-data')
    for key in ('dataset-root','manifest','protocol','output'):audit.add_argument('--'+key,required=True)
    cal=sub.add_parser('calibrate');cal.add_argument('--output',required=True)
    cal.add_argument('--datasets',type=int,default=1000);cal.add_argument('--resamples',type=int,default=1000)
    cal.add_argument('--workers',type=int,default=1,help='Independent CPU workers; RNGs do not depend on this')
    args=p.parse_args()
    if args.command=='freeze':protocol.freeze(args.output,args.position_documents)
    elif args.command=='plan':
        from .run import plan
        print(plan(args.output,args.protocol,dict(A10=args.a10,A20=args.a20,P20=args.p20),args.dataset_root,args.manifest,
                   args.position_sample,args.calibration,args.batch_size,args.device,args.legacy_f,args.defer_paloma))
    elif args.command=='run':
        from .run import execute
        execute(args.plan)
    elif args.command=='evaluate':
        from .run import evaluate
        evaluate(args.checkpoint,args.task,args.suite,args.output,args.protocol,args.dataset_root,args.manifest,
                 args.device,args.batch_size,args.path,artifacts.load_json(args.selection) if args.selection else None)
    elif args.command=='analyze':
        from .analysis import suite
        frozen=protocol.read(args.protocol)  # Before opening ANY legacy paired outputs (T7).
        if args.legacy_f and args.suite!='F':p.error('Historical import is restricted to descriptive Suite F')
        protocol.write_json(args.output,suite(args.root,frozen,args.suite,artifacts.load_json(args.legacy_f) if args.legacy_f else None))
    elif args.command=='freeze-position':
        from .position import freeze
        from transformers import AutoTokenizer
        from train import load_text
        from datasets import load_dataset
        files=sorted(str(p) for p in Path(args.data).glob('*.parquet'))
        rows=load_dataset('parquet',data_files=files,split='train') if files else load_text(args.data)
        tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        freeze(rows,tok,args.output,protocol.read(args.protocol))
    elif args.command=='position':
        from .position import evaluate
        evaluate(args.checkpoint,args.sample,args.output,args.protocol,args.device,args.path,args.batch_size)
    elif args.command=='analyze-position':
        from .position import analyze
        protocol.write_json(args.output,analyze(dict(A10=args.a10,A20=args.a20,P20=args.p20),protocol.read(args.protocol)))
    elif args.command=='audit-data':
        from .audit import data
        protocol.write_json(args.output,data(args.dataset_root,args.manifest,protocol.read(args.protocol)))
    elif args.command=='calibrate':
        from .audit import calibration
        protocol.write_json(args.output,calibration(args.datasets,args.resamples,workers=args.workers))


if __name__=='__main__':main()
