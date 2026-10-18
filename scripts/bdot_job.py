"""Verify native Trainer completion for the generic sequential job runner."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',required=True,type=int);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();out=Path(a.output);receipt=Path(a.receipt)
    if out.exists() and any(out.iterdir()):p.error('Pilot jobs require fresh output directories')
    if receipt.exists():p.error('Receipt already exists')
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:p.error('Missing training command')
    subprocess.run(command,check=True)
    metrics=json.loads((out/'train_results.json').read_text())
    if metrics.get('global_step')!=a.steps:raise RuntimeError('Trainer did not reach the declared budget')
    if not list(out.glob('*.safetensors')):raise RuntimeError('Missing final model weights')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as handle:json.dump(dict(status='ok',global_step=a.steps,output_dir=str(out)),handle)

if __name__=='__main__':main()
