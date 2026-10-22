"""Run one approved Trainer stage and require its native completion receipt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arguments',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--expected-step',type=int,required=True)
    p.add_argument('--start-step',type=int,required=True)
    p.add_argument('--schedule-steps',type=int,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cfg=json.loads(a.arguments.read_text());output=Path(cfg['output_dir'])
    if a.receipt.exists():raise ValueError('Receipt already exists')
    if a.start_step==0 and output.exists() and any(output.iterdir()):raise ValueError('Fresh run output is not empty')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('Missing training command')
    subprocess.run(command,check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('training_results.py')),'check',
        '--output-dir',str(output),'--expected-step',str(a.expected_step),
        '--schedule-steps',str(a.schedule_steps),'--start-step',str(a.start_step),
        '--output',str(a.receipt)],check=True)


if __name__=='__main__':main()
