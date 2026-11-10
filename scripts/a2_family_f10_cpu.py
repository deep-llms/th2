"""F10a CPU entry point; no model runs, downloads, or GPU mutation."""
import argparse
from eval.a2_family_f10 import f10a

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--f9-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();f10a(a.f9_root,a.output)
