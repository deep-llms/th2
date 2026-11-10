"""Reuse F10 scoring numerics with explicit F11 output provenance."""
from eval import a2_family_f10_score as worker
from eval.a2_family_f11 import write

if __name__=='__main__':
    worker.write=write
    worker.main()
