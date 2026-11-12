"""DOT-pool: explicit process-level protocol, defaulting to the original pilot."""
import os

PILOT_SPEC_SHA='a0b41acecbd462db6f9277ebec7173fe242d3493d9fb7f2d77b127d88bc97f05'
E1B_SPEC_SHA='41aea5e294682e4ee703cff1df4e95a2de84662b7eba880eb862b08067ff2065'
SPEC_SHA=os.environ.get('DOT_POOL_SPEC_SHA256',PILOT_SPEC_SHA)
if SPEC_SHA not in (PILOT_SPEC_SHA,E1B_SPEC_SHA):
    raise ValueError('Unrecognized DOT-pool spec SHA')
