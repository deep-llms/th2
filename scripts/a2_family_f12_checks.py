"""Family F6 identities and non-blocking scientific checks."""
import math
import json
from pathlib import Path
import traceback
from scripts.a2_quant_io import write as atomic_write

from eval.a2_family_f12 import SPEC_SHA


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    nonfinite=[]
    def clean(v,location=''):
        if isinstance(v,float) and not math.isfinite(v):
            nonfinite.append(dict(path=location,value=repr(v)));return None
        if isinstance(v,dict):return {k:clean(x,location+'/'+str(k)) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x,location+'/'+str(i)) for i,x in enumerate(v)]
        return v
    result=clean(dict(value,spec_sha256=SPEC_SHA))
    if nonfinite:result.update(nonfinite_values=nonfinite,unverified=True)
    return atomic_write(path,result)



class Checks:
    def __init__(self, path):
        self.path, self.items = Path(path), []

    def record(self, name, passed, detail=None):
        event = dict(test=name, passed=bool(passed), detail=detail, spec_sha256=SPEC_SHA)
        self.items.append(event)
        write(self.path, dict(checks=self.items))
        print('FAMILY_F12_CHECK', json.dumps(event), flush=True)
        return bool(passed)

    def attempt(self, name, function):
        try:
            return function()
        except Exception as error:
            traceback.print_exc()
            self.record(name, False, dict(error=repr(error), action='flag affected outputs; continue registered work'))
            return None
