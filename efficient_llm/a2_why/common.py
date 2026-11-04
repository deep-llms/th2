"""Why identities and non-blocking scientific checks."""
from contextlib import contextmanager
import json
from pathlib import Path
import traceback
from scripts.a2_quant_io import write as atomic_write

SPEC_SHA = '87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    return atomic_write(path, dict(value, spec_sha256=SPEC_SHA))



class Checks:
    def __init__(self, path):
        self.path, self.items = Path(path), []

    def record(self, name, passed, detail=None):
        event = dict(test=name, passed=bool(passed), detail=detail, spec_sha256=SPEC_SHA)
        self.items.append(event)
        write(self.path, dict(checks=self.items))
        print('WHY_CHECK', json.dumps(event), flush=True)
        return bool(passed)

    def attempt(self, name, function):
        try:
            return function()
        except Exception as error:
            traceback.print_exc()
            self.record(name, False, dict(error=repr(error), action='flag affected outputs; continue registered work'))
            return None
