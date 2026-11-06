"""Family provenance and scoped scientific checks."""
import json
import os
import tempfile
from pathlib import Path
from efficient_llm.diagnostic_data import sha256 as sha

INPUT_PREPARATION_SPEC_SHA = 'd73dbf6736e9cf9c26facb9dc75522019280f63321fbd35086a8192dcb8fd99c'
SPEC_SHA = '40089a98702a285d38799adcdfa89c72c7eec2e75e627cfef8552448f9a593c0'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        value = dict(value, spec_sha256=SPEC_SHA)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(f.name, path)
    return dict(path=str(path), sha256=sha(path))

class Checks:
    def __init__(self, path):
        self.path, self.items = Path(path), []

    def record(self, test, passed, detail=None):
        item = dict(test=test, passed=bool(passed), detail=detail)
        self.items.append(item)
        write(self.path, dict(checks=self.items))
        print('FAMILY_CHECK', json.dumps(item), flush=True)
        return bool(passed)

    def attempt(self, test, fn):
        try:
            return fn()
        except Exception as error:
            self.record(test, False, dict(error=repr(error), action='flag affected results; continue independent registered work'))
            return None
