"""Depth identities, reversible native pruning and non-blocking scientific checks."""
from contextlib import contextmanager
import json
from pathlib import Path
import traceback
from scripts.a2_quant_io import write as atomic_write

SPEC_SHA = '2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe'
DATASETS = ('dclm', 'fwe', 'nem')
PARENTS = ('S72', 'P300', 'S188', 'P1T')
SETS = tuple(f'W{k}' for k in range(1, 10)) + ('S2', 'S3', 'S4')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    return atomic_write(path, dict(value, spec_sha256=SPEC_SHA))


def layer_sets(recorded_groups):
    windows = {f'W{k}': list(range(2*k, 2*k+6)) for k in range(1, 10)}
    return {ds: dict(windows, **{s: sorted(2*g+i for g in recorded_groups[ds][s] for i in (0, 1))
                                for s in ('S2', 'S3', 'S4')}) for ds in DATASETS}


@contextmanager
def removed_layers(model, removed):
    """Use the tested A2 physical-pruning operation, restore the exact parent."""
    from efficient_llm.a2.model import prune
    original, count = model.model.layers, model.config.num_hidden_layers
    had_map = hasattr(model.config, 'a2_kept_layers')
    old_map = getattr(model.config, 'a2_kept_layers', None)
    try:
        yield prune(model, removed)
    finally:
        model.model.layers = original
        model.config.num_hidden_layers = count
        if had_map:
            model.config.a2_kept_layers = old_map
        elif hasattr(model.config, 'a2_kept_layers'):
            del model.config.a2_kept_layers


class Checks:
    def __init__(self, path):
        self.path, self.items = Path(path), []

    def record(self, name, passed, detail=None):
        event = dict(test=name, passed=bool(passed), detail=detail, spec_sha256=SPEC_SHA)
        self.items.append(event)
        write(self.path, dict(checks=self.items))
        print('DEPTH_CHECK', json.dumps(event), flush=True)
        return bool(passed)

    def attempt(self, name, function):
        try:
            return function()
        except Exception as error:
            traceback.print_exc()
            self.record(name, False, dict(error=repr(error), action='flag affected outputs; continue registered work'))
            return None
