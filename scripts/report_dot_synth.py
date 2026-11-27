"""Tables for the synthetic DOT benchmark (docs/DOT_SYNTH_20261124.md) from an extracted export's report.json."""
import json
import sys
from pathlib import Path


def main(report_path):
    report = json.loads(Path(report_path).read_text())
    tasks = sorted({v['task'] for v in report.values()})
    for task in tasks:
        rows = {k: v for k, v in report.items() if v['task'] == task}
        levels = sorted(next(iter(rows.values()))['results']['native'], key=int)
        print(f'\n## {task}: exact-match accuracy (%) per difficulty; columns after | are harder than training')
        print('arm | params | apps | setting | ' + ' '.join(f'k{k:>3}' for k in levels) + ' | depth@answer')
        for key, v in rows.items():
            for setting, res in v['results'].items():
                acc = ' '.join(f'{100 * res[k]["exact"]:5.1f}' for k in levels)
                depth = ' '.join(f'{res[k]["answer_depth"]:.1f}' for k in levels) if 'answer_depth' in res[levels[0]] else ''
                print(f'{v["arm"]:15s} {v["params"] / 1e6:4.2f}M {v["applications"]:3d} {setting:8s} | {acc} | {depth}')


if __name__ == '__main__':
    main(sys.argv[1])
