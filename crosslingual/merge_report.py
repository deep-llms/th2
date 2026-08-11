"""Merge per-model cross-lingual JSONs into one comparison report.

Collects <base-dir>/*/<label>.json files produced by run_crosslingual.py /
run_parallel.py (possibly pulled from different machines) and writes the
combined markdown report with baseline deltas and T6 McNemar tests.

Usage:
  python crosslingual/merge_report.py \
      --base-dir temp/xling_results \
      --baseline-label baseline@10000 \
      --tests t6 t8 probe_b \
      --output temp/xling_results/report.md
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crosslingual.run_crosslingual import write_report


def main():
    parser = argparse.ArgumentParser(description="Merge xling results")
    parser.add_argument("--base-dir", required=True,
                        help="Dir containing <label>/<label>.json subdirs "
                             "(or loose <label>.json files)")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--tests", nargs="+", default=["t6", "t8", "probe_b"])
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Restrict/order the labels (default: all found)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = (glob.glob(os.path.join(args.base_dir, "*", "*.json"))
             + glob.glob(os.path.join(args.base_dir, "*.json")))
    all_results = {}
    for path in sorted(paths):
        label = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            all_results[label] = json.load(f)

    if args.labels:
        missing = [lb for lb in args.labels if lb not in all_results]
        if missing:
            raise SystemExit(f"labels not found: {missing} "
                             f"(have: {sorted(all_results)})")
        all_results = {lb: all_results[lb] for lb in args.labels}

    if args.baseline_label not in all_results:
        raise SystemExit(f"baseline label {args.baseline_label!r} not found "
                         f"(have: {sorted(all_results)})")

    out = args.output or os.path.join(args.base_dir, "report.md")
    report = write_report(all_results, args.baseline_label, args.tests, out)
    print(report)


if __name__ == "__main__":
    main()
