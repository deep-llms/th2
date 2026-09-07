"""Harmless CPU job demonstrating the generic runner artifact contract."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump({"status": "ok", "value": sum(range(10)) - 3}, handle)
    print("CPU example complete", flush=True)


if __name__ == "__main__":
    main()
