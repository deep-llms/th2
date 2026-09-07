"""Create/verify explicit local file manifests (SHA256 and byte size)."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def resolve_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Manifest paths must be relative to their root")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or escaping path: {relative}")
    return path


def fingerprint(root, relative):
    path = resolve_file(root, relative)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest}


def create(root, files):
    if not files or len(set(files)) != len(files):
        raise ValueError("Select a nonempty list of unique file paths")
    return {"version": 1, "files": [fingerprint(root, name) for name in files]}


def verify(root, manifest, strict=False):
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported manifest version")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest must contain files")
    seen = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid manifest entry")
        name, size, digest = item["path"], item["bytes"], item["sha256"]
        if not isinstance(name, str) or name in seen:
            raise ValueError("Invalid or duplicate manifest path")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            raise ValueError("Invalid size/hash in manifest")
        seen.add(name)
        actual = fingerprint(root, name)
        if actual["bytes"] != size or actual["sha256"] != digest.lower():
            raise ValueError(f"Size/hash mismatch: {name}")
    if strict:
        actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual_files != seen:
            raise ValueError("Directory contains files outside the manifest")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("create", "verify"):
        command = sub.add_parser(mode)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--manifest", required=True, type=Path)
        if mode == "create":
            command.add_argument("--files", required=True, nargs="+")
        else:
            command.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        if not args.root.is_dir():
            raise ValueError("Root directory does not exist")
        if args.mode == "create":
            manifest = create(args.root, args.files)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            with args.manifest.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
            print(f"Created manifest with {len(manifest['files'])} files")
        else:
            with args.manifest.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            print(f"Verified {verify(args.root, manifest, args.strict)} files")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Manifest FAILED: {error}\n")


if __name__ == "__main__":
    main()
