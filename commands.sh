#0
#th2-q359-dot-project-inventory-20261018-a01
set -euo pipefail

# Read-only inventory: no deletions, process signals, environment changes or GPU work.
TASK_PROJECT_DIR=/mnt/local/deep-llms_th2
test -d "$TASK_PROJECT_DIR"
python3 - "$TASK_PROJECT_DIR" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys

root = Path(sys.argv[1])
if root.is_symlink() or root.resolve() != root:
    raise SystemExit("Refusing an unexpected project-directory symlink")

def emit(kind, **values):
    print(kind + " " + json.dumps(values, sort_keys=True), flush=True)

def describe(path):
    info = path.lstat()
    kind = ("symlink" if stat.S_ISLNK(info.st_mode) else
            "directory" if stat.S_ISDIR(info.st_mode) else
            "file" if stat.S_ISREG(info.st_mode) else "other")
    return {"path": str(path.relative_to(root)), "type": kind,
            "bytes": info.st_size, "mode": oct(stat.S_IMODE(info.st_mode)),
            "modified_utc": datetime.datetime.fromtimestamp(
                info.st_mtime, datetime.timezone.utc).isoformat()}

emit("INVENTORY_START", job="th2-q359-dot-project-inventory-20261018-a01",
     utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
     hostname=socket.gethostname(), cwd=os.getcwd(), project_dir=str(root))
entries = sorted(root.iterdir(), key=lambda p: p.name)
errors = []
for entry in entries:
    try:
        description = describe(entry)
        emit("ROOT_ENTRY", **description)
        # Do not follow symlinks or inspect Git internals. Bound the second level.
        if description["type"] == "directory" and entry.name != ".git":
            children = sorted(entry.iterdir(), key=lambda p: p.name)
            emit("DIRECTORY_SUMMARY", path=entry.name, children=len(children),
                 shown=min(100, len(children)))
            for child in children[:100]:
                emit("CHILD_ENTRY", **describe(child))
    except OSError as error:
        errors.append({"path": entry.name, "error_type": type(error).__name__})

command = root / "commands.sh"
if command.is_symlink() or not command.is_file():
    raise SystemExit("Expected regular commands.sh is missing")
emit("INVENTORY_SUMMARY", root_entries=len(entries), errors=errors,
     commands_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),
     scope="Top-level metadata plus at most 100 children per directory; no file contents")
if errors:
    raise SystemExit("Inventory encountered filesystem errors; inspect summary")
print("DOT_TH2_PROJECT_INVENTORY_OK", flush=True)
PY
