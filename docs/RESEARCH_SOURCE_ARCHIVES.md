# Lossless historical payload archives

To keep the execution source below the runner's 25 MiB limit, five historical
result JSON payloads are stored as gzip with exact original and compressed hashes
in `resources/research_source_archives_20261119.json`. Scientific values are
unchanged; original development copies remain locally. This is source packaging,
not deletion of experiment data or checkpoints.

Before using an old script that needs the original JSON paths, run:

```bash
python3 scripts/restore_research_archive.py
```

The helper verifies both compressed and restored hashes, restores missing files,
and refuses to overwrite an existing changed file. Do not add the restored plain
payloads back to an execution commit without rechecking the source-size cap.
