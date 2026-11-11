# wc69 backup retention — 11 November 2026

Scope: `/mnt/local/_backup/deep-llms_th2` only. Originals under `_outputs`,
`_models` and `_data` are not deleted or changed. This is a reduction of the
existing node backup, not an independently durable off-node backup.

Keep all model/checkpoint weights, scientific results (including per-sequence
and token-level arrays), reports, replay evidence, raw result archives,
configuration/source snapshots, tokenizers, seeds, document orders, indices,
consumption traces, source revision/hash manifests and exclusion lists.

Remove only explicitly enumerated input files from the verified preservation
ledger, using `scripts/a2_prune_backup_inputs.py`. Never delete directories or
search generally for `.bin`, `.npy` or names containing `token`. The allowlist
covers mixture input text/token caches, packed mixture training inputs,
Family prepared FWE/CulturaX token caches, and Why 2031/2032 packed streams.
The original CulturaX packed input backup remains: the historical
`a2_rebuild_wc69_data.py restore` entry point depends on it. Metadata and
ambiguous files remain even when they could theoretically be regenerated.

Before any deletion, both each backup candidate and its original must match
the historical SHA256. The plan fixes every path, size, hash and file identity;
execution requires that exact plan SHA. All retained backup files must have
exactly unchanged path, inode, size, modification time and mode afterward.
This final retained-file check is an identity check, not another full weight
hash pass. The previous backup already performed a full SHA256 verification.

The historical `wc69-preservation-20261111-a01/files.jsonl` and `complete.json`
remain evidence of the original copy. After cleanup, consult
`wc69-input-prune-20261111-a01/plan.json`, `deleted.jsonl`, and `complete.json`
for intentional removals; the old ledger alone is no longer a current inventory.

## Reconstructing removed inputs

The fastest exact recovery on this node is copying the original path recorded
in the prune plan and checking its saved SHA256. Cleanup never removes those
originals. For recovery after losing the node, exact upstream bytes must still
be available; public availability alone does not guarantee byte identity.

Preserve `artifacts/a2_reproduction_20261031.tar.gz` on the development machine
and its index `resources/a2_data_reproduction_20261031.json`. It contains the
original source lists, revisions, hashes, tokenizer and frozen preparation code.
Use the actual frozen source and metadata in the backup, not a newer generic
sampling recipe:

- Why: `scripts/a2_why_prepare.py`, the backed-up
  `wc69-a2-why-data-20261104-a01/corpus_plan.json`, original FWE/DCLM source
  locks, document orders and per-stream completion manifests. Rebuild the
  original full corpus before applying the recorded seed/permutation/prefix.
- Family FWE: `scripts/a2_family_prepare_fwe.py`, its original corpus locks,
  shared tokenizer, document orders 2030/2031, offsets, consumption traces and
  completion manifests. Do not substitute the older tokenizer.
- Family mixture: frozen `a2_family_prepare_mix.py` and
  `a2_family_repair_mix.py`, `resources/a2_family_mixture_plan_20261107.json`,
  `resources/a2_family_mixture_content_correction_20261107.json`, and the
  **actual** a02 `mixture/effective_plan.json`. Preserve a01 source locks too:
  a02 reused completed a01 sources. Keep primary-exposure exclusion traces,
  source document orders, raw-file hashes and the stream manifest. The first
  failed a01 cache is historical, not an alternative approved mixture recipe.
- Family CulturaX retokenization: the frozen Family preparation code, original
  preserved CulturaX inputs/document identities, tokenizer and split metadata.

Compare regenerated content hashes to the saved expected hashes before use.
Timestamp/path-dependent provenance JSON will need new bindings even when
content is identical. DCLM/FWE have historical successful content rebuilds;
a complete independent second rebuild of the mixture is not claimed.
