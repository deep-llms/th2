# A2/DOT data reconstruction on OHMV

Authorized 18 November 2026: prepare reusable historical data, no training.

- Source recipe: resources/a2_corpora_30b_20261029.json; exactly 216 consumed
  DCLM shards and 44 consumed FineWeb-Edu shards, immutable HF revisions and
  raw SHA256. Total raw bytes: 144,925,612,535. Controller #d batches <=10 shards.
- Same GPT-NeoX tokenizer, no BOS, appended EOS, explicit document starts,
  original content-hash heal/tuning/final split and ordered source prefixes.
  Recreate 30B-token packed corpus and complete-document seed2030 shuffle
  for each corpus. Compare actual content, order and Depth prefix hashes to
  historical receipts. Fresh paths/timings require fresh provenance manifests.
- DOT E2: original first2B DCLM heal tokens / 10M final prefix, 512-token rows,
  seed0 row permutation and explicit boundaries; compare every output hash.
- DOT E1/E1b KG and DOT-RL KG: original separate generators and split rules;
  compare every generated file with historical hashes before READY.
- CPU-only frozen worker: scripts/a2_dot_rebuild_data.py. Data root:
  /mnt/local/_data/deep-llms_th2/a2_dot_20261118. Output/proof root:
  /mnt/local/_outputs/deep-llms_th2/a2_dot_data_20261118_a01.
- No deletion, no GPU process signals, no model downloads or new training.
  Existing partial outputs are retained on failure. Never relaunch blindly.
- This scope restores DCLM/FWE and DOT data. CulturaX, SmolLM3 retokenization,
  and the broader Family mixed corpus remain separate preparations if needed.
  Data readiness alone does not authorize a new training arm.

Validation: 10 local CPU checks passed (source selection, batch coverage,
explicit boundaries, seed order, tamper rejection and legacy shuffle tests).
Full KG regeneration: E1/E1b 10 files and RL12 files match historical SHA.
The worker reruns the three reconstruction tests on-node before preparation.

Watcher: scripts/queue_watch.py, same thread, interval1800seconds, state
 temp/a2_dot_data_20261118_watch. Queue acknowledgement and actual delivery
are separate observations. Stop only after verified data/proofs/handoff.
