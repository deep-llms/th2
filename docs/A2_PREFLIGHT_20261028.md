# A2 preflight — 28 October 2026 (Singapore time)

Data preparation completed **07:11 SGT**, with successful exit and SHA-verified
terminal artifacts. Parent weights are still downloading:26/54files passed actual
SHA checks at07:12SGT, including five complete weights. **No healing
run has started.** The 127-fit matrix remains gated on the real-parent tests and
whole-workload cost report in [the preregistration](A2_pilot_spec.md).

## Verified so far

- The earlier reduced-model one/eight B200 regression passed; this does not
  substitute for actual 1.7B parent checks. See
  [the exact receipt](results/a2_review_20261028/validation.json).
- Read-only inspection at 06:42 SGT: correct 6Y3O node, eight known communicating
  burn workers still owned by the earlier review supervisor; no new GPU job.
- Existing `cx_sampled_dd1b1513` English pool: pinned manifest matches; all 3,008
  files exist with expected sizes. Full content hashes subsequently passed in136.37seconds on the node.
  The pool has 36,495,424 train and 11,717 validation documents. No full-pool
  download is needed.
- Input lock: [12 parents and tokenizers](../resources/a2_inputs_20261028.json).
  Controller request `ab350b8` is acknowledged; actual weight SHA verification
  remains pending. Existing environments and driver are unchanged.
- 32 CPU tests pass after adding input verification, progress reporting, and
  streaming secondary readers. Sampling, packing, loss and schedules are unchanged.

## Secondary data and logged LR

[Secondary lock](../resources/a2_secondary_20261028.json) selects one fixed shard
per corpus before scoring: the first listed DCLM-baseline shard, the first
FineWeb-Edu **v1.0.0** shard, and the first original Nemotron-CC
`quality=high/kind=actual/kind2=actual` shard. Use the first 256 packed sequences
from each, never select shards by losses. Controller request `76f4c90` is
complete for all three files; node checksums and256-sequence preparations passed.

The original Nemotron source is available from [Common Crawl's official
listing](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html).
Its full 371,190,509-byte shard was downloaded on dev, hashed, and checked against
its single-part ETag before pinning the node download. This is the original
corpus, not Nemotron-CC-v2. These secondary sets may overlap pretraining, more
for longer-trained parents; they never choose an LR or decision label.

[Logged-LR evidence](../resources/a2_logged_lr_20261028.json) binds the exact
published archive and the log members matching the pinned parent run names:

| Dataset | Step 72,000 | Step 188,000 |
|---|---|---|
| DCLM | Not present in matching log member | Not present |
| FineWeb-Edu | Not present | Not present |
| Nemotron | Not present | **0.004**, exact logged step |

The matching logs begin at 200,050 / 196,050 / 172,050 respectively. The other
five requested values remain **documented by the WSD schedule and iteration,
not verified from logs**. Nearby logged values are preserved as context only,
never substituted. These diagnostics do not exclude a parent.

A local real-shard packing check also passed: exactly256×4096 GPT-NeoX tokens
from Nemotron, with EOS boundaries and stream hashes retained in
[the receipt](results/a2_preflight_20261028/data_probe.json). This is a data test,
not model evaluation.

## Execution boundary

`scripts/launch_a2_inputs.py` freezes source and launches a CPU-only job using the
existing sequential runner. The worker checks exact tokenizer bytes,
prepares secondary sets, verifies every English shard, then makes the fixed
1.2B-Qwen-token reservation and GPT-NeoX heal/tuning/final streams. It appends
EOS 0 after each document and carries packing remainders across documents.
A checksum failure, split overlap, or insufficient budget stops preparation;
old outputs and caches are preserved. Parent weights can finish downloading in
parallel: this CPU stage never loads them or claims their verification. Actual
model loaders check their pinned hashes before GPU use. The CPU stage does not
touch GPU burns or the guard.

CPU job launched06:58:22SGT, source2589987, run
`6y3o-a2-inputs-20261028-a01`. [07:01 health receipt](results/a2_preflight_20261028/input_status_0701.json):
source hashes match, no errors, all source shards verified, three secondary
streams complete, Nemotron matches dev byte-for-byte. Main data is not complete
yet. GPU burns were not stopped. No extra optional tokenizer tests are queued.

Once data passes: measure actual-model T8, finish real-parent numerical and
survey gates, include loading/scoring/saving/diagnostic time in both phase
estimates, and send the prelaunch report **before any healing**. No actual-model
throughput or phase completion estimate is claimed yet.

## Prepared next stage

[07:08 health receipt](results/a2_preflight_20261028/input_status_0708.json) confirms
continued sampling progress and advancing communicating burns. Five weight files
have expected sizes; sizes alone are not acceptance. No GPU preflight has launched.

`launch_a2_preflight.py` prepares a bounded initial cost gate: twelve discarded
updates of the actual1.7B-derived student/teacher at64×4096tokens, then a single
DCLM/P300 survey. Both use eight GPUs and the unchanged burn supervisor. Main
input completion, pinned spec/contracts, packed bytes and weight hashes are
checked before reclaiming GPUs. The Accelerate cache copy/env verification and
automatic burn handoff are retained. This launcher cannot start scientific heals.

34CPUtests passed; the two new tests check stale/changed/failed data refusal and
the exact bounded job manifest. Existing model/loss/data/schedule code is unchanged.
This is preparation, not a claim that actual-parent numerical or cost gates passed.

## Main data completed

[07:12 receipt](results/a2_preflight_20261028/input_status_0712.json): CPU job
finished07:11:32SGT in790.30seconds. Twelve exported artifacts and59frozen source
files verified independently. The fixed reservation reached1,200,002,619Qwen
tokens, and document-id/content-hash disjointness passed.

| Packed split | Sequences of4096 | Tokens |
|---|---:|---:|
| Healing |207,519|849,997,824|
| Tuning |1,024|4,194,304|
| Final validation |2,461|10,080,256|

All11,717validation documents were processed; only the final incomplete packed
sequence was dropped. Three secondary sets each contain256×4096tokens.
No change to the frozen sampling/packing recipe was needed.

All four DCLM weights and FWE/P300 now match pinned SHA hashes. FWE/P1T and later
parents remain pending. No duplicate download or GPU launch. Initial preflight
code is prepared/tested and waits for complete inputs; actual-model acceptance
and the full pilot cost estimate remain outstanding.
