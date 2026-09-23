# CulturaX sampling → PCC data handoff

The user reports that the CulturaX subset has been downloaded on B200 and
`prepare_data.py sample` is still running there. **Do not access B200, restart
sampling, or consume those in-progress outputs.** This review inspected only
local source files and tested the sampler with tiny temporary synthetic data.
The running script and `commands.sh` were left unchanged.

## What the current sampler produces

`prepare_data.py` saves Hugging Face Arrow datasets containing a single `text`
column. It uses the tokenizer only to count tokens with
`add_special_tokens=False`; it does not save token IDs, insert EOS/BOS tokens,
pack contexts, or create a separate development set.

For logical tokenizer name `ORG/MODEL`, the layout is:

```text
<data-dir>/ORG_MODEL/train/en/shard_0000/
<data-dir>/ORG_MODEL/train/en/shard_0001/
...
<data-dir>/ORG_MODEL/eval/en/
```

`--tokenizer-name` determines the directory name even when `--tokenizer-path`
loads a differently named local snapshot. A folder name therefore does not
establish tokenizer revision. The raw subset/document sampling uses seed 42;
the probe's ordered training-context seed 20260922 is a separate stage. Preserve
the existing selected documents rather than rerunning raw sampling to change
that seed.

Whole documents cross token boundaries. In the current implementation, the
stopping condition uses the **combined train + eval** token count, so train
overshoot can leave eval below its nominal token budget. A local regression
fixture demonstrates this: documents of seven tokens with nominal train/eval
budgets of ten each produce fourteen train tokens and seven eval tokens. Do
not assume that the real `eval/en` contains exactly, or at least, 10M tokens.
Its usable context/target counts must be measured after sampling completes and
with the final pinned tokenizer and context policy.

The sampler assigns different document rows to train/eval. It does not deduplicate
identical text that may occur in different rows, and no new deduplication scheme
is required by the probe contract. Also, directory existence is not a completion
signal: train shards are published incrementally, and the sampler has no final
success marker. Confirm normal process completion before starting a training pipeline.

## Direct loading for experiments

Set `train_data` to the existing `train/en` directory and `val_data` to the
existing `eval/en` Dataset in `pcc.pipeline.example.json`. PCC consumes those
fixed splits directly. There is no `pcc prepare` command, explicit document-range
configuration, or required NPZ export. It neither resamples documents nor makes
another validation split. The low-level packed-context reader remains available
for existing NPZ inputs and test fixtures.

The shared loader loads training shards in sorted order and follows `train.py`'s
single-process recipe: tokenize without special tokens, concatenate within each
1000-document map batch, drop that batch's incomplete context, then shuffle
2048-token contexts with seed 20260922. The same recipe is used for validation.
Screen and full probe use token-identical prefixes of this shared ordered stream.
The final evaluation context is right-padded to the exact requested token count.
Every arm uses the same batch order, masks, initialization seed within its stage,
and global 32,768 input tokens per optimizer update. Matching the fixed raw set
and seed alone would not enforce these other details; the shared loader and
training loops do.

Tokenization, packing, and shuffle-index caches are internal Arrow files under
`<run>/data-cache/`. They do not modify the sampler directories. Preprocessing
covers the entire input split once per run before the global context shuffle;
screen/full stages reuse it. No separate data-preparation job is required.

The existing research budgets remain: 4,194,304 screen training tokens and 2M
screen validation tokens; 19,988,480 full training tokens and 10M full validation
tokens. An undersized split fails explicitly instead of duplicating examples or
borrowing from training. Since the sampler's nominal 10M eval output can be
short even before packing, a real run must check its usable count. Using a
smaller validation budget would need an explicit protocol change, not a silent
loader adjustment.

The pipeline checks full train/validation budgets and screen-prefix equality
before starting the screen. `input-check.json` records source metadata, exact
input/target counts, and ordered-context fingerprints for both stages. Use
`pipeline --check-only` with the same config to run model/data checks without
starting an experiment. This still preprocesses the complete supplied datasets;
use `--dry-run` for a config-only preview. Neither mode verifies completion of
an external sampling process or reads the optional test Dataset.

`test_data` is optional. Without it, the run finishes after validation gates;
a positive dev result is recorded as `validation_complete_test_not_supplied`.
There is no confirmatory test or scaling recommendation. If a separate fixed
test Dataset is supplied later, it is opened only after development decisions
are frozen and all gates pass. Validation is never automatically reused as test.
The locked research document's full confirmatory claim still requires that test.

Trainer loops save at exactly 128 screen or 610 full optimizer updates per arm.
`run_experiments.py` continues to queue complete experiments sequentially. Its
wall-clock timeout is a failure limit, not an equivalent optimizer-update cutoff.

The actual sampler outputs and pinned-model/tokenizer provenance remain unverified
locally; sampling assets and the research model have different recorded revisions.
No access to B200 or modification of the running sampler is needed for this code.
