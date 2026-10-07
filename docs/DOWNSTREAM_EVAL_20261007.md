# Downstream evaluation of the completed proxy screen

Scope: A plus all ten completed P4–P7 variants at checkpoint-2500, seed 42,
FA4, zero-shot, no chat template. Nine English tasks, existing lm-eval 0.4.10
prompts and scoring. One checkpoint per GPU, up to eight concurrently. No training.

Launch: `5ac2fbf`, job `th2-tjx3-downstream-2500-20261007-a01`.
Root: `/mnt/local/_outputs/deep-llms_th2/downstream-2500-20261007-a01`.

## Data and environment

31 pinned files, 104,884,425 bytes, downloaded through controller job `f025c70`.
Location: `/mnt/local/_data/deep-llms_th2/downstream-english-20261007`.
Raw file hashes and repository revisions: resources/downstream_english_20261007.json.
All file hashes and processed document hashes verified on B200 at
16:19:32 Singapore, 7 October. Reference: resources/downstream_english_reference_20261007.json.
No Hub requests are made from GPU-shell jobs. Dataset loading uses explicit
local files because incomplete multilingual repository snapshots cannot rely
on Hugging Face's automatic config discovery. Task prompts/scoring are unchanged.

| Task | Scored examples per checkpoint |
|---|---:|
| hellaswag | 10,042 |
| xnli_en | 2,490 |
| belebele_eng_Latn | 900 |
| xstorycloze_en | 1,511 |
| paws_en | 2,000 |
| piqa | 1,838 |
| arc_easy | 2,376 |
| arc_challenge | 1,172 |
| winogrande | 1,267 |

XNLI uses validation per the installed task template; ARC uses test;
HellaSwag/PIQA/WinoGrande use validation; XStoryCloze uses eval;
Belebele/PAWS-X use test. All training splits are available for template
initialization but zero-shot evaluation uses no demonstrations.

Separate pinned `eval_fa4` environment installed via `4e07abc`. Core packages
match training: torch 2.14.1+cu130, Transformers 5.9.0, datasets 4.8.5,
Accelerate 1.13.0, FA4 4.0.0b33, CUTLASS DSL 4.8.0; lm_eval 0.4.10.
Pip dependency check and FA4 import passed. Training environments untouched.
Accelerate config copied/byte-verified; accelerate env confirms eight processes,
MULTI_GPU, BF16. Evaluation itself uses independent single-GPU processes.

## Acceptance gates and outputs

Local checks: ten existing groups passed; expanded real-template CLI test covers
all nine tasks. Additional local-file/hash-failure test passed after its fixture
was corrected to provide the training split required by the task template.
Real benchmark data/labels/counts validated locally and identically on B200.

GPU sequence: eleven checkpoint numerical checks (training-forward loss,
FA4 execution, document isolation, padding, active proxies, unchanged buffers,
SDPA same-weight logits), then eight examples per task for all eleven models,
then smoke validation, then full evaluation and final completeness validation.
The final validator checks step/backend/data/seed/few-shot settings, all expected
sample counts, finite accuracies and identical document/prompt/target hashes
across checkpoints. Smoke and full outputs are separate.

The existing supervisor stops only reverified authorized burn workers, waits and
requires free GPUs before work. It cleans only owned children and restores
communicating burns on success/failure. At 16:19:32 it had identified/stopped
workers 308656–308663 and was in the mandatory wait; later gate results pending.
Monitor evidence: artifacts/downstream-monitor-20261007-a01/.
Do not treat smoke metrics as scientific results.
