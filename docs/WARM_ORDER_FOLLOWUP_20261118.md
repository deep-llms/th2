# Follow-up decision after warm-order screening

Status: design registered after pilot node results; trainer/data reservation/evaluation/report and queue implemented;24CPU tests PASS. Node CPU admission55035cf PASS. GPU probe/launch pending. Existing watcher continues under the user's autonomous experiment authorization. Preserve the six completed pilot runs and their frozen source.

## What failed and what it implies

Pilot node result at20:03 SGT: dense mean4.076121, balanced-order learned4.140863, warm-up learned4.157342 nats. Both learned routers also lose slightly to their training-selected constant route. Even per-example oracle routing over the frozen R/W backbones loses to dense. Therefore fitting a more powerful head alone cannot close the observed backbone gap within this action set. This is a conclusion about this76M/196.6M-token conditional-suffix screening, not every dynamic-order method.

The next test should retain the trained dense model's useful computation while allowing small order changes and joint adaptation. It should not repeat shallow warm-up or tune the current head against test labels. No old A2/DOT training resumes.

## Proposed controlled experiment

Initialize all arms for each seed from that seed's verified dense pilot checkpoint. Keep512 tokens,32 observed prefix,480 scored targets and the same six blocks. Actions: canonical0123 and the three single adjacent swaps1023,0213,0132 of the four middle blocks. Every action executes the same six blocks; no STOP/repetition.

Eight fits: two seeds × four arms:

- C: canonical dense continuation.
- U: query-independent exploratory local-order mixture, with its exact probability schedule recorded.
- CF: joint backbone and causal prefix router, trained from full-information counterfactual action NLLs on training batches.
- PG: joint backbone and causal prefix router using a sampled policy-gradient objective; same-weight canonical NLL supplies an action-independent baseline.

Use a separate module and output root. Reset optimizer for every arm equally; this is a new fine-tuning experiment, not exact continuation of the pilot optimizer. Initial budget4000 updates ×32×512 =65,536,000 input tokens per arm, LR1e-4 backbone, LR1e-3 router,100-update warmup/cosine backbone. Router initial probabilities are fixed at[.85,.05,.05,.05] via zero final weight and log-prior bias. Exploration epsilon=.1 throughout. U uses the corresponding fixed initial mixture[.79,.07,.07,.07]. CF collects all four pre-update action losses every training batch, T=.02; PG computes a same-weight canonical baseline every batch. Final checkpoint only, no head epoch selection. Include all extra counterfactual/baseline forwards, prefix-feature passes and wall time in compute accounting; equal tokens do not imply equal total compute.

Routing observes prefix only. Head features detach from backbone for the policy objective. CF targets and PG rewards must come from the same pre-update backbone used for the sampled CE update. PG log probabilities must correspond to the actual sampling distribution including exploration, and the baseline must not depend on the sampled action. The body receives CE gradients only. Test analytic expected PG gradients against enumerated actions, plus causality, grouped gradients, source/data admission, real entry point, exact fullstate resume, and B200 numerical/throughput gates before launch. Do not mutate the completed pilot source.

Use fresh training windows disjoint from pilot backbone and router-fit windows, selected reproducibly from the same verified2B DCLM source. Keep document isolation. DCLM pilot test has now influenced hypothesis selection, so further use is explicitly exploratory; no untouched-confirmation claim. Reserve up to4096 DCLM final windows excluding every document represented in pilot test; fail below1024. Use4096 previously unused-in-this-study FWE final windows as a cross-domain check. New train set128000 windows is disjoint from BOTH pilot seeds and router-fit windows; post-train route selection uses2048 additional disjoint training windows. Train rows may come from documents with other training rows used earlier; do not claim training-document novelty. Lock exact split counts/hashes before launch. Do not optimize on either new evaluation set.

Primary comparisons: CF/PG versus C on identical held-out examples; versus U to distinguish conditional routing from order noise. Report the learned route distribution and best training-selected constant route. If the policy becomes effectively canonical, report that rather than calling it successful dynamic routing. Two-seed agreement and paired document-cluster intervals are required for a promising screening result; matched-compute larger-scale confirmation remains a later gate.

## Next actions

First finish the pilot's result-only export, local replay and human report. Then implement this follow-up, resolve routine scientific choices autonomously, test, benchmark and queue only after fresh ownership and receipt reconciliation. Known native burn is now restored; inspect exact new worker identities before any later authorized reclaim. Preserve infrastructure and all existing outputs. Only small proof/results return locally, all parameter/state files stay on node.

CPU implementation progress: `efficient_llm/warm_order/local_policy.py`, all19 warm-order regression tests PASS, including five focused tests in `tests/warm_order/test_local_policy.py` PASS: analytic enumerated PG expectation including exploration, sampling RNG restore, detached features/costs, grouped backbone gradients and prefix causality, fail-closed arguments/initial prior. No production runtime or trainer/resume validation is implied.

Locked node inputs (20:29SGT18Nov): prepared manifest SHA b50fb87f7399482e441b18cc04e186f8a8cbcb82a180ed825c9ae9b72b99ef81;128000 fresh train rows,2048 disjoint train-domain route-fit rows,3475 DCLM test windows with no pilot-test document overlap,4096 FWE cross-domain windows. All golden data and two initial dense checkpoint hashes PASS. Fullstate restores body/head optimizers, sampler RNG, global RNGs, step/counts and immutable recipe; changed LR/exploration/source/data refused. Final checkpoint only. GPU probe first40 CF-s0/PG-s0 belong to the real4000-step schedule and must resume exactly. Alleight train/eval → array replay/report →40 checkpoint/metadata files verified in node backup → unique native handoff and owned-guard release are queued automatically. Nodeproduction gates remain required.
