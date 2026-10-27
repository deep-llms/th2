# Fixed KV-sharing placement follow-up

The longer recovery leaves a 16.37% PPL penalty with 50% KV reduction. Do not add MLP
pruning yet. Test whether protecting early layers improves conversion quality.

All models use the original 28 layers; zero-based layer indices:

| Output name | Consumers reuse preceding layer's KV | KV cache retained |
|---|---|---:|
| full50 | 1,3,5,...,27 | 50% |
| uniform25 | 3,7,11,15,19,23,27 | 75% |
| upper25 | 15,17,19,21,23,25,27 | 75% |

Every remaining layer produces its own KV. Q, attention, O, MLP, norms, RoPE,
loss and backend stay unchanged. Consumers have no K/V projections or duplicate
cache. The existing SharedAttention runtime already supports these fixed maps.
No adapters, auxiliary objectives, routing or custom kernel.

Each starts from exactly the **dense A-512** final checkpoint from the original
pilot, verifies its parent losses, then converts and starts a new 2,048-step
phase. Reuse the completed recovery study's 131,072 training prefixes and 256 eval
prefixes, identical seed/order/schedule/batch, frozen original teacher and all
student parameters train. The completed dense A recovery is an exact matched
control: same parent, tokens, order and recipe. Reusing its result saves a run.
The previous B continuation has different initialization and is historical only.

This uses three of the four authorized follow-up training jobs. Priority:
upper25, uniform25, full50. Evaluate/save every 256 steps, retain all checkpoints.
Reusing already hashed input tensors avoids repeating 20 minutes of tokenization.
Each fit should take roughly 9–10 minutes based on the completed pair, plus GPU
numerical gates, smoke, model loading and handoff. Timeout 2 hours/job is a bound,
not a time estimate.

CPU checks: explicit reference outputs/gradients for all maps, full50 identity
to existing conversion, full-vs-cached equivalence, exact cache bytes, save/reload,
and two-rank new conversion training. Actual-weight GPU gates for each layout
precede an eight-step eight-GPU smoke and all three real fits. Use unchanged
Accelerate config verification and exact burn identity/free-GPU/handoff workflow.

Judge upper25 versus uniform25 at equal cache size first. Compare each against
dense using paired document NLL and the prior 5% PPL screening gate. If a layout
passes, measure longer-context quality and actual whole-model latency/cache
before claiming useful efficiency. If none passes, do not keep adding broad
variants or assume more compression will help. Fixed late-layer sharing is
existing practice, so this is a conversion baseline, not novelty by itself.

Preflight evidence: 57 infrastructure checks pass, as do the three placement
checks (independent outputs/gradients/cache/reload, two-rank training, and
summary matching/rejection). CPU explicit-attention versus SDPA gradient relative
L2 is below 6e-7; same-SDPA independent decoder outputs are exactly equal.
The elementwise absolute gradient tolerance is 2e-5, matching the existing
reference test; a separate relative-L2 bound of 1e-5 prevents hiding a real drift.
GPU checks are mandatory before the real fits; CPU success alone is insufficient.

## Launch and initial health

Launch bca23aa acknowledged; health e1c58de at 08:07:11 SGT confirms all three
actual-weight GPU gates pass. Same-SDPA independent BF16 decoder logits agree
exactly; gradient relative L2 < 7e-9. FP32 explicit-math output/gradient gates,
fixed-input BF16 attention checks, cache capacity and cached decoding also pass.
Full-model cross-kernel BF16 differences are still reported and not accepted
as interchangeable; runtime remains the original SDPA implementation.

Eight-step eight-GPU smoke passes, parent/reload differences zero. Accelerate
cache copied/checked, exact known burns stopped and all GPUs verified free.
Upper25 reached step 1024/2048, validation PPL 21.2232; uniform25/full50 queued.
These are interim training-precision scores, not final BF16 comparisons.
Raw evidence and exact-file SHA receipts: temp/kv_placement_health01.log and
 temp/kv_placement_health01/. Existing supervisor handles final burns even on
failure. Existing watcher checks again at 08:23:46 SGT; no new monitor created.
