# Routing literature recheck during teacher-order evaluation

18 Nov 2026, primary sources checked from the dev machine only. This is a
research note, not a replacement for the running frozen teacher-order protocol.

[Library of Layers](https://arxiv.org/html/2611.03526v1), already cited in our
DOT-RL spec, routes whole blocks per strided causal window. Routing uses earlier
windows; the first window uses the first token and a depth embedding. Its
selected probability scales a residual update, providing a gradient path despite
hard expert selection. Depth-conditioned normalization and progressive depth
training are additional ingredients. The paper reports competitive results;
this does not establish a gain in our small-model regime.

[Dynamic Layer Tying](https://arxiv.org/abs/2401.12819) uses RL for training-time
parameter-sharing decisions. This differs from per-input execution ordering.

[MACRO](https://arxiv.org/abs/2608.05872) searches task-specific routes over frozen
LLMs with a learned Markov policy. Its reported task-accuracy gains do not prove
our general-text NLL objective will improve.

Our inference: repeated weak oracle headroom suggests changing only a prefix
router is insufficient. If teacher-order is negative, prioritize a registered
window-routing study with full independent blocks, preserving causal attention
spaces within each window. Use a matched window-attention fixed-order baseline,
a full-attention dense baseline, and a random-route control; count parameters,
training work, active inference work and actual latency separately. Test strict
future-independence, window boundary behavior, padding, selected-probability
gradients, resume RNG and no KV mixing before launching. No implementation or
GPU run is authorized by these papers themselves; any follow-up remains under
the user's existing autonomous research authorization and project safety guide.
