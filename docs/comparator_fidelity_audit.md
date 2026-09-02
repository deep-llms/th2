# Published-comparator fidelity audit

Date: 2026-09-01

This note records what is reproduced literally, what is an intentional Qwen
adaptation, and what must not be described as a reproduction. It is the source
of truth for naming the compressed-interface experiments.

## Audit rules

- The paper defines the method. Author-released code resolves implementation
  details that the paper omits, but an experimental-protocol difference is
  still reported explicitly.
- Every Qwen arm keeps the existing tokenizer, 1024-dimensional Transformer,
  flat full-vocabulary softmax, training data, optimizer, and BF16 protocol.
- `tied` means that the input and output paths reuse the same trainable token
  parameters. It does not imply that the final expanded input vector is the
  classifier row when the original method ties only a low-dimensional map.
- Parameter-matched Qwen arms use Qwen's initialization and no extra dropout
  unless the method itself requires otherwise. This is a controlled
  architectural adaptation, not a reproduction of the paper's optimizer or
  model family.

## Method-by-method result

| Method | Author code found | Local status | Fidelity boundary |
|---|---|---|---|
| Slim Embeddings | No method-specific release is linked by the paper; it says its Torch model was based on an earlier open-source LSTM implementation. | Correct exact-tied efficient-output adaptation. | The paper's output construction partitions `M` subvectors into `K` position-specific sets. `SlimEmbed` does the same, fixes a balanced random mapping before training, and `TiedSlimHead` implements the paper's `K` partial GEMMs followed by mapped summation. The Qwen initialization differs from the paper's LSTM-wide uniform initialization. |
| TT-Embedding | Yes: `tt-embedding/tt-embeddings`. Audited commit `5382bb7e8ba4021c458dbf6f747c233fc71050ef`. | TT-matrix algebra and modified Glorot initialization match the release; exact tying is an adaptation. | The paper's Transformer LM/NMT experiments use **two separate** TT decompositions for input and softmax. `tt_tied_r219` deliberately shares one TT object to satisfy this project's tied-interface protocol and must be named an exact-tied TT adaptation, not a protocol reproduction. |
| DeFINE | The authors' later DeFINE/DeLighT release is `sacmehta/delight`. Audited commit `cc499c53087cd248ee7a0d0b0e70c507e670cba3`. | Corrected to author-code HGT group algebra; fixed-tokenizer flat-softmax adaptation. | HGT owns one matrix and bias **per group**, then applies group-wise LayerNorm and GELU. The prior local shared-matrix implementation was wrong by another factor of `g` and has been removed. The paper uses DeFINE for input expansion and ties the low-dimensional input/output mapping; it does not tie the final expanded input table. |
| Distilled Embedding / Funneling | No author implementation is linked in the paper and no author release was identified in this audit. | Equation-3 architecture is correct; the current launcher is only a from-scratch nonlinear control. | `FunnelingEmbed` implements `E_tilde = ReLU(U) V^T` and shares that effective table at input/output. The published method first trains a dense model, learns `U,V` by reconstruction, then fine-tunes with the dense embedding reconstruction loss. A random-init run is not Distilled Embedding. |
| GroupReduce | No author code release was linked or identified. | Compact block factors and offline weighted-SVD/refinement procedure follow the paper; the end-to-end tied launcher is an adaptation. | Published GroupReduce compresses an already trained embedding and softmax separately. Frequency blocks, frequency-proportional ranks, row-weighted SVD, and Algorithm-1 reassignment are implemented. Algorithm 1 ranks move candidates by **lowest destination reconstruction error**, not largest improvement. Exact-budget rounding and the donor-size guard are recorded implementation choices. |
| P-VQ | No author code release was linked or identified. | Compact algebra and staged curriculum are implemented, with a disclosed scalable balanced-clustering approximation. | The paper pretrains dense, performs scheduled dense-table quantization, converts to a compact codebook plus per-token exclusive suffix, then fine-tunes. The default no longer adds a non-paper final reclustering event. The repository's capacity-repair k-means avoids the original balanced-k-means assignment cost at Qwen vocabulary scale and is not claimed bit-for-bit identical. |
| RankLift | Not prior work; this project's proposal. | Implemented as the proposed method. | It must beat the matched controls; no comparator-fidelity claim applies. |

## Source-derived checks in the test suite

- TT table materialization is compared with a direct port of the released
  `TensorTrain.full()` reshape/permute order and with an independent scalar
  TT definition. Core values are checked against the released modified-Glorot
  formula under an identical RNG seed.
- DeFINE expansion is compared with a literal port of the released
  group-first batched-matrix-multiplication path. Tests require the weight
  shapes `(groups, input_per_group, output_per_group)`.
- Slim and P-VQ heads are compared with explicitly materialized effective
  tables, including gradients.
- GroupReduce weighted SVD is checked at full rank, its compact table and tied
  head are checked against explicit reconstruction, and reassignment retains
  the paper's candidate ordering.
- All structural integer data—Slim mappings, P-VQ assignments, GroupReduce
  memberships, and TT shapes/ranks—must survive strict checkpoint loading.

## Experiment names that are safe to use

- `slim_tied_k4_m76484`: exact-tied Slim efficient-output adaptation.
- `tt_tied_r219`: exact-tied TT adaptation, not the paper's two-TT protocol.
- `define_tied_n112_k1724`: fixed-tokenizer DeFINE adaptation with a tied
  low-dimensional classifier; 19,579,556 interface parameters.
- `funneling_tied_r128`: from-scratch Funneling architectural control, not
  Distilled Embedding.
- `groupreduce_e2e_tied_g20`: from-scratch tied GroupReduce-style control.
  Use the dense conversion pipeline for canonical post-hoc GroupReduce.
- `pvq_*`: call a run P-VQ only when it uses dense pretraining, the curriculum,
  compact conversion, and compact fine-tuning. Random fixed codes are a P-VQ
  architecture ablation.

Primary sources:

- Slim Embeddings: <https://ojs.aaai.org/index.php/AAAI/article/view/12000>
- TT-Embedding paper: <https://aclanthology.org/2020.findings-emnlp.436/>
- TT-Embedding code: <https://github.com/tt-embedding/tt-embeddings>
- DeFINE: <https://openreview.net/forum?id=rJeXS04FPH>
- DeFINE/DeLighT author code: <https://github.com/sacmehta/delight>
- Distilled Embedding: <https://aclanthology.org/2020.findings-emnlp.250/>
- GroupReduce: <https://papers.neurips.cc/paper/8295-groupreduce-block-wise-low-rank-approximation-for-neural-language-model-shrinking>
- P-VQ: <https://ojs.aaai.org/index.php/AAAI/article/view/17688>
