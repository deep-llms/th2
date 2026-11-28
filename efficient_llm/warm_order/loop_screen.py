"""Loop screen: dense baselines, loop mechanisms, AttnRes and their combinations.

Design and pre-registered decision rules: docs/LOOP_SCREEN_20261123.md (round 1)
and docs/LOOP_SCREEN2_20261123.md (round 2). All arms reuse the EmbeddingAnchor
embedding, native dense Block and shared tokenwise head, so only the depth
computation differs. Middle "path" names stored block identities; for loop arms
path[0] is the prelude, path[1:3] the looped core and path[3] the coda.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from efficient_llm.dot_pool.model import Block, Config as BlockConfig
from .embedding_anchor import EmbeddingAnchor

# blocks: stored blocks; loop: None|serial|memory|gated; mix: residual|attnres; var: variable passes
SPECS = {
    'dense4': dict(blocks=4, loop=None, mix='residual', var=False, steps=12288),
    'dense4_2x': dict(blocks=4, loop=None, mix='residual', var=False, steps=24576),
    'dense8': dict(blocks=8, loop=None, mix='residual', var=False, steps=12288),
    'serial': dict(blocks=4, loop='serial', mix='residual', var=False, steps=12288),
    'memory': dict(blocks=4, loop='memory', mix='residual', var=False, steps=12288),
    'gated': dict(blocks=4, loop='gated', mix='residual', var=False, steps=12288),
    'gated_var': dict(blocks=4, loop='gated', mix='residual', var=True, steps=12288),
    'attnres': dict(blocks=4, loop=None, mix='attnres', var=False, steps=12288),
    # round 2
    'memory_var': dict(blocks=4, loop='memory', mix='residual', var=True, steps=12288),
    'memory_2x': dict(blocks=4, loop='memory', mix='residual', var=False, steps=24576),
    'loop_attnres': dict(blocks=4, loop='serial', mix='attnres', var=False, steps=12288),
    'loop_attnres_var': dict(blocks=4, loop='serial', mix='attnres', var=True, steps=12288),
    'attnres8': dict(blocks=8, loop=None, mix='attnres', var=False, steps=12288),
    'attnres_2x': dict(blocks=4, loop=None, mix='attnres', var=False, steps=24576),
    # round 4: per-token random depth during training (copy-through, docs/LOOP_SCREEN4_20261123.md)
    'loop_attnres_tokvar': dict(blocks=4, loop='serial', mix='attnres', var='token', steps=12288),
    'gated_tokvar': dict(blocks=4, loop='gated', mix='residual', var='token', steps=12288),
    'loop_attnres_2x': dict(blocks=4, loop='serial', mix='attnres', var=False, steps=24576),
    'dense8_2x': dict(blocks=8, loop=None, mix='residual', var=False, steps=24576),
    # scale round (docs/SCALE_ROUND_20261123.md): 4x data, rows extend into the other seed's array
    'loop_attnres_4x': dict(blocks=4, loop='serial', mix='attnres', var=False, steps=49152),
    'attnres_4x': dict(blocks=4, loop=None, mix='attnres', var=False, steps=49152),
    'attnres8_4x': dict(blocks=8, loop=None, mix='attnres', var=False, steps=49152),
    'dense8_4x': dict(blocks=8, loop=None, mix='residual', var=False, steps=49152),
    'dense4_4x': dict(blocks=4, loop=None, mix='residual', var=False, steps=49152),
    # dynamic AttnRes (docs/MOE_DYN_ROUND_20261123.md): token-dependent depth queries, zero-init delta
    'dynattn': dict(blocks=4, loop=None, mix='attnres', var=False, steps=12288, dyn=True),
    'dynattn8': dict(blocks=8, loop=None, mix='attnres', var=False, steps=12288, dyn=True),
    'loop_dynattn': dict(blocks=4, loop='serial', mix='attnres', var=False, steps=12288, dyn=True),
    # token-level MoE in the loop line (docs/MOE_LOOP_ROUND_20261123.md): the MLPs of the listed stored
    # blocks become 8 x FF/2 experts, Mixtral top-2, shared across loop passes; bounded dynamic AttnRes
    'attnres_moe': dict(blocks=4, loop=None, mix='attnres', var=False, steps=12288, moe=(1, 2)),
    'loop_attnres_moe': dict(blocks=4, loop='serial', mix='attnres', var=False, steps=12288, moe=(1, 2)),
    'dense4_moe': dict(blocks=4, loop=None, mix='residual', var=False, steps=12288, moe=(1, 2)),
    'attnres8_moe': dict(blocks=8, loop=None, mix='attnres', var=False, steps=12288, moe=(1, 2, 3, 4, 5, 6)),
    'dynattn_tanh': dict(blocks=4, loop=None, mix='attnres', var=False, steps=12288, dyn='tanh'),
    # MoE variants (docs/MOE_VARIANTS_20261123.md): per-pass routers in the loop core (experts shared),
    # finer experts (16 x FF/4, top-4) and a DeepSeek-style shared expert (FF/2 + 8 x FF/4 top-2)
    'loop_attnres_moe_pr': dict(blocks=4, loop='serial', mix='attnres', var=False, steps=12288, moe=(1, 2), moe_pass=3),
    'attnres_moe16': dict(blocks=4, loop=None, mix='attnres', var=False, steps=12288, moe=(1, 2),
                          moe_k=16, moe_top=4, moe_ffdiv=4),
    'attnres_moe_shared': dict(blocks=4, loop=None, mix='attnres', var=False, steps=12288, moe=(1, 2),
                               moe_ffdiv=4, moe_shared=True),
    # MoE at 4x data (docs/MOE_SCALE_20261123.md)
    'attnres_moe_4x': dict(blocks=4, loop=None, mix='attnres', var=False, steps=49152, moe=(1, 2)),
    'loop_attnres_moe_4x': dict(blocks=4, loop='serial', mix='attnres', var=False, steps=49152, moe=(1, 2)),
    'attnres8_moe_4x': dict(blocks=8, loop=None, mix='attnres', var=False, steps=49152, moe=(1, 2, 3, 4, 5, 6)),
    # token-level DOT (docs/DOT_TOKEN_20261123.md): library of the 4 stored blocks, T steps per token
    'dot_route_t8': dict(blocks=4, loop=None, mix='dot', var=False, steps=12288, dot=dict(steps=8, mode='route', residual='multi')),
    'dot_fixed_t8': dict(blocks=4, loop=None, mix='dot', var=False, steps=12288, dot=dict(steps=8, mode='fixed', residual='multi')),
    'dot_random_t8': dict(blocks=4, loop=None, mix='dot', var=False, steps=12288, dot=dict(steps=8, mode='random', residual='multi')),
    'dot_route_res_t8': dict(blocks=4, loop=None, mix='dot', var=False, steps=12288, dot=dict(steps=8, mode='route', residual='plain')),
    'dot_route_t4': dict(blocks=4, loop=None, mix='dot', var=False, steps=12288, dot=dict(steps=4, mode='route', residual='multi')),
    # DOT sandwich, token-level router, no warm-up (docs/DOT_SANDWICH_20261123.md): T=block 0, middles 1..3, depth 3
    'sw_route': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                     sandwich=dict(depth=3, mode='route', residual='plain', recipient=True)),
    'sw_fixed': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                     sandwich=dict(depth=3, mode='fixed', residual='plain', recipient=True)),
    'sw_random': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                      sandwich=dict(depth=3, mode='random', residual='plain', recipient=True)),
    'sw_route_noid': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                          sandwich=dict(depth=3, mode='route', residual='plain', recipient=False)),
    'sw_route_multi': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                           sandwich=dict(depth=3, mode='route', residual='multi', recipient=True)),
    # router-training variants (user, 23 Nov 22:20 SGT): RL with explore/exploit, and try-every-module labels
    'sw_rl': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                  sandwich=dict(depth=3, mode='rl', residual='plain', recipient=True)),
    'sw_cf': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                  sandwich=dict(depth=3, mode='cf', residual='plain', recipient=True)),
    # a04 (docs/DOT_SANDWICH_20261123.md): routing as deviations from the fixed order, a larger middle library,
    # and stabilised versions of the user's two router-training ideas
    'sw_prior': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                     sandwich=dict(depth=3, mode='route', residual='plain', recipient=True, prior=True)),
    'sw6_route': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                      sandwich=dict(depth=3, mode='route', residual='plain', recipient=True, middles=6)),
    'sw6_prior': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                      sandwich=dict(depth=3, mode='route', residual='plain', recipient=True, middles=6, prior=True)),
    'sw_cf_soft': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                       sandwich=dict(depth=3, mode='cf', residual='plain', recipient=True, soft=True, balance=.1)),
    'sw_rl2': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                   sandwich=dict(depth=3, mode='rl', residual='plain', recipient=True, v2=True)),
    # a05 multi-path router training (user, 24 Nov): final-loss credit, several paths per token, bounded cost
    'sw_mpg': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                   sandwich=dict(depth=3, mode='mpg', residual='plain', recipient=True)),
    'sw_branch': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                      sandwich=dict(depth=3, mode='branch', residual='plain', recipient=True)),
    'sw_mpg_prior': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                         sandwich=dict(depth=3, mode='mpg', residual='plain', recipient=True, prior=True)),
    'sw_branch_prior': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                            sandwich=dict(depth=3, mode='branch', residual='plain', recipient=True, prior=True)),
    # a06 joint path + STOP policy (docs/DOT_STOP_SANDWICH_20261124.md): depth up to 4 middles
    'swst_dot_l02': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                         sandwich=dict(depth=4, mode='stop', residual='plain', recipient=True, lam=.02)),
    'swst_dot_l05': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                         sandwich=dict(depth=4, mode='stop', residual='plain', recipient=True, lam=.05)),
    'swst_fixed_l02': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                           sandwich=dict(depth=4, mode='stop', residual='plain', recipient=True, lam=.02, fixed_path=True)),
    'swst_budget_l02': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                            sandwich=dict(depth=4, mode='stop', residual='plain', recipient=True, lam=.02, budget=True)),
    'swst_route_d4': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                          sandwich=dict(depth=4, mode='stop', residual='plain', recipient=True, lam=0., stop=False)),
    'swst_fixed_d4': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                          sandwich=dict(depth=4, mode='stop', residual='plain', recipient=True, lam=0., stop=False,
                                        fixed_path=True)),
}
# a07 (25 Nov): the synthetic-benchmark recipe on language modelling (docs/DOT_SANDWICH_RECIPE_LM_20261125.md):
# tied sandwich trained at a random depth per step, measured at several depths and with per-token confidence halting
_LM_SW = dict(mode='stop', residual='plain', recipient=True, lam=0., stop=False)
SPECS.update({
    'swst_fixed_d4r': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                           sandwich=dict(_LM_SW, depth=4, fixed_path=True, rand_depth=(2, 6))),
    'swst_fixed_d4r8': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                            sandwich=dict(_LM_SW, depth=4, fixed_path=True, rand_depth=(2, 8))),
    'swst_loop1_d4r': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                           sandwich=dict(_LM_SW, depth=4, fixed_path=True, fixed_middle=0, rand_depth=(2, 6))),
    'swst_prior_d4gr': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                            sandwich=dict(_LM_SW, depth=4, prior=True, body_greedy=True, rand_depth=(2, 6))),
})
# t01 (25 Nov): DOT's T layer vs looped stacks without T, all with 4 stored blocks and 7 block applications at the
# trained depth (docs/DOT_T_VS_LOOPS_20261125.md)
SPECS.update({
    'swst_fixed_d3': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                          sandwich=dict(_LM_SW, depth=3, fixed_path=True)),
    'swst_fixed_d3r': dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288,
                           sandwich=dict(_LM_SW, depth=3, fixed_path=True, rand_depth=(2, 6))),
    'loop_cycle_d3': dict(blocks=4, loop=None, mix='schedule', var=False, steps=12288, schedule='cycle', depth=3),
    'loop_cycle_d3r': dict(blocks=4, loop=None, mix='schedule', var=False, steps=12288, schedule='cycle', depth=3,
                           rand_depth=(2, 6)),
    'loop_prc_d3': dict(blocks=4, loop=None, mix='schedule', var=False, steps=12288, schedule='prc', depth=3),
})
# f01 (25 Nov): improving fixed-order DOT; T gets per-recipient capacity or is made light
# (docs/DOT_FIXED_IMPROVE_20261125.md)
_F01 = dict(blocks=4, loop=None, mix='sandwich', var=False, steps=12288)
SPECS.update({
    'swst_fixed_d3_film': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, film=True)),
    'swst_fixed_d3_lora': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, lora_rank=32)),
    'swst_fixed_d3_fl': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, film=True, lora_rank=32)),
    'swst_fixed_d4_light': dict(_F01, sandwich=dict(_LM_SW, depth=4, fixed_path=True, t_ff_div=4)),
    'swst_fixed_d4_lightf': dict(_F01, sandwich=dict(_LM_SW, depth=4, fixed_path=True, t_ff_div=4, film=True)),
})
# f02 (25 Nov, user: shrink T's width or heads): compact T so compute moves to the middles
# (docs/DOT_COMPACT_T_20261125.md)
_NT384 = dict(t_width=384, t_heads=6, t_kv=2, t_ff=1024)
_NT256 = dict(t_width=256, t_heads=4, t_kv=2, t_ff=704)
SPECS.update({
    'swst_d3_nt384': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, **_NT384)),
    'swst_m4d4_nt384': dict(_F01, sandwich=dict(_LM_SW, depth=4, fixed_path=True, middles=4, **_NT384)),
    'swst_m4d5_nt384': dict(_F01, sandwich=dict(_LM_SW, depth=5, fixed_path=True, middles=4, **_NT384)),
    'swst_d3_nt256': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, **_NT256)),
    'swst_m4d5_nt256': dict(_F01, sandwich=dict(_LM_SW, depth=5, fixed_path=True, middles=4, **_NT256)),
    'swst_d3_h4': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, t_heads=4, t_kv=2)),
})
# f03 (25 Nov, user: share attention heads across layers instead of a separate T; also T as a nonlinear MLP only)
# (docs/DOT_SHARED_HEADS_20261125.md)
_SH = dict(blocks=4, loop=None, mix='schedule', var=False, steps=12288)
SPECS.update({
    'dense4_shrep': dict(_SH, schedule='dense', shared=dict(mode='replace')),
    'loop_cycle_d3_shrep': dict(_SH, schedule='cycle', depth=3, shared=dict(mode='replace')),
    'loop_cycle_d3_shcat': dict(_SH, schedule='cycle', depth=3, shared=dict(mode='concat')),
    'loop_cycle_d3_shvadd': dict(_SH, schedule='cycle', depth=3, shared=dict(mode='vadd')),
    'loop_cycle_d3_shoadd': dict(_SH, schedule='cycle', depth=3, shared=dict(mode='oadd')),
    'swst_fixed_d3_tmlp': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, t_mlp=True)),
})
# f04 (25 Nov): MLP-only T confirmed and given the saved compute (one more middle step), FiLM on the MLP-only T, and
# a looped baseline with 8 block applications (dense4 twice, 53.5M MAC) (docs/DOT_TMLP_20261125.md)
SPECS.update({
    'swst_d4_tmlp': dict(_F01, sandwich=dict(_LM_SW, depth=4, fixed_path=True, t_mlp=True)),
    'swst_d3_tmlp_film': dict(_F01, sandwich=dict(_LM_SW, depth=3, fixed_path=True, t_mlp=True, film=True)),
    'loop_cycle_a8': dict(_SH, schedule='cycle', apps=8),
})
# f05 (25 Nov): depth scaling at matched compute - MLP-only T at depth 5 vs a 9-application loop, 4-head T at depth 4
# vs the 8-application loops, and the prelude-recurrent-coda loop at 8 applications (docs/DOT_DEPTH_SCALE_20261125.md)
SPECS.update({
    'swst_d5_tmlp': dict(_F01, sandwich=dict(_LM_SW, depth=5, fixed_path=True, t_mlp=True)),
    'swst_d4_h4': dict(_F01, sandwich=dict(_LM_SW, depth=4, fixed_path=True, t_heads=4, t_kv=2)),
    'loop_cycle_a9': dict(_SH, schedule='cycle', apps=9),
    'loop_prc_a8': dict(_SH, schedule='prc', apps=8),
})
# f06 (25 Nov): the depth curve one step further at matched compute (docs/DOT_DEPTH_SCALE_20261125.md, f06 section)
SPECS.update({
    'swst_d6_tmlp': dict(_F01, sandwich=dict(_LM_SW, depth=6, fixed_path=True, t_mlp=True)),
    'swst_d5_h4': dict(_F01, sandwich=dict(_LM_SW, depth=5, fixed_path=True, t_heads=4, t_kv=2)),
    'loop_cycle_a10': dict(_SH, schedule='cycle', apps=10),
    'loop_cycle_a11': dict(_SH, schedule='cycle', apps=11),
})
# f07 (25 Nov): learning-rate fairness at the deepest points - lower peak lr for the deep loops and for DOT alike
# (docs/DOT_LR_FAIRNESS_20261125.md)
SPECS.update({
    'loop_cycle_a11_lr05': dict(SPECS['loop_cycle_a11'], lr_scale=.5),
    'loop_cycle_a11_lr07': dict(SPECS['loop_cycle_a11'], lr_scale=.7),
    'loop_cycle_a9_lr05': dict(SPECS['loop_cycle_a9'], lr_scale=.5),
    'swst_d6_tmlp_lr05': dict(SPECS['swst_d6_tmlp'], lr_scale=.5),
    'swst_d6_tmlp_lr07': dict(SPECS['swst_d6_tmlp'], lr_scale=.7),
    'swst_d5_tmlp_lr05': dict(SPECS['swst_d5_tmlp'], lr_scale=.5),
})
# f08 (25 Nov): higher peak lr for both sides (DOT's best lr in f07 was at the 1.0 edge) (docs/DOT_LR_FAIRNESS_20261125.md)
SPECS.update({
    'loop_cycle_a11_lr14': dict(SPECS['loop_cycle_a11'], lr_scale=1.4),
    'loop_cycle_a11_lr20': dict(SPECS['loop_cycle_a11'], lr_scale=2.),
    'swst_d6_tmlp_lr14': dict(SPECS['swst_d6_tmlp'], lr_scale=1.4),
    'swst_d6_tmlp_lr20': dict(SPECS['swst_d6_tmlp'], lr_scale=2.),
})
# f09 (25 Nov, user: why is d6 not cheaper than dense8?): a narrower MLP-only T, so T costs a fraction of a block; target
# dense8's compute with half its parameters (docs/DOT_NARROW_TMLP_20261125.md)
SPECS.update({
    'swst_d6_tmlp_ff1024': dict(_F01, sandwich=dict(_LM_SW, depth=6, fixed_path=True, t_mlp=True, t_ff=1024)),
    'swst_d7_tmlp_ff512': dict(_F01, sandwich=dict(_LM_SW, depth=7, fixed_path=True, t_mlp=True, t_ff=512)),
    'swst_d6_tmlp_ff512': dict(_F01, sandwich=dict(_LM_SW, depth=6, fixed_path=True, t_mlp=True, t_ff=512)),
    'swst_d8_tmlp_ff512': dict(_F01, sandwich=dict(_LM_SW, depth=8, fixed_path=True, t_mlp=True, t_ff=512)),
})
# f10 (25 Nov, user: train with optimized attention as deep2shallow did on B200): fused implicit-causal SDPA without
# deterministic mode, checked against the MATH runs at lr x1.4, plus dense8 at lr x1.4 (docs/DOT_FAST_ATTN_20261125.md)
SPECS.update({
    'swst_d6_tmlp_lr14_fa': dict(SPECS['swst_d6_tmlp_lr14'], fast_attn=True),
    'loop_cycle_a11_lr14_fa': dict(SPECS['loop_cycle_a11_lr14'], fast_attn=True),
    'dense8_lr14': dict(SPECS['dense8'], lr_scale=1.4),
    'dense8_lr14_fa': dict(SPECS['dense8'], lr_scale=1.4, fast_attn=True),
})
# f11 (26 Nov): fused attention from now on; best-lr comparison of the narrow-T DOT against dense8
# (docs/DOT_VS_DENSE8_20261126.md)
SPECS.update({
    'swst_d7_tmlp_ff512_fa': dict(SPECS['swst_d7_tmlp_ff512'], fast_attn=True),
    'swst_d7_tmlp_ff512_lr14_fa': dict(SPECS['swst_d7_tmlp_ff512'], lr_scale=1.4, fast_attn=True),
    'swst_d8_tmlp_ff512_lr14_fa': dict(SPECS['swst_d8_tmlp_ff512'], lr_scale=1.4, fast_attn=True),
    'dense8_lr07_fa': dict(SPECS['dense8'], lr_scale=.7, fast_attn=True),
})
# f12 (26 Nov): twice the training tokens (24,576 steps) for the narrow-T DOT vs dense8 at their best lrs, fused
# attention, policy-free static path (docs/DOT_VS_DENSE8_2X_20261126.md)
SPECS.update({
    'swst_d7_tmlp_ff512_lr14_fa_2x': dict(SPECS['swst_d7_tmlp_ff512_lr14_fa'], steps=24576),
    'swst_d8_tmlp_ff512_lr14_fa_2x': dict(SPECS['swst_d8_tmlp_ff512_lr14_fa'], steps=24576),
    'dense8_fa_2x': dict(SPECS['dense8'], fast_attn=True, steps=24576),
    'dense8_lr07_fa_2x': dict(SPECS['dense8'], lr_scale=.7, fast_attn=True, steps=24576),
})
# f13 (26 Nov): equal-parameter baselines at twice the tokens - looped stacks without T and dense4 (fused attention)
# (docs/DOT_VS_LOOPS_2X_20261126.md)
SPECS.update({
    'loop_cycle_a8_fa_2x': dict(SPECS['loop_cycle_a8'], fast_attn=True, steps=24576),
    'loop_cycle_a8_lr14_fa_2x': dict(SPECS['loop_cycle_a8'], lr_scale=1.4, fast_attn=True, steps=24576),
    'loop_prc_a8_fa_2x': dict(SPECS['loop_prc_a8'], fast_attn=True, steps=24576),
    'dense4_fa_2x': dict(SPECS['dense4'], fast_attn=True, steps=24576),
})
# f14 (26 Nov, user: is T needed, and how do other looped models connect their loops?): T ablations (no recipient id,
# unshared T) and stronger looped baselines (Huginn-style input injection, ITT-style step encoding), at about the
# narrow-T DOT d7's compute, fused attention (docs/DOT_T_ABLATION_20261126.md)
_D7 = SPECS['swst_d7_tmlp_ff512_lr14_fa']
SPECS.update({
    'swst_d7_tmlp_ff512_noid_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], recipient=False)),
    'swst_d7_tmlp_ff512_unshared_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], t_unshared=True)),
    'loop_cycle_a7_inj_lr14_fa': dict(_SH, schedule='cycle', apps=7, inject=True, lr_scale=1.4, fast_attn=True),
    'loop_cycle_a8_step_lr14_fa': dict(_SH, schedule='cycle', apps=8, step_scale=True, lr_scale=1.4, fast_attn=True),
    'loop_cycle_a8_lr14_fa': dict(SPECS['loop_cycle_a8'], lr_scale=1.4, fast_attn=True),
})
# f15 (26 Nov): stable additive input injection for the loop; the unshared narrow T confirmed and at 2x tokens; narrow
# shared T-MLPs interleaved into a non-looped 8-block transformer (docs/DOT_F15_20261126.md)
_U = SPECS['swst_d7_tmlp_ff512_unshared_lr14_fa']
_M8 = dict(_F01, fast_attn=True, sandwich=dict(_LM_SW, depth=8, middles=8, fixed_path=True, t_mlp=True, t_ff=512))
SPECS.update({
    'loop_cycle_a8_injadd_fa': dict(_SH, schedule='cycle', apps=8, inject_add=True, fast_attn=True),
    'swst_d7_tmlp_ff512_unshared_lr14_fa_2x': dict(_U, steps=24576),
    'swst_m8d8_tmlp_ff512_fa': _M8,
    'swst_m8d8_tmlp_ff512_lr14_fa': dict(_M8, lr_scale=1.4),
})
# f16 (26 Nov): equal-compute wider-FFN controls - is the gain from interleaving narrow MLPs, or just more FFN?
# (docs/DOT_FFN_CONTROL_20261126.md)
SPECS.update({
    'dense8_ff2624_fa': dict(SPECS['dense8'], ff=2624, fast_attn=True),
    'loop_cycle_a8_ff2176_fa': dict(_SH, schedule='cycle', apps=8, ff=2176, fast_attn=True),
    'loop_cycle_a7_ff2624_fa': dict(_SH, schedule='cycle', apps=7, ff=2624, fast_attn=True),
    'swst_m8d8_tmlp_ff512_fa_2x': dict(SPECS['swst_m8d8_tmlp_ff512_fa'], steps=24576),
})
# f17 (26 Nov, user: compare with the other looped methods too, incl. Inkling 2 Mini): the 23 Nov Inkling-style loop
# arms re-run with the current recipe (fused attention, both seeds) as baselines for DOT (docs/DOT_VS_INKLING_20261126.md)
SPECS.update({
    'gated_fa': dict(SPECS['gated'], fast_attn=True),
    'memory_fa': dict(SPECS['memory'], fast_attn=True),
    'loop_attnres_fa': dict(SPECS['loop_attnres'], fast_attn=True),
    'attnres8_fa': dict(SPECS['attnres8'], fast_attn=True),
})
# f18 (26 Nov): DOT's narrow T on top of the Inkling-like loop + AttnRes (a component test) vs more loop passes; and the
# recipient id under shuffled middle order (docs/DOT_F18_20261126.md)
SPECS.update({
    'loop_attnres_t512_fa': dict(SPECS['loop_attnres'], attnres_t=512, fast_attn=True),
    'loop_attnres_p4_fa': dict(SPECS['loop_attnres'], passes=4, fast_attn=True),
    'swst_d7_tmlp_ff512_rand_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], random_order=True)),
    'swst_d7_tmlp_ff512_rand_noid_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], random_order=True, recipient=False)),
})
# f19 (26 Nov): the FULL published looped methods, ported exactly and checked against the official code
# (docs/FULL_LOOPED_METHODS_20261126.md; parity docs/results/full_methods_parity_20261126.json). Each method uses its
# own block, loop, loss, optimizer and schedule shape; lr_scale multiplies our peak 6e-4 (published peaks are for
# 100-1000x larger batches). Compute-matched arms: about 8 layer applications at inference.
_FULL = dict(blocks=4, loop=None, mix='full', var=False, steps=12288, fast_attn=True)
_HUGINN = dict(kind='huginn', prelude=1, core=2, coda=1)
_OURO = dict(kind='ouro', layers=4)
_MOR = dict(kind='mor', recursions=3)
SPECS.update({
    'huginn_r3': dict(_FULL, full=dict(_HUGINN, mean_recurrence=3, backprop=3)),
    'huginn_r3_lr05': dict(_FULL, lr_scale=.5, full=dict(_HUGINN, mean_recurrence=3, backprop=3)),
    'huginn_r32': dict(_FULL, full=dict(_HUGINN, mean_recurrence=32, backprop=8)),  # published r=32, k=8
    'ouro_t2': dict(_FULL, full=dict(_OURO, steps=2)),
    'ouro_t2_lr05': dict(_FULL, lr_scale=.5, full=dict(_OURO, steps=2)),
    'ouro_t4': dict(_FULL, full=dict(_OURO, steps=4)),  # published T_max = 4
    'mor_mc3': dict(_FULL, full=dict(_MOR, base_depth=3)),  # Middle-Cycle 1+3+1, capacities 1, 2/3, 1/3
    'mor_mc3_lr20': dict(_FULL, lr_scale=2., full=dict(_MOR, base_depth=3)),
    'mor_mc2': dict(_FULL, full=dict(_MOR, base_depth=2)),  # parameter-matched 1+2+1 (about 6 applications)
})
# f20 (26 Nov): f19's best lr (x1.0) sat at a bracket edge for every method: extend the bracket, plus Huginn at
# r_bar = 2 (training compute matched to ~8 applications; evaluated at r = 2 and r = 3)
SPECS.update({
    'huginn_r3_lr14': dict(_FULL, lr_scale=1.4, full=dict(_HUGINN, mean_recurrence=3, backprop=3)),
    'huginn_r3_lr20': dict(_FULL, lr_scale=2., full=dict(_HUGINN, mean_recurrence=3, backprop=3)),
    'huginn_r2': dict(_FULL, full=dict(_HUGINN, mean_recurrence=2, backprop=2)),
    'ouro_t2_lr14': dict(_FULL, lr_scale=1.4, full=dict(_OURO, steps=2)),
    'ouro_t2_lr20': dict(_FULL, lr_scale=2., full=dict(_OURO, steps=2)),
    'mor_mc3_lr07': dict(_FULL, lr_scale=.7, full=dict(_MOR, base_depth=3)),
    'mor_mc3_lr05': dict(_FULL, lr_scale=.5, full=dict(_MOR, base_depth=3)),
})
# f21 (26 Nov): Huginn still improves at x2.0 (edge): x2.8 / x4.0 and the training-compute-matched r_bar = 2 at x2.0;
# DOT d7 unshared T at x2.0 (its best so far x1.4) so both sides get the higher-lr check; seed-1 confirmations
SPECS.update({
    'huginn_r3_lr28': dict(_FULL, lr_scale=2.8, full=dict(_HUGINN, mean_recurrence=3, backprop=3)),
    'huginn_r3_lr40': dict(_FULL, lr_scale=4., full=dict(_HUGINN, mean_recurrence=3, backprop=3)),
    'huginn_r2_lr20': dict(_FULL, lr_scale=2., full=dict(_HUGINN, mean_recurrence=2, backprop=2)),
    'swst_d7_tmlp_ff512_unshared_lr20_fa': dict(_U, lr_scale=2.),
})
# f23 (26 Nov, local A100s): Huginn's block (sandwich norms + sqrt(d) embedding scale) in DOT d7 unshared T and in
# dense8, to separate a better block from the loop / T mechanism (docs/FULL_LOOPED_METHODS_20261126.md)
SPECS.update({
    'dense8_lr10_fa': dict(SPECS['dense8'], fast_attn=True),
    **{f'dense8_hb_lr{t}_fa': dict(SPECS['dense8'], huginn_block=True, lr_scale=v, fast_attn=True)
       for t, v in (('10', 1.), ('14', 1.4), ('20', 2.), ('28', 2.8))},
    **{f'swst_d7_tmlp_ff512_unshared_hb_lr{t}_fa': dict(_U, huginn_block=True, lr_scale=v)
       for t, v in (('10', 1.), ('14', 1.4), ('20', 2.), ('28', 2.8))},
    # unshared but very narrow T (t_ff 128): about 26.0M params and 49M MACs, both below Huginn (26.4M, ~57M)
    **{f'swst_d7_tmlp_ff128_unshared_hb_lr{t}_fa': dict(_U, sandwich=dict(_U['sandwich'], t_ff=128), huginn_block=True,
                                                         lr_scale=v) for t, v in (('14', 1.4), ('20', 2.))},
    # T ablation under the Huginn block: the plain cyclic loop over the same 4 blocks (8 applications, no T)
    **{f'loop_cycle_a8_hb_lr{t}_fa': dict(SPECS['loop_cycle_a8'], huginn_block=True, lr_scale=v, fast_attn=True)
       for t, v in (('14', 1.4), ('20', 2.))},
    # L9 (DOT-native, no Huginn block): only T re-standardizes the state (RMSNorm on T's output), blocks stay pre-norm;
    # with / without the sqrt(d) embedding scale; plus a shared normalized T trained at random depth (depth extension)
    **{f'swst_d7_tmlp_ff{w}_unshared_tn_lr{t}_fa': dict(_U, sandwich=dict(_U['sandwich'], t_ff=w), dot_t_norm=True,
                                                         emb_sqrt_d=True, lr_scale=v)
       for w in (128, 512) for t, v in (('14', 1.4), ('20', 2.))},
    # equal params AND equal inference compute to Huginn r3: depth 8, narrow unshared T, T-output norm + sqrt(d)
    'swst_d8_tmlp_ff128_unshared_tn_lr14_fa': dict(_U, sandwich=dict(_U['sandwich'], depth=8, t_ff=128), dot_t_norm=True,
                                                   emb_sqrt_d=True, lr_scale=1.4),
    'swst_d7_tmlp_ff512_unshared_tn_lr14_fa_2x': dict(_U, dot_t_norm=True, emb_sqrt_d=True, lr_scale=1.4, steps=24576),
    'swst_d8_tmlp_ff128_unshared_tn_lr14_fa_2x': dict(_U, sandwich=dict(_U['sandwich'], depth=8, t_ff=128), dot_t_norm=True,
                                                      emb_sqrt_d=True, lr_scale=1.4, steps=24576),
    # L18: T-norm-only ablation (no T MLP) and dense8 + sqrt(d) (fair dense control)
    **{f'swst_d8_tnorm_only_lr{t}_fa': dict(_U, sandwich=dict(_U['sandwich'], depth=8, t_ff=128), dot_t_norm=True,
                                           t_norm_only=True, emb_sqrt_d=True, lr_scale=v)
       for t, v in (('14', 1.4), ('20', 2.0))},
    **{f'dense8_sq_lr{t}_fa': dict(SPECS['dense8'], emb_sqrt_d=True, lr_scale=v, fast_attn=True)
       for t, v in (('10', 1.0), ('14', 1.4))},
    **{f'dense8_sq_lr{t}_fa_2x': dict(SPECS['dense8'], emb_sqrt_d=True, lr_scale=v, fast_attn=True, steps=24576)
       for t, v in (('10', 1.0), ('14', 1.4))},
    **{f'dense4_sq_lr{t}_fa': dict(SPECS['dense4'], emb_sqrt_d=True, lr_scale=v, fast_attn=True)
       for t, v in (('10', 1.0), ('14', 1.4))},  # L20: equal-params dense control with sqrt(d)
    # L21: does the T-output norm lower the cost of a flexible module order (f18: +0.14 at the canonical order)?
    'swst_d7_tmlp_ff512_oe_lr14_fa': dict(_D7, order_eval=True),  # fixed order, old recipe, also scored on 3 random orders
    'swst_d7_tmlp_ff512_tn_oe_lr14_fa': dict(_D7, dot_t_norm=True, emb_sqrt_d=True, order_eval=True),
    'swst_d7_tmlp_ff512_rand_tn_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], random_order=True), dot_t_norm=True,
                                               emb_sqrt_d=True),
    'swst_d7_tmlp_ff512_rand_tn_noid_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], random_order=True, recipient=False),
                                                    dot_t_norm=True, emb_sqrt_d=True),
    # L22: translator capacity vs the cost of a flexible order: T as wide as a block MLP (t_ff 2048)
    'swst_d7_tmlp_ff2048_tn_oe_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], t_ff=2048), dot_t_norm=True,
                                              emb_sqrt_d=True, order_eval=True),
    'swst_d7_tmlp_ff2048_rand_tn_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], t_ff=2048, random_order=True),
                                                dot_t_norm=True, emb_sqrt_d=True),
    # L23: 4x tokens (49152 steps): does DOT's edge over the equal-compute dense8 + sqrt(d) survive longer training?
    'swst_d7_tmlp_ff512_unshared_tn_lr14_fa_4x': dict(_U, dot_t_norm=True, emb_sqrt_d=True, lr_scale=1.4, steps=49152),
    'swst_d8_tmlp_ff128_unshared_tn_lr14_fa_4x': dict(_U, sandwich=dict(_U['sandwich'], depth=8, t_ff=128), dot_t_norm=True,
                                                      emb_sqrt_d=True, lr_scale=1.4, steps=49152),
    'dense8_sq_lr10_fa_4x': dict(SPECS['dense8'], emb_sqrt_d=True, lr_scale=1.0, fast_attn=True, steps=49152),
    # L24: recurrent DOT with Huginn's depth sampler and truncation (A: T(s); B: T(s, e); C: norm-only control)
    **{f'dotrec_{name}_lr{t}': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                    lr_scale=v, dotrec=dict(mean_recurrence=3, backprop=3, **cfg))
       for name, cfg in (('t384', dict(t='mlp', t_ff=384)), ('inj_t224', dict(t='mlp', t_ff=224, inject=True)),
                         ('norm', dict(t='norm')), ('t384_fullbp', dict(t='mlp', t_ff=384, full_backprop=True)))
       for t, v in (('14', 1.4), ('20', 2.0))},
    # L25 diagnostic: B trained at a fixed r = 3 with full backprop (cost of the random-depth recipe for DOT-R)
    'dotrec_inj_t224_fixed_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                       lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=224,
                                                                 inject=True, fixed_r=True)),
    # L26: the same sampled depth sequence with full backprop for both (equal training compute again)
    'dotrec_inj_t224_fullbp_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                        lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=224,
                                                                  inject=True, full_backprop=True)),
    'huginn_r3_lr28_fullbp': dict(_FULL, lr_scale=2.8, full=dict(_HUGINN, mean_recurrence=3, backprop=3, ablate=('full_bp',))),
    # local control for the claim "random depth costs DOT more than Huginn": Huginn trained at a fixed r = 3
    'huginn_r3_lr28_fixedr': dict(_FULL, lr_scale=2.8, full=dict(_HUGINN, mean_recurrence=3, backprop=3, ablate=('fixed_r',))),
    # L27: the full DOT (3 cycling middles, T512 seeing the embedding, T-norm + sqrt(d)) at Huginn's sampled depth,
    # full backprop; r -> 2 r + 2 applications (r = 3 is DOT d8's 8)
    **{f'dotfull_rand_inj_t512_lr{t}': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                           lr_scale=v, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                   inject=True, full_backprop=True, layout='cycle3'))
       for t, v in (('10', 1.0), ('14', 1.4), ('20', 2.0))},
    # L28: wider translators in the same full DOT (user request): t_ff 1024 / 2048 (more params and compute than Huginn)
    **{f'dotfull_rand_inj_t{w}_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                          lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=w,
                                                                    inject=True, full_backprop=True, layout='cycle3'))
       for w in (1024, 2048)},
    # L29 diagnostics (analysis only): D1 the full DOT at a fixed depth (r = 3 -> 8 applications, full backprop);
    # D2 the full DOT at Huginn's random depth with Huginn's sandwich blocks
    'dotfull_fixed_inj_t512_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                        lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                  inject=True, layout='cycle3', fixed_r=True)),
    'dotfull_rand_inj_t512_sw_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                          lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                    inject=True, full_backprop=True, layout='cycle3',
                                                                    sandwich_blocks=True)),
    # L30 (user idea): one translator per 2-layer block (Huginn's layout with T for the adapter), random or fixed depth
    **{f'dotpair_{mode}_inj_t512_lr{t}': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                             lr_scale=v, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                     inject=True, layout='pair', **extra))
       for mode, extra in (('rand', dict(full_backprop=True)), ('fixed', dict(fixed_r=True)))
       for t, v in (('14', 1.4), ('20', 2.0))},
    # L31: GPT-2 style scaled init of the blocks' residual output projections (Huginn's blocks start near identity)
    'dotpair_rand_inj_t512_si_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                          lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                    inject=True, layout='pair', full_backprop=True,
                                                                    scaled_init=True)),
    'dotfull_rand_inj_t512_si_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                          lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                    inject=True, layout='cycle3', full_backprop=True,
                                                                    scaled_init=True)),
    # L32 (user idea): the original shared DOT translator (one T for all recipients + recipient code, as d7) at
    # Huginn's random depth with the embedding anchor, sized to Huginn's parameters: a wide one-hidden-layer T
    # (t_ff 750) vs a deeper two-hidden-layer T (576 x 2) at equal parameters
    **{f'dotshared_rand_{name}_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                          lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', inject=True,
                                                                    full_backprop=True, layout='cycle3', shared_t=True,
                                                                    **cfg))
       for name, cfg in (('t750', dict(t_ff=750)), ('t576x2', dict(t_ff=576, t_hidden=2)), ('t512', dict(t_ff=512)))},
    # L33 (T improvements on the best layout, pair + scaled init): idea 1 anchor gate, idea 3 T sees the last change
    **{f'dotpair_rand_inj_t512_si_{name}_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288,
                                                    fast_attn=True, lr_scale=1.4,
                                                    dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                inject=True, layout='pair', full_backprop=True,
                                                                scaled_init=True, **cfg))
       for name, cfg in (('gate', dict(anchor_gate=True)), ('delta', dict(delta=True)))},
    # L35 (user): the DOT ending vs Huginn's: (a) an unshared attention coda after T_out in place of the head FFN,
    # (c) no final T (the core pair's output goes straight to the DOT head FFN); both on DOT #1 (pair + scaled init)
    **{f'dotpair_rand_inj_t512_si_{name}_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288,
                                                    fast_attn=True, lr_scale=1.4,
                                                    dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                inject=True, layout='pair', full_backprop=True,
                                                                scaled_init=True, **cfg))
       for name, cfg in (('coda', dict(coda=True)), ('not', dict(final_t=False)))},
    # L34 (user): one shared d7-style T + scaled init, and the 4x-token B200 batch (train 4x directly; the calibration
    # curve gives the whole trajectory): Huginn, DOT #1 (pair + init), DOT #2 (full + init), one-T DOT (+ init)
    'dotshared_rand_t512_si_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                        lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                  inject=True, full_backprop=True, layout='cycle3',
                                                                  shared_t=True, scaled_init=True)),
    # L27 companion: the same full DOT with Huginn's truncated backprop (gradients through the last 3 recurrences)
    'dotfull_rand_inj_t512_trunc_lr14': dict(blocks=4, loop=None, mix='dotrec', var=False, steps=12288, fast_attn=True,
                                             lr_scale=1.4, dotrec=dict(mean_recurrence=3, backprop=3, t='mlp', t_ff=512,
                                                                       inject=True, layout='cycle3')),
    'swst_d8_tnorm_only_lr14_fa_2x': dict(_U, sandwich=dict(_U['sandwich'], depth=8, t_ff=128), dot_t_norm=True,
                                          t_norm_only=True, emb_sqrt_d=True, lr_scale=1.4, steps=24576),
    'swst_d7_tmlp_ff256_unshared_tn_lr14_fa': dict(_U, sandwich=dict(_U['sandwich'], t_ff=256), dot_t_norm=True,
                                                   emb_sqrt_d=True, lr_scale=1.4),
    'swst_d7_tmlp_ff128_unshared_tn_nosc_lr14_fa': dict(_U, sandwich=dict(_U['sandwich'], t_ff=128), dot_t_norm=True,
                                                        lr_scale=1.4),
    'swst_d7_tmlp_ff512_tn_rand_lr14_fa': dict(_D7, sandwich=dict(_D7['sandwich'], rand_depth=(4, 10)), dot_t_norm=True,
                                               emb_sqrt_d=True),
    # L7: twice the training tokens (24576 steps) for the best DOT + Huginn block arms and Huginn
    'swst_d7_tmlp_ff128_unshared_hb_lr20_fa_2x': dict(_U, sandwich=dict(_U['sandwich'], t_ff=128), huginn_block=True,
                                                      lr_scale=2., steps=24576),
    'swst_d7_tmlp_ff512_unshared_hb_lr14_fa_2x': dict(_U, huginn_block=True, lr_scale=1.4, steps=24576),
    # shared T (24.8M non-embedding params, fewer than Huginn's 26.4M) with the Huginn block
    **{f'swst_d7_tmlp_ff512_hb_lr{t}_fa': dict(_D7, huginn_block=True, lr_scale=v)
       for t, v in (('10', 1.), ('14', 1.4), ('20', 2.))},
})
# f22 (26 Nov): Huginn component analysis, one component removed at a time from huginn_r3 at lr x2.0 (s0 3.9917,
# s1 4.0024), plus the last lr points (huginn_r3 x2.8 s1, huginn_r2 x2.8 s0)
SPECS.update({
    'huginn_r2_lr28': dict(_FULL, lr_scale=2.8, full=dict(_HUGINN, mean_recurrence=2, backprop=2)),
    'huginn_r3_lr28_2x': dict(_FULL, lr_scale=2.8, steps=24576, full=dict(_HUGINN, mean_recurrence=3, backprop=3)),
    'huginn_r2_lr28_2x': dict(_FULL, lr_scale=2.8, steps=24576, full=dict(_HUGINN, mean_recurrence=2, backprop=2)),
    **{f'huginn_r3_lr20_{a}': dict(_FULL, lr_scale=2., full=dict(_HUGINN, mean_recurrence=3, backprop=3, ablate=(a,)))
       for a in ('prenorm', 'adamw', 'zero_state', 'fixed_r', 'add_inject', 'no_emb_scale')},
})
# Synthetic DOT benchmark arms (docs/DOT_SYNTH_20261124.md); steps are set by the synthetic study.
_SYN = dict(blocks=4, loop=None, mix='sandwich', var=False, steps=0)
_SYN_SW = dict(mode='stop', residual='plain', recipient=True, context_credit=True, roll_rows=64)
SPECS.update({
    'syn_fixed_d4': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, fixed_path=True)),
    'syn_fixed_d8': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=0., stop=False, fixed_path=True)),
    'syn_route_d8': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=0., stop=False)),
    'syn_stop_d8': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=.01)),
    'syn_stopfix_d8': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=.01, fixed_path=True)),
    'syn_budget_d8': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=.01, budget=True)),
    # s02: compute penalty only after half of training (the body first learns to use depth)
    'syn_stop_d8w': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=.01, lam_start=.5)),
    'syn_stopfix_d8w': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=.01, lam_start=.5, fixed_path=True)),
    'syn_budget_d8w': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=.01, lam_start=.5, budget=True)),
    # s05: equal trained depth 4 for routing vs fixed order; STOP at depth 4 (extended at inference)
    'syn_route_d4': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False)),
    'syn_stop_d4w': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=.01, lam_start=.5)),
    'syn_stopfix_d4w': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=.01, lam_start=.5, fixed_path=True)),
    # s06: routing with a fixed-order prior (+PRIOR_BETA logit on the fixed middle), no STOP
    'syn_prior_d4': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, prior=True)),
    'syn_prior_d8': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=0., stop=False, prior=True)),
    # s07: prior routing trained at a random depth in [3, 6] each step (router steps beyond 4 reuse step 4's router)
    'syn_prior_d4r': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, prior=True, rand_depth=(3, 6))),
    # s08: body on the greedy path, policy from separate sampled rollouts
    'syn_prior_d4g': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, prior=True, body_greedy=True)),
    'syn_route_d4g': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, body_greedy=True)),
    # s09: pure loop control (always middle 0) and the greedy-body prior router at depth 8 (opseq)
    'syn_loop1_d4': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, fixed_path=True, fixed_middle=0)),
    'syn_prior_d8g': dict(_SYN, sandwich=dict(_SYN_SW, depth=8, lam=0., stop=False, prior=True, body_greedy=True)),
    # s10: random training depth 2..8 at lr 1e-3 (s07 tried 3..6 at lr 5e-4, where nothing trained): deeper inference
    # should stay safe (a fixed point once the chain end is reached), so convergence halting becomes possible
    'syn_prior_d4gr': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, prior=True, body_greedy=True,
                                               rand_depth=(2, 8))),
    'syn_loop1_d4r': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, fixed_path=True, fixed_middle=0,
                                              rand_depth=(2, 8))),
    'syn_fixed_d4r': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, fixed_path=True, rand_depth=(2, 8))),
    # s11: the greedy-body router WITHOUT the fixed-order prior, random training depth
    'syn_route_d4gr': dict(_SYN, sandwich=dict(_SYN_SW, depth=4, lam=0., stop=False, body_greedy=True, rand_depth=(2, 8))),
})
DYN_RANK = 32
DYN_BOUND = .05  # dyn='tanh': per-element query delta bounded to +-0.05 (trained AttnRes queries ~0.004-0.02)
MOE_EXPERTS, MOE_TOP, MOE_FF_DIV, MOE_BALANCE = 8, 2, 2, .01
ARMS = ('dense4', 'dense4_2x', 'dense8', 'serial', 'memory', 'gated', 'gated_var', 'attnres')  # round 1
# L36 (user): the full DOT and the one-T DOT (both + scaled init) with M3 and its T replaced by an unshared coda
SPECS.update({a.replace('_si_', '_si_coda_'): dict(SPECS[a], dotrec=dict(SPECS[a]['dotrec'], cycle_coda=True))
              for a in ('dotfull_rand_inj_t512_si_lr14', 'dotshared_rand_t512_si_lr14')})

# L37: on DOT #1 + coda (L35a): no final T (coda reads the core pair's state directly), and the T-isolating ablation
# with a linear translator (Huginn's linear concat adapter, W [n(s), n(e)], in place of T's MLP)
SPECS['dotpair_rand_inj_t512_si_coda_not_lr14'] = dict(
    SPECS['dotpair_rand_inj_t512_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_t512_si_coda_lr14']['dotrec'], final_t=False))
SPECS['dotpair_rand_inj_lin_si_coda_lr14'] = dict(
    SPECS['dotpair_rand_inj_t512_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_t512_si_coda_lr14']['dotrec'], t='linear'))  # t_ff unused

# L39: Huginn under DOT's training recipe (AdamW, wd 0, our warmup-stable-decay schedule; f22 'adamw' ablation), the
# cleanest architecture-only control; x1.4 is DOT's LR, x2.0 the f22 point (-0.018 vs ELLISAdam there)
SPECS.update({f'huginn_r3_lr{n}_adamw': dict(SPECS['huginn_r3_lr28'], lr_scale=lr,
                                            full=dict(SPECS['huginn_r3_lr28']['full'], ablate=('adamw',)))
              for n, lr in (('14', 1.4),)})

# L40: the full DOT (three middles, four Ts, scaled init; best at 4x in L34) with an added attention coda after T_out in
# place of the head FFN; the cycle runs 2 r + 1 applications so attention applications stay Huginn's 2 r + 2
SPECS['dotfull_rand_inj_t512_si_coda3_lr14'] = dict(
    SPECS['dotfull_rand_inj_t512_si_lr14'], dotrec=dict(SPECS['dotfull_rand_inj_t512_si_lr14']['dotrec'], coda=True))

# L41 (user: other kinds of T), on DOT #1 + coda: the minimal translator norm(s + g * n(e)) (re-standardize and
# re-inject the anchor only), and the linear translator (best in L37) plus a causal depthwise token mix (kernel 4,
# zero init); the MLP-T + conv variant stays registered for reference
SPECS['dotpair_rand_inj_gate_si_coda_lr14'] = dict(
    SPECS['dotpair_rand_inj_t512_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_t512_si_coda_lr14']['dotrec'], t='gate'))
SPECS['dotpair_rand_inj_t512_si_coda_conv_lr14'] = dict(
    SPECS['dotpair_rand_inj_t512_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_t512_si_coda_lr14']['dotrec'], t_conv=4))
SPECS['dotpair_rand_inj_lin_si_coda_conv_lr14'] = dict(
    SPECS['dotpair_rand_inj_lin_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14']['dotrec'], t_conv=4))

# L42: the full DOT + added coda (L40 arm) at 1x locally, with the MLP T and with linear Ts (best T in the pair layout)
SPECS['dotfull_rand_inj_lin_si_coda3_lr14'] = dict(
    SPECS['dotfull_rand_inj_t512_si_coda3_lr14'],
    dotrec=dict(SPECS['dotfull_rand_inj_t512_si_coda3_lr14']['dotrec'], t='linear'))

# L43: LR check for the best arm (linear T + coda, tuned only at x1.4 so far; Huginn had x1.4 / x2.0 / x2.8)
SPECS['dotpair_rand_inj_lin_si_coda_lr20'] = dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14'], lr_scale=2.0)
SPECS['dotpair_rand_inj_lin_si_coda_lr28'] = dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14'], lr_scale=2.8)  # L43b

# L44: the best arm (linear T + coda) at 2x tokens locally (Huginn / dense8 + sqrt(d) 2x already exist locally)
SPECS['dotpair_rand_inj_lin_si_coda_lr14_2x'] = dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14'], steps=24576)
# L48: the 2x comparison at each side's better recipe: Huginn + AdamW x1.4 (best Huginn at 4x / d1024), DOT at x2.0
SPECS['huginn_r3_lr14_adamw_2x'] = dict(SPECS['huginn_r3_lr14_adamw'], steps=24576)
SPECS['dotpair_rand_inj_lin_si_coda_lr20_2x'] = dict(SPECS['dotpair_rand_inj_lin_si_coda_lr20'], steps=24576)

# L45 (B200, scale check): the main comparison at d = 1024 (16 heads, 4 KV heads, head 64, ff 2752 = 8/3 d rounded to
# 64), 4x tokens: linear T + coda, Huginn (original recipe and AdamW x1.4), dense8 + sqrt(d)
D1024 = dict(d=1024, heads=16, kv_heads=4, head_dim=64, ff=2752)
SPECS.update({f'{a}_d1024': dict(SPECS[a], config=D1024) for a in (
    'dotpair_rand_inj_lin_si_coda_lr14', 'huginn_r3_lr28', 'huginn_r3_lr14_adamw', 'dense8_sq_lr10_fa')})

# L46: linear T + coda with a translator between B1 and B2 as well (T at every core layer)
SPECS['dotpair_rand_inj_lin_si_coda_tmid_lr14'] = dict(
    SPECS['dotpair_rand_inj_lin_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14']['dotrec'], t_mid=True))

# L47 (B200, second scale step): the same comparison at d = 1536 (24 heads, 8 KV heads, head 64, ff 4096 = 8/3 d)
D1536 = dict(d=1536, heads=24, kv_heads=8, head_dim=64, ff=4096)
SPECS.update({f'{a}_d1536': dict(SPECS[a], config=D1536) for a in (
    'dotpair_rand_inj_lin_si_coda_lr14', 'huginn_r3_lr28', 'huginn_r3_lr14_adamw', 'dense8_sq_lr10_fa')})

# L49 (analysis: why DOT beats Huginn): Huginn with DOT's readout translator before its coda (linear T on [n(s), n(e)],
# residual, output norm) in place of ln_f; original recipe and AdamW x1.4
SPECS.update({f'{a}_codat': dict(SPECS[a], full=dict(SPECS[a]['full'], coda_t=True))
              for a in ('huginn_r3_lr28', 'huginn_r3_lr14_adamw')})

# L50 (B200, LR check at d1536 before reading the size trend): DOT x1.0 / x2.0 and Huginn's original recipe x2.0 / x4.0
SPECS['dotpair_rand_inj_lin_si_coda_lr10'] = dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14'], lr_scale=1.0)
SPECS.update({f'{a}_d1536': dict(SPECS[a], config=D1536) for a in (
    'dotpair_rand_inj_lin_si_coda_lr10', 'dotpair_rand_inj_lin_si_coda_lr20', 'huginn_r3_lr20', 'huginn_r3_lr40')})

# L51 (analysis): DOT #1 + coda with Huginn's adapter form W [s, e] (no norms, no residual) in place of the linear T
SPECS['dotpair_rand_inj_adapter_si_coda_lr14'] = dict(
    SPECS['dotpair_rand_inj_lin_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14']['dotrec'], t='adapter'))

# L52 (analysis): DOT #1 + coda with linear T, starting the recurrence from s0 = 0 instead of s0 = e
SPECS['dotpair_rand_inj_lin_si_coda_s0zero_lr14'] = dict(
    SPECS['dotpair_rand_inj_lin_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14']['dotrec'], s0='zero'))

# L53 (analysis): DOT #1 + coda with linear T and Huginn's sandwich blocks (instead of pre-norm)
SPECS['dotpair_rand_inj_lin_si_coda_sw_lr14'] = dict(
    SPECS['dotpair_rand_inj_lin_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14']['dotrec'], sandwich_blocks=True))

# L54 (training-compute efficiency): DOT #1 + coda with linear T trained at Huginn's sampler with mean 2 (6 attention
# applications at the mean) against Huginn at mean 3 (8)
SPECS['dotpair_rand_inj_lin_si_coda_r2_lr14'] = dict(
    SPECS['dotpair_rand_inj_lin_si_coda_lr14'],
    dotrec=dict(SPECS['dotpair_rand_inj_lin_si_coda_lr14']['dotrec'], mean_recurrence=2, backprop=2))

# L55 (B200): LR sweep at d1024 and at d768 4x so every size in the summary compares each side at its best LR
SPECS.update({f'{a}_d1024': dict(SPECS[a], config=D1024) for a in (
    'dotpair_rand_inj_lin_si_coda_lr10', 'dotpair_rand_inj_lin_si_coda_lr20', 'huginn_r3_lr20', 'huginn_r3_lr40')})

# L34: the 4x-token versions of the B200 batch (built from the registered 1x specs; only the step count changes)
SPECS.update({f'{a}_4x': dict(SPECS[a], steps=49152) for a in (
    'huginn_r3_lr28', 'dotpair_rand_inj_t512_si_lr14', 'dotfull_rand_inj_t512_si_lr14', 'dotshared_rand_t512_si_lr14',
    # L38 (second B200 batch): the coda arms of L35-L37
    'dotpair_rand_inj_t512_si_coda_lr14', 'dotfull_rand_inj_t512_si_coda_lr14', 'dotshared_rand_t512_si_coda_lr14',
    'dotpair_rand_inj_t512_si_coda_not_lr14', 'dotpair_rand_inj_lin_si_coda_lr14',
    # L40 (third B200 batch)
    'dotfull_rand_inj_t512_si_coda3_lr14', 'huginn_r3_lr20_adamw', 'huginn_r3_lr14_adamw',
    # L45
    'dotpair_rand_inj_lin_si_coda_lr14_d1024', 'huginn_r3_lr28_d1024', 'huginn_r3_lr14_adamw_d1024',
    'dense8_sq_lr10_fa_d1024',
    # L47
    'dotpair_rand_inj_lin_si_coda_lr14_d1536', 'huginn_r3_lr28_d1536', 'huginn_r3_lr14_adamw_d1536',
    'dense8_sq_lr10_fa_d1536',
    # L50
    'dotpair_rand_inj_lin_si_coda_lr10_d1536', 'dotpair_rand_inj_lin_si_coda_lr20_d1536', 'huginn_r3_lr20_d1536',
    'huginn_r3_lr40_d1536',
    # L55
    'dotpair_rand_inj_lin_si_coda_lr10_d1024', 'dotpair_rand_inj_lin_si_coda_lr20_d1024', 'huginn_r3_lr20_d1024',
    'huginn_r3_lr40_d1024', 'dotpair_rand_inj_lin_si_coda_lr20', 'huginn_r3_lr20', 'huginn_r3_lr40')})
ALL_ARMS = tuple(SPECS)


def config_for(arm):
    """The model config of an arm: the default Config unless the spec overrides fields (L45 'config')."""
    from .model import Config
    return Config(**SPECS[arm].get('config', {}))

LOOP_ARMS = tuple(a for a, s in SPECS.items() if s['loop'])
STEPS = {a: s['steps'] for a, s in SPECS.items()}
VAR_PASSES = (3, 1, 3, 2, 3, 4, 3, 5)  # P(3)=1/2, P(1|2|4|5)=1/8 each, mean 3
NATIVE4 = (0, 1, 2, 3)
SWAP4 = (0, 2, 1, 3)
NATIVE8 = tuple(range(8))


def lr_at(step, total, peak=6e-4, warmup=128, cooldown=0.2):
    """Warmup-stable-decay: linear warm-up, constant, linear decay to 0 over the last 20%."""
    if type(step) is not int or not 0 <= step < total:
        raise ValueError('step outside schedule')
    start = int(total * (1 - cooldown))
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < start:
        return peak
    return peak * (total - step) / (total - start)


def train_passes(arm, step):
    """Sequence-level pass count for a step (token-level arms run up to max(VAR_PASSES))."""
    spec = SPECS[arm]
    if spec['var'] == 'token':
        return max(VAR_PASSES)
    if spec['var']:
        return VAR_PASSES[step % len(VAR_PASSES)]
    return spec.get('passes', 3) if spec['loop'] else None


def token_pass_targets(shape, step, seed, device):
    """Per-token pass counts drawn from the VAR_PASSES distribution, deterministic in (seed, step)."""
    g = torch.Generator().manual_seed(1_000_003 * (seed + 1) + step)
    pick = torch.randint(0, len(VAR_PASSES), shape, generator=g)
    return torch.tensor(VAR_PASSES)[pick].to(device)


def block_applications(arm, passes=3):
    spec = SPECS[arm]
    if spec.get('full'):  # nominal layer applications at inference (MoR: expected under its capacities)
        f = spec['full']
        if f['kind'] == 'huginn':
            return f['prelude'] + f['core'] * f['mean_recurrence'] + f['coda']
        if f['kind'] == 'ouro':
            return f['layers'] * f['steps']
        caps = f.get('capacity') or [(f['recursions'] - i) / f['recursions'] for i in range(f['recursions'])]
        return 2 + f['base_depth'] * sum(caps)
    if spec.get('dotrec'):  # attention-block applications at the native recurrence: prelude + 2 r (cycle3: 2 r + 2)
        r = spec['dotrec']['mean_recurrence']
        return 2 * r + 2 if spec['dotrec'].get('layout') == 'cycle3' else 1 + 2 * r + bool(spec['dotrec'].get('coda'))
    if spec.get('schedule'):
        return 4 if spec['schedule'] == 'dense' else spec.get('apps') or 2 * spec['depth'] + 1
    if spec.get('dot'):
        return spec['dot']['steps']
    if spec.get('sandwich'):
        return 2 * spec['sandwich']['depth'] + 1
    return 2 + 2 * spec.get('passes', passes) if spec['loop'] else spec['blocks']


def make_model(config, arm):
    """Model for any registered arm: LoopScreen, token-level DOT or DOT sandwich."""
    spec = SPECS[arm]
    if spec.get('full'):  # f19: full published looped methods
        from .full_methods import make_full_model
        return make_full_model(config, arm, spec)
    if spec.get('dotrec'):  # L24: recurrent DOT (fair recurrent-depth counterpart of Huginn); L27: the full cycling DOT
        from .dot_recurrent import DotCycle, DotPair, DotRecurrent
        kind = dict(cycle3=DotCycle, pair=DotPair).get(spec['dotrec'].get('layout'), DotRecurrent)
        return kind(config, arm, spec)
    if spec.get('schedule'):
        if spec.get('shared'):
            from .shared_heads import SharedHeadSchedule
            return SharedHeadSchedule(config, arm, spec)
        from .loop_baselines import BlockSchedule
        return BlockSchedule(config, arm, spec)
    if spec.get('dot'):
        from .dot_token import DotToken
        return DotToken(config, arm, spec)
    if spec.get('sandwich'):
        if spec['sandwich']['mode'] == 'stop':
            from .dot_stop_sandwich import DotSandwichStop
            return DotSandwichStop(config, arm, spec)
        from .dot_sandwich import DotSandwich
        return DotSandwich(config, arm, spec)
    return LoopScreen(config, arm)


def native_path(arm):
    return NATIVE8 if SPECS[arm]['blocks'] == 8 else NATIVE4


class LoopScreen(EmbeddingAnchor):
    def __init__(self, config, arm):
        super().__init__(config)  # identical base init (embed, B0..B3, head) for every arm
        if arm not in SPECS:
            raise ValueError('Unknown screen arm')
        self.arm = arm
        self.spec = SPECS[arm]
        self.passes = 3
        self.memory_enabled = True
        self.token_passes = None  # per-token pass targets (token-var arms, training only)
        d = config.d
        if self.spec['blocks'] == 8:
            bc = BlockConfig(d=d, heads=config.heads, kv_heads=config.kv_heads,
                             head_dim=config.head_dim, ff=config.ff)
            extra = [Block(bc, True) for _ in range(4)]
            for block in extra:
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=.02)
            self.blocks.extend(extra)
        self.huginn_block = bool(self.spec.get('huginn_block')) and self.spec['mix'] == 'residual'
        if self.huginn_block:  # f23: Huginn's sandwich norms in every block (+ sqrt(d) embedding scale); DOT arms
            from efficient_llm.dot_pool.model import enable_sandwich  # (mix 'sandwich') apply it in their own class
            if self.spec['loop']:
                raise ValueError('huginn_block is defined for the plain residual stack')
            for block in self.blocks:
                enable_sandwich(block, d)
        if self.spec.get('ff'):  # f16 control: a wider FFN in every block (fresh init), to match a compute budget
            from efficient_llm.dot_pool.model import MLP as _MLP
            for block in self.blocks:
                block.mlp = _MLP(d, self.spec['ff'])
                for module in block.mlp.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=.02)
        if self.spec['loop'] == 'memory':
            self.memory_query = nn.Parameter(torch.zeros(2, d))
            self.memory_gain = nn.Parameter(torch.zeros(2, d))
        if self.spec['loop'] == 'gated':
            self.message_gate = nn.Parameter(torch.zeros(d))  # closed at init: every pass = one pass
        if self.spec['mix'] == 'attnres':
            # one pseudo-query per stored (block, sublayer), shared across loop passes, plus readout
            self.attnres_query = nn.Parameter(torch.zeros(2 * self.spec['blocks'] + 1, d))
        if self.spec.get('attnres_t'):  # f18: DOT's narrow MLP-only T as an extra AttnRes sublayer at every step
            from efficient_llm.dot_pool.model import MLP, RMSNorm
            if self.spec['mix'] != 'attnres':
                raise ValueError('attnres_t is an option of the AttnRes arms')
            n_t = len(self.application_order(NATIVE8 if self.spec['blocks'] == 8 else NATIVE4)) + 1
            self.t_norm, self.t_mlp = RMSNorm(d), MLP(d, self.spec['attnres_t'])
            for module in self.t_mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
            self.t_query = nn.Parameter(torch.zeros(n_t, d))  # one AttnRes query per T position (later reuse last)
        if self.spec.get('dyn'):
            # q_l(t) = w_l + B_l A_l RMSNorm(latest source at t); B zero-init, so training starts as AttnRes
            n = 2 * self.spec['blocks'] + 1
            self.attnres_dyn_a = nn.Parameter(torch.randn(n, d, DYN_RANK) / math.sqrt(d))
            self.attnres_dyn_b = nn.Parameter(torch.zeros(n, DYN_RANK, d))
        self.aux = None
        self.moe_trace = None  # set to {} to record top-k expert indices per (block, application)
        if self.spec.get('moe'):
            from efficient_llm.dot_pool.model import MLP
            k, ff = self.moe_k, config.ff // self.spec.get('moe_ffdiv', MOE_FF_DIV)
            self.moe_experts = nn.ModuleDict({str(i): nn.ModuleList(MLP(d, ff) for _ in range(k)) for i in self.spec['moe']})
            for module in self.moe_experts.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
            # small random router init: no exact top-k ties (CPU/GPU tie-breaking could differ)
            self.moe_router = nn.ParameterDict({str(i): nn.Parameter(torch.randn(k, d) * .02) for i in self.spec['moe']})
            for i in self.spec['moe']:
                self.register_buffer(f'moe_usage_{i}', torch.full((k,), 1. / k))
            # extra parameters are created after the base MoE ones, so base MoE arms keep their init
            for i in self.spec['moe']:
                for p in range(1, self.spec.get('moe_pass', 1)):
                    self.moe_router[f'{i}_p{p}'] = nn.Parameter(torch.randn(k, d) * .02)
            if self.spec.get('moe_shared'):
                self.moe_shared = nn.ModuleDict({str(i): MLP(d, config.ff // 2) for i in self.spec['moe']})
                for module in self.moe_shared.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=.02)

    @property
    def moe_k(self):
        return self.spec.get('moe_k', MOE_EXPERTS)

    @property
    def moe_top(self):
        return self.spec.get('moe_top', MOE_TOP)

    # ---- shared helpers -------------------------------------------------
    def _guard(self, x):
        from .embedding_anchor_runtime import anchor_device
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        if x.ndim != 2 or not 1 <= x.shape[1] <= self.c.length:
            raise ValueError('Invalid token matrix')

    def _check_path(self, path):
        path = tuple(path)
        expected = set(range(self.spec['blocks']))
        if set(path) != expected or len(path) != len(expected):
            raise ValueError('Path must be a permutation of all stored blocks')
        if self.spec['loop'] and (path[0], path[3]) != (0, 3):
            raise ValueError('Loop arms keep prelude 0 and coda 3 fixed')
        if self.spec['loop'] and not 1 <= self.passes <= 16:
            raise ValueError('Pass count outside supported range')
        return path

    def read_memory(self, h, sources, identity):
        # Identical to anchor_loop.LoopAnchor.read_memory (completed long-loop study).
        values = torch.stack([F.rms_norm(v.float(), (self.c.d,), eps=1e-6) for v in sources], dim=-2)
        q = self.memory_query[identity - 1]
        weights = torch.softmax((values * q).sum(dim=-1) / math.sqrt(self.c.d), dim=-1)
        message = (values * weights.unsqueeze(-1)).sum(dim=-2)
        scale = (h.float().square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        return h + scale * self.memory_gain[identity - 1].tanh() * message

    def _attnres_mix(self, sources, index, query=None):
        """Kimi AttnRes: softmax over depth of w^T RMSNorm(v_i), w zero-init -> uniform mean."""
        values = torch.stack(sources, dim=0)
        keys = F.rms_norm(values.float(), (self.c.d,), eps=1e-6)
        if query is not None:  # an explicit query (the f18 T sublayer), no dynamic part
            weights = torch.softmax((keys * query.float()).sum(-1), dim=0).to(values.dtype)
            return (weights.unsqueeze(-1) * values).sum(0)
        query = self.attnres_query[index].float()
        if self.spec.get('dyn'):
            state = keys[-1]  # RMSNorm of the latest source at each token
            delta = (state @ self.attnres_dyn_a[index].float()) @ self.attnres_dyn_b[index].float()
            query = query + (DYN_BOUND * torch.tanh(delta) if self.spec['dyn'] == 'tanh' else delta)
        logits = (keys * query).sum(-1)
        weights = torch.softmax(logits, dim=0).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(0)

    def _mlp(self, identity, u, application=0):
        if not self.spec.get('moe') or identity not in self.spec['moe']:
            return self.blocks[identity].mlp(u)
        experts, k, top = self.moe_experts[str(identity)], self.moe_k, self.moe_top
        key = str(identity)
        if self.spec.get('moe_pass') and application:  # per-pass router; extra eval passes reuse the last one
            key = f"{identity}_p{min(application, self.spec['moe_pass'] - 1)}"
        probs = torch.softmax(F.rms_norm(u.float(), (self.c.d,), eps=1e-6) @ self.moe_router[key].float().t(), -1)
        value, index = probs.topk(top, -1)
        value = value / value.sum(-1, keepdim=True)  # Mixtral: renormalized over the chosen experts
        weights = sum(F.one_hot(index[..., j], k).to(probs.dtype) * value[..., j:j + 1] for j in range(top))
        out = sum(weights[..., i:i + 1].to(u.dtype) * experts[i](u) for i in range(k))  # exact top-k
        if self.spec.get('moe_shared'):
            out = out + self.moe_shared[str(identity)](u)
        if self.moe_trace is not None:
            self.moe_trace.setdefault(identity, []).append(index.detach())
        if self.training and torch.is_grad_enabled():
            frac = sum(F.one_hot(index[..., j], k).float() for j in range(top)).reshape(-1, k).mean(0) / top
            with torch.no_grad():
                usage = getattr(self, f'moe_usage_{identity}')
                usage.mul_(.95).add_(.05 * frac)
            balance = MOE_BALANCE * k * (frac.detach() * probs.reshape(-1, k).mean(0)).sum()
            self.aux = balance if self.aux is None else self.aux + balance
        return out

    def _block(self, identity, h, pos, seg):
        if not self.spec.get('moe') or identity not in self.spec['moe']:
            return self.blocks[identity](h, pos, seg)
        block = self.blocks[identity]
        h = block.attention(h, pos, seg)
        return h + self._mlp(identity, block.post_attention_layernorm(h))

    def application_order(self, path):
        """Stored block identities in execution order (loops unrolled)."""
        if not self.spec['loop']:
            return tuple(path)
        return (path[0], *(path[1:3] * self.passes), path[3])

    # ---- forward ----------------------------------------------------------
    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('Screen arms take no anchor, donor or skip options')
        if self.token_passes is not None:
            from .loop_stop import halted_hidden
            targets = self.token_passes
            if targets.shape != x.shape:
                raise ValueError('token_passes must match the token matrix')
            return halted_hidden(self, x, path, int(targets.max()), lambda k, s: targets <= k)[0]
        self._guard(x)
        path = self._check_path(path)
        self.aux = None
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = None  # one sequence per row (no packed documents in this input)
        e = self.embed(x)
        if self.huginn_block or (self.spec.get('emb_sqrt_d') and self.spec['mix'] == 'residual' and not self.spec['loop']):
            e = e * math.sqrt(self.c.d)  # L18: dense8 + sqrt(d) alone, the fair dense control for the DOT-native arms
        if self.spec['mix'] == 'attnres':
            return self._attnres_hidden(e, self.application_order(path), pos, seg)
        if not self.spec['loop']:
            h = e
            for identity in path:
                h = self._block(identity, h, pos, seg)
            return h
        prelude = self.blocks[path[0]](e, pos, seg)
        core = path[1:3]
        if self.spec['loop'] == 'gated':
            message = None
            for _ in range(self.passes):
                h = prelude if message is None else prelude + self.message_gate * (message - prelude)
                for identity in core:
                    h = self.blocks[identity](h, pos, seg)
                message = h
            return self.blocks[path[3]](message, pos, seg)
        h = prelude
        previous = ()
        for _ in range(self.passes):
            current = []
            for identity in core:
                if self.spec['loop'] == 'memory' and self.memory_enabled:
                    h = self.read_memory(h, (e, prelude, *previous), identity)
                h = self.blocks[identity](h, pos, seg)
                current.append(h)
            previous = tuple(current)
        return self.blocks[path[3]](h, pos, seg)

    def _attnres_hidden(self, e, order, pos, seg):
        sources = [e]
        seen = {}
        t_on = self.spec.get('attnres_t')

        def t_step(i):  # DOT T: x = mix(sources), then the narrow MLP's output becomes a new source
            x = self._attnres_mix(sources, None, self.t_query[min(i, self.t_query.shape[0] - 1)])
            sources.append(self.t_mlp(self.t_norm(x)))

        for i, identity in enumerate(order):
            if t_on:
                t_step(i)
            block = self.blocks[identity]
            x = self._attnres_mix(sources, 2 * identity)
            sources.append(block.self_attn(block.input_layernorm(x), pos, seg))
            x = self._attnres_mix(sources, 2 * identity + 1)
            application = seen.get(identity, 0)
            seen[identity] = application + 1
            sources.append(self._mlp(identity, block.post_attention_layernorm(x), application))
        if t_on:
            t_step(len(order))
        return self._attnres_mix(sources, 2 * self.spec['blocks'])

