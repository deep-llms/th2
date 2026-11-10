# A2 Family v1.3

Spec SHA256: 40089a98702a285d38799adcdfa89c72c7eec2e75e627cfef8552448f9a593c0

Final readout: True; coverage: True.

- FE1: {'value': 0.00971901488669057, 'label': 'persists', 'budget': '3.2B', 'not_converged': True, 'unverified': True}
- FE2: {'label': 'changing', 'direction': 'down', 'difference': -0.006535701030237153, 'unverified': True}
- FE3: {'value': 0.007312437315304306, 'label': 'grows', 'budget': '3.2B', 'not_converged': False, 'unverified': True}
- data_order_noise: {'value': 0.0004369966630324712, 'flag': False, 'unverified': True}
- FE4: {'FE4': 'robust', 'difference': 0.0014928528840014443, 'direction': 'up', 'data_order_difference': 0.0004369966630324712, 'run_F5': False, 'unverified': True}

| Run | Parent NLL | NLL | Delta pre | Delta post | Exposure | Flags |
|---|---:|---:|---:|---:|---|---:|
| F2-fwe-k20-BS1-0.2B-lr0.0001-s2030 | 2.678332086 | 2.936645220 | 1.127589809 | 0.258313134 | not seen | 72 |
| F2-fwe-k20-BS1-0.2B-lr0.0001-s2031 | 2.678332086 | 2.935072186 | 1.127589809 | 0.256740100 | not seen | 72 |
| F2-fwe-k20-BS1-0.2B-lr2.5e-05-s2030 | 2.678332086 | 3.022093862 | 1.127589809 | 0.343761775 | not seen | 72 |
| F2-fwe-k20-BS1-0.2B-lr5e-05-s2030 | 2.678332086 | 2.975353894 | 1.127589809 | 0.297021808 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr0.0001-s2030 | 2.723293590 | 2.961746300 | 1.026185746 | 0.238452710 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr0.0001-s2031 | 2.723293590 | 2.959736269 | 1.026185746 | 0.236442679 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr2.5e-05-s2030 | 2.723293590 | 3.039208359 | 1.026185746 | 0.315914768 | not seen | 72 |
| F2-fwe-k6-BS1-0.2B-lr5e-05-s2030 | 2.723293590 | 2.996886943 | 1.026185746 | 0.273593353 | not seen | 72 |
| F2c-mix-k20-BS1-0.2B-lr0.0001-s2030 | 2.678332086 | 2.942843111 | 1.127589809 | 0.264511025 | not seen | 88 |
| F2c-mix-k6-BS1-0.2B-lr0.0001-s2030 | 2.723293590 | 2.966451338 | 1.026185746 | 0.243157747 | not seen | 88 |
| F3-fwe-k20-BS1-0.8B-lr0.0001-s2030 | 2.678332086 | 2.855236285 | 1.127589809 | 0.176904198 | not seen | 72 |
| F3-fwe-k20-BS1-3.2B-lr0.0001-s2030 | 2.678332086 | 2.801630386 | 1.127589809 | 0.123298300 | seen | 72 |
| F3-fwe-k6-BS1-0.8B-lr0.0001-s2030 | 2.723293590 | 2.886873065 | 1.026185746 | 0.163579475 | not seen | 72 |
| F3-fwe-k6-BS1-3.2B-lr0.0001-s2030 | 2.723293590 | 2.836872875 | 1.026185746 | 0.113579285 | seen | 72 |
| F4-fwe-k64-BS1-0.2B-lr0.0001-s2030 | 2.650000282 | 2.922642446 | 2.040374105 | 0.272642165 | not seen | 72 |
| F4-fwe-k64-BS1-0.8B-lr0.0001-s2030 | 2.650000282 | 2.837047271 | 2.040374105 | 0.187046990 | not seen | 72 |
| F4-fwe-k64-BS1-3.2B-lr0.0001-s2030 | 2.650000282 | 2.780611019 | 2.040374105 | 0.130610737 | seen | 72 |

| Contrast / data / budget / LR / order | Full | Reduced | Reduced minus full | Primary | 95% CI | E_pre | G | E/G | E/G 95% CI | Dropped docs / targets |
|---|---:|---:|---:|---|---|---:|---:|---:|---|---|
| E / fwe / 0.2B / 0.0001 / 2030 | 0.019860424 | 0.019863951 | +0.000003527 | full | [0.01911916670735676, 0.020606882993532052] | 0.101404063 | 0.044961504 | 0.44172064250308585 | [0.4288592363137444, 0.45492415253944585] | 3 / 1140 |
| E64 / fwe / 0.2B / 0.0001 / 2030 | 0.034189455 | 0.034189153 | -0.000000302 | full | [0.03328134518666394, 0.03519982542594845] | 1.014188359 | 0.073293309 | 0.4664744391201238 | [0.456911690728306, 0.4765215361054859] | 3 / 1140 |
| D64 / fwe / 0.2B / 0.0001 / 2030 | 0.014329031 | 0.014325201 | -0.000003829 | full | [0.013618781326432211, 0.015077983702223898] | 0.912784296 | 0.028331805 | 0.505757779707981 | [0.485923114448883, 0.5259532285139887] | 3 / 1140 |
| E / fwe / 0.2B / 0.0001 / 2031 | 0.020297421 | 0.020299540 | +0.000002119 | full | [0.019544622656445042, 0.02107822744457871] | 0.101404063 | 0.044961504 | 0.45143999403347645 | [0.4381819953742655, 0.4649124491244255] | 3 / 1140 |
| E / fwe / 0.2B / 2.5e-05 / 2030 | 0.027847007 | 0.027851073 | +0.000004066 | full | [0.027060587701655585, 0.028637616232778943] | 0.101404063 | 0.044961504 | 0.6193522089503595 | [0.605913142800819, 0.632707928341694] | 3 / 1140 |
| E / fwe / 0.2B / 5e-05 / 2030 | 0.023428455 | 0.023432508 | +0.000004053 | full | [0.022665189591775006, 0.02418887673043295] | 0.101404063 | 0.044961504 | 0.5210780960320273 | [0.5080179338980361, 0.5340904212884937] | 3 / 1140 |
| E / mix / 0.2B / 0.0001 / 2030 | 0.021353277 | 0.021356850 | +0.000003573 | full | [0.0206018207902273, 0.022108638345715054] | 0.101404063 | 0.044961504 | 0.4749235549969264 | [0.46206429704812074, 0.48769120473265326] | 3 / 1140 |
| E / fwe / 0.8B / 0.0001 / 2030 | 0.013324723 | 0.013327384 | +0.000002661 | full | [0.01259927161685909, 0.014057804107447808] | 0.101404063 | 0.044961504 | 0.29635848946144827 | [0.2828428722270998, 0.3101328950111438] | 3 / 1140 |
| E64 / fwe / 0.8B / 0.0001 / 2030 | 0.023467515 | 0.023467399 | -0.000000116 | full | [0.02261695158859439, 0.02442956901905133] | 1.014188359 | 0.073293309 | 0.32018631974025186 | [0.31069630253905317, 0.33040604982838917] | 3 / 1140 |
| D64 / fwe / 0.8B / 0.0001 / 2030 | 0.010142791 | 0.010140015 | -0.000002777 | full | [0.009461373976680829, 0.010853471373500878] | 0.912784296 | 0.028331805 | 0.35800018671157713 | [0.3386663513674781, 0.3778172507015205] | 3 / 1140 |
| E / fwe / 3.2B / 0.0001 / 2030 | 0.009719015 | 0.009721389 | +0.000002374 | full | [0.009022865073839276, 0.010465821393833934] | 0.101404063 | 0.044961504 | 0.21616302869989215 | [0.2025555849900522, 0.23057547773787312] | 3 / 1140 |
| E64 / fwe / 3.2B / 0.0001 / 2030 | 0.017031452 | 0.017031637 | +0.000000184 | full | [0.016261573587966215, 0.017885384609443087] | 1.014188359 | 0.073293309 | 0.23237390340546138 | [0.22333215481620913, 0.24197638068053337] | 3 / 1140 |
| D64 / fwe / 3.2B / 0.0001 / 2030 | 0.007312437 | 0.007310248 | -0.000002189 | full | [0.006691771781837816, 0.00794790779711956] | 0.912784296 | 0.028331805 | 0.25809994947985004 | [0.23969918789285458, 0.2769041385854895] | 3 / 1140 |

Matched LR (peak/4): 5e-05. OpenSci stable E/G: DCLM 0.51, FWE 0.81; annealed: 0.41, 0.50.

F0 dclm: {'status': 'available', 'nll': 2.7736634966730858, 'parent_P300': 2.7377930322126844, 'reference_minus_parent': 0.03587046446040132, 'break_even': {'budget': None, 'largest_budget': 12.8, 'gap': 0.04299901353590707}, 'nonembedding_parameters': 1205424128, 'reference_pretraining_tokens': 300000000000.0, 'flags': [], 'cross_node_12_8B_note': 'First-node final scalar; cross-node difference about 1e-4'}

| Student | Heal budget | NLL | Student minus 1.3B | Total training tokens |
|---|---|---:|---:|---:|
| P300 | 0.2B | 2.942554079 | +0.168890582 | 300200000000 |
| P1T | 0.2B | 2.912766332 | +0.139102835 | 1000200000000 |
| P300 | 0.8B | 2.879720978 | +0.106057482 | 300800000000 |
| P1T | 0.8B | 2.849623360 | +0.075959864 | 1000800000000 |
| P300 | 3.2B | 2.839891311 | +0.066227815 | 303200000000 |
| P1T | 3.2B | 2.810678156 | +0.037014659 | 1003200000000 |
| P300 | 12.8B | 2.816662510 | +0.042999014 | 312800000000 |
| P1T | 12.8B | 2.786168130 | +0.012504633 | 1012800000000 |

F0 fwe: {'status': 'available', 'nll': 2.893222287314092, 'parent_P300': 2.860631325856815, 'reference_minus_parent': 0.0325909614572768, 'break_even': {'budget': None, 'largest_budget': 3.2, 'gap': 0.06259006519599719}, 'nonembedding_parameters': 1205424128, 'reference_pretraining_tokens': 300000000000.0, 'flags': [], 'cross_node_12_8B_note': 'First-node final scalar; cross-node difference about 1e-4'}

| Student | Heal budget | NLL | Student minus 1.3B | Total training tokens |
|---|---|---:|---:|---:|
| P300 | 0.2B | 3.056353314 | +0.163131027 | 300200000000 |
| P1T | 0.2B | 3.030653544 | +0.137431256 | 1000200000000 |
| P300 | 0.8B | 2.997096324 | +0.103874037 | 300800000000 |
| P1T | 0.8B | 2.967992111 | +0.074769824 | 1000800000000 |
| P300 | 3.2B | 2.955812353 | +0.062590065 | 303200000000 |
| P1T | 3.2B | 2.927349511 | +0.034127223 | 1003200000000 |

Conditional extensions and trigger evidence: {'F3': True, 'F4': True, 'F5': False, 'FE4': {'FE4': 'robust', 'difference': 0.0014928528840014443, 'direction': 'up', 'data_order_difference': 0.0004369966630324712, 'run_F5': False}, 'trigger_records': [{'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.01986042444939224, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 12}, {'budget': '0.8B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.013324723419155087, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 12}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k20', 'k64'], 'value': 0.014329030549193596, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 16}, {'budget': '0.8B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k20', 'k64'], 'value': 0.0101427913080627, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 16}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.01986042444939224, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 17}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2030, 'data': 'mix', 'arms': ['k6', 'k20'], 'value': 0.021353277333393685, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 17}, {'budget': '0.2B', 'lr': 0.0001, 'seed': 2031, 'data': 'fwe', 'arms': ['k6', 'k20'], 'value': 0.02029742111242471, 'overlap_basis': {'immaterial': True, 'primary': 'full'}, 'completed_runs': 17}], 'trigger_policy': 'Primary overlap-adjudicated quantity using evidence available at each registered trigger; records immutable after later runs', 'missing_trigger_policy': 'report unverified and continue independent registered stages; no invented measurement', 'spec_sha256': '40089a98702a285d38799adcdfa89c72c7eec2e75e627cfef8552448f9a593c0'}

Paired document bootstrap: 10,000 resamples, seed 0, token-weighted NLL. Evaluation noise only.
Mixture sampling/source qualifications apply to mixture rows. Shared stable-parent caveat: healing also anneals the student.
Input qualifications: [{'test': 'mixture_source_qualification', 'data': 'mix', 'detail': {'test': 'full_source_uniform_document_sample', 'passed': False, 'detail': 'Uniform documents inside declared shard pools; new corpora use seeded uniform shard selection rather than a global iid document sample. Record finite-pool/cluster-sampling limitation on mixture results; primary FWE unaffected.'}}, {'test': 'FT8_shares_0.2B', 'data': 'mix', 'detail': {'test': 'FT8_shares_0.2B', 'passed': False}}]

Missing: []; invalid: [].
Remaining: verify backup/native handoff and resume Depth.


# Family v1.4 — F6 stable-state extension

Original F0–F5 above retain their v1.3 provenance and scientific flags. F6 uses its own locked v1.4 source.

| Run | Parent | NLL | Δpre | Δpost | Reduced NLL | Reduced Δpost | Flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| ST1-dclm-S188-S1-0.8B-lr1e-3 | 2.886706698 | 2.984526638 | +1.125950111 | +0.097819940 | 2.984526638 | +0.097819940 | 0 |
| ST1-dclm-S72-S1-0.2B-lr1e-3 | 2.908575296 | 3.066636571 | +0.945968926 | +0.158061275 | 3.066636571 | +0.158061275 | 0 |
| ST1-dclm-S72-S1-0.8B-lr1e-3 | 2.908575296 | 2.997767892 | +0.945968926 | +0.089192595 | 2.997767892 | +0.089192595 | 0 |
| ST1-fwe-S188-S1-0.8B-lr1e-3 | 3.023807592 | 3.113324808 | +1.175531177 | +0.089517217 | 3.113467590 | +0.089522732 | 0 |
| ST1-fwe-S188-S1-3.2B-lr1e-3 | 3.023807592 | 3.068752538 | +1.175531177 | +0.044944946 | 3.068890991 | +0.044946134 | 0 |
| ST1-fwe-S72-S1-0.8B-lr1e-3 | 3.042152002 | 3.123058220 | +0.965155970 | +0.080906218 | 3.123197970 | +0.080910756 | 0 |
| ST1-fwe-S72-S1-3.2B-lr1e-3 | 3.042152002 | 3.082068435 | +0.965155970 | +0.039916433 | 3.082204817 | +0.039917604 | 0 |

Full numeric results, CIs, exposure and failed checks follow; no failed row is silently marked verified.

```json
{
  "rows": [
    {
      "run_id": "ST1-dclm-S188-S1-0.8B-lr1e-3",
      "dataset": "dclm",
      "arm": "S188",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 2.88670669804388,
      "NLL": 2.9845266381346627,
      "delta_pre": 1.1259501105510854,
      "delta_post": 0.09781994009078249,
      "reduced_NLL": 2.9845266381346627,
      "reduced_delta_pre": 1.1259501105510854,
      "reduced_delta_post": 0.09781994009078249,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "exposure": {
        "status": "not applicable; FWE overlap only"
      },
      "flags": []
    },
    {
      "run_id": "ST1-dclm-S72-S1-0.2B-lr1e-3",
      "dataset": "dclm",
      "arm": "S72",
      "budget": "0.2B",
      "steps": 762,
      "tokens": 199753728,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "FT9",
      "parent": 2.908575296107871,
      "NLL": 3.0666365707899006,
      "delta_pre": 0.9459689258930865,
      "delta_post": 0.15806127468202968,
      "reduced_NLL": 3.0666365707899006,
      "reduced_delta_pre": 0.9459689258930865,
      "reduced_delta_post": 0.15806127468202968,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "exposure": {
        "status": "not applicable; FWE overlap only"
      },
      "flags": []
    },
    {
      "run_id": "ST1-dclm-S72-S1-0.8B-lr1e-3",
      "dataset": "dclm",
      "arm": "S72",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 2.908575296107871,
      "NLL": 2.997767891533186,
      "delta_pre": 0.9459689258930865,
      "delta_post": 0.08919259542531567,
      "reduced_NLL": 2.997767891533186,
      "reduced_delta_pre": 0.9459689258930865,
      "reduced_delta_post": 0.08919259542531567,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "exposure": {
        "status": "not applicable; FWE overlap only"
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S188-S1-0.8B-lr1e-3",
      "dataset": "fwe",
      "arm": "S188",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.0238075916258365,
      "NLL": 3.113324808186919,
      "delta_pre": 1.1755311765990717,
      "delta_post": 0.08951721656108358,
      "reduced_NLL": 3.1134675899283915,
      "reduced_delta_pre": 1.175551583551625,
      "reduced_delta_post": 0.08952273219749085,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "not seen",
        "actual_rows": 195264,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S188-S1-3.2B-lr1e-3",
      "dataset": "fwe",
      "arm": "S188",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.0238075916258365,
      "NLL": 3.068752537731151,
      "delta_pre": 1.1755311765990717,
      "delta_post": 0.044944946105315346,
      "reduced_NLL": 3.06889099141491,
      "reduced_delta_pre": 1.175551583551625,
      "reduced_delta_post": 0.04494613368400984,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "seen",
        "actual_rows": 781248,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 685
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S72-S1-0.8B-lr1e-3",
      "dataset": "fwe",
      "arm": "S72",
      "budget": "0.8B",
      "steps": 3051,
      "tokens": 799801344,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.042152001946299,
      "NLL": 3.1230582197769947,
      "delta_pre": 0.9651559695530937,
      "delta_post": 0.08090621783069607,
      "reduced_NLL": 3.1231979698317116,
      "reduced_delta_pre": 0.9652157434331661,
      "reduced_delta_post": 0.08091075634803858,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "not seen",
        "actual_rows": 195264,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    },
    {
      "run_id": "ST1-fwe-S72-S1-3.2B-lr1e-3",
      "dataset": "fwe",
      "arm": "S72",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "variant": "drop",
      "phase": "F6",
      "parent": 3.042152001946299,
      "NLL": 3.0820684346236717,
      "delta_pre": 0.9651559695530937,
      "delta_post": 0.03991643267737394,
      "reduced_NLL": 3.082204817023445,
      "reduced_delta_pre": 0.9652157434331661,
      "reduced_delta_post": 0.03991760353977203,
      "dropped_sequences": 3,
      "dropped_prediction_tokens": 12285,
      "exposure": {
        "status": "seen",
        "actual_rows": 781248,
        "documents": [
          {
            "pool_index": 28767024,
            "permutation_position": 1313362,
            "token_start": 1356141489,
            "token_stop": 1356142174,
            "text_sha256": "c15d0766b161a987ef155e8dd455a3859857942a98ee3e10b34b2cd996fcd318",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 685
            },
            "actual_consumed_tokens": 685
          },
          {
            "pool_index": 14947840,
            "permutation_position": 6125552,
            "token_start": 6303817369,
            "token_stop": 6303817568,
            "text_sha256": "5baa944c315747108d022c20546c5ad3223c3e913c871f9e5ece7813296aeb78",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          },
          {
            "pool_index": 28299105,
            "permutation_position": 17225137,
            "token_start": 17731899194,
            "token_stop": 17731899478,
            "text_sha256": "edb0ff7bea45d20e671713b99c22169de68dc5ba84949be8b648916c5dba9227",
            "prefix_tokens": {
              "199753728": 0,
              "799801344": 0,
              "3199991808": 0
            },
            "actual_consumed_tokens": 0
          }
        ]
      },
      "flags": []
    }
  ],
  "pairs": [
    {
      "dataset": "dclm",
      "budget": "0.8B",
      "full": {
        "E": 0.008627344665466826,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3945083557812922,
        "bootstrap": {
          "E_ci95": [
            0.008146503158218799,
            0.009101567875214807
          ],
          "E_over_G_ci95": [
            0.37560361078697274,
            0.4137477241031917
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.09672714000895767
      },
      "reduced": {
        "E": 0.008627344665466826,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3945083557812922,
        "bootstrap": {
          "E_ci95": [
            0.008146503158218799,
            0.009101567875214807
          ],
          "E_over_G_ci95": [
            0.37560361078697274,
            0.4137477241031917
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.09672714000895767
      },
      "difference": 0.0
    },
    {
      "dataset": "fwe",
      "budget": "0.8B",
      "full": {
        "E": 0.008610998730387487,
        "E_pre": 0.21037520704597804,
        "G": 0.018344410320462974,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.469407224323914,
        "bootstrap": {
          "E_ci95": [
            0.008100929041604121,
            0.009123613133210277
          ],
          "E_over_G_ci95": [
            0.4446942800383174,
            0.4946839052677514
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.10643185358641802
      },
      "reduced": {
        "E": 0.008611975849452258,
        "E_pre": 0.21033584011845904,
        "G": 0.018342355752773404,
        "dropped_sequences": 3,
        "dropped_prediction_tokens": 12285,
        "kept_sequences": 2458,
        "kept_prediction_tokens": 10065510,
        "E_over_G": 0.4695130748486387,
        "bootstrap": {
          "E_ci95": [
            0.008096019867176346,
            0.009126679800003918
          ],
          "E_over_G_ci95": [
            0.44441820935624904,
            0.4942767370292942
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.10643796002113416
      },
      "difference": 9.771190647711592e-07
    },
    {
      "dataset": "fwe",
      "budget": "3.2B",
      "full": {
        "E": 0.005028513427941408,
        "E_pre": 0.21037520704597804,
        "G": 0.018344410320462974,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.2741169293587028,
        "bootstrap": {
          "E_ci95": [
            0.004554093812499127,
            0.005517363521021101
          ],
          "E_over_G_ci95": [
            0.251801136453651,
            0.2968443200702413
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.1259760226717792
      },
      "reduced": {
        "E": 0.005028530144237816,
        "E_pre": 0.21033584011845904,
        "G": 0.018342355752773404,
        "dropped_sequences": 3,
        "dropped_prediction_tokens": 12285,
        "kept_sequences": 2458,
        "kept_prediction_tokens": 10065510,
        "E_over_G": 0.2741485451495232,
        "bootstrap": {
          "E_ci95": [
            0.0045502337907834065,
            0.005513036159708115
          ],
          "E_over_G_ci95": [
            0.25082823572612917,
            0.29673741020708644
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences (native A2 S estimator)",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "E_over_short_delta": 0.1259727463154852
      },
      "difference": 1.6716296407856546e-08
    }
  ],
  "missing_or_invalid": [],
  "FE6": {
    "dclm": {
      "label": "flat",
      "difference": -0.002504629739437653,
      "unverified": false
    },
    "fwe": {
      "label": "falls",
      "difference": -0.006169745700618403,
      "unverified": false
    }
  },
  "FE6_extended": {
    "fwe": {
      "label": "falls",
      "difference": -0.0035824853024460784
    }
  },
  "E02": {
    "dclm": 0.011131974404904479,
    "fwe": 0.01478074443100589
  },
  "overlap": {
    "immaterial": true,
    "primary_FWE": "full"
  },
  "interval_caveat": "OpenSci native final set is packed, not one document per row; paired sequence CI preserves A2 S and does not claim document-level independence.",
  "events": [
    {
      "name": "ST1-dclm-S72-S1-0.2B-lr1e-3",
      "returncode": 0,
      "seconds": 865.2041508480906,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-dclm-S72-S1-0.2B-lr1e-3.log"
    },
    {
      "name": "ST1-dclm-S72-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3063.4453087556176,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-dclm-S72-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-dclm-S188-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3066.0976773388684,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-dclm-S188-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S72-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3066.2043864717707,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S72-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S188-S1-0.8B-lr1e-3",
      "returncode": 0,
      "seconds": 3062.561924180016,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S188-S1-0.8B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S72-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11860.461530853994,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S72-S1-3.2B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S188-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11863.846811673604,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107/logs/ST1-fwe-S188-S1-3.2B-lr1e-3.log"
    }
  ],
  "extension": {
    "dataset": "fwe",
    "datasets": {
      "dclm": {
        "E02": 0.011131974404904479,
        "E08": 0.008627344665466826,
        "difference": -0.002504629739437653,
        "threshold": 0.003,
        "relative_change": 0.22499420572995332,
        "trigger": false
      },
      "fwe": {
        "E02": 0.01478074443100589,
        "E08": 0.008610998730387487,
        "difference": -0.006169745700618403,
        "threshold": 0.003,
        "relative_change": 0.4174177917369299,
        "trigger": true
      }
    },
    "tie_policy": "DCLM first for an exact relative-change tie"
  },
  "historical_02B": {
    "dclm": {
      "E_full": 0.011131974404904479,
      "E_reduced": 0.011131974404904479,
      "full": {
        "E": 0.011131974404904479,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.5090392338974279
      },
      "reduced": {
        "E": 0.011131974404904479,
        "E_pre": 0.17998118465799937,
        "G": 0.021868598063990462,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.5090392338974279
      },
      "delta_post": {
        "S72": 0.1584114774050014,
        "S188": 0.1695434518099055
      },
      "origin": "A2 S first node 6y3o; exact saved arrays; no rerun except FT9",
      "E_over_short_delta": 0.07027252436036568
    },
    "fwe": {
      "E_full": 0.01478074443100589,
      "E_reduced": 0.014785743201733266,
      "full": {
        "E": 0.01478074443100589,
        "E_pre": 0.21037520704597804,
        "G": 0.018344410320462974,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.8057355986263642
      },
      "reduced": {
        "E": 0.014785743201733266,
        "E_pre": 0.21033584011845904,
        "G": 0.018342355752773404,
        "dropped_sequences": 3,
        "dropped_prediction_tokens": 12285,
        "kept_sequences": 2458,
        "kept_prediction_tokens": 10065510,
        "E_over_G": 0.8060983769490803
      },
      "delta_post": {
        "S72": 0.1469997437496553,
        "S188": 0.16178048818066193
      },
      "origin": "A2 S first node 6y3o; exact saved arrays; no rerun except FT9",
      "E_over_short_delta": 0.10054945712135331
    }
  },
  "Smol_E_over_short_delta": [
    {
      "budget": "0.2B",
      "E": 0.01986042444939224,
      "short_delta_post": 0.23845270954987827,
      "E_over_short_delta": 0.08328873463791756,
      "primary": "full"
    },
    {
      "budget": "0.8B",
      "E": 0.013324723419155087,
      "short_delta_post": 0.16357947506698883,
      "E_over_short_delta": 0.08145718412226452,
      "primary": "full"
    },
    {
      "budget": "3.2B",
      "E": 0.00971901488669057,
      "short_delta_post": 0.11357928527113147,
      "E_over_short_delta": 0.08557031208190619,
      "primary": "full"
    }
  ],
  "missing_or_failed_runs": [],
  "FT9_pass": true,
  "FT9_flags": [],
  "joint_reading": "mixed",
  "joint_reading_unverified": true,
  "previous_decision": {
    "path": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01/results/a2_family_20261107/decision_family.json",
    "sha256": "9d2ab5b3388247780f270381c0c4ba585a1d66438aac0002b91e4408729557bb"
  },
  "historical_origin": "A2 S on 6y3o; full .2B values reused, not replaced by FT9; node differences retained",
  "FT9_reference": 3.0669867735128715
}
```


# Family v1.7 — F8 and F7

Spec SHA256: `70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209`. Owner-requested extension; original F0–F6 values/readings/flags stand unchanged. FT10 skipped: same wc69 node.

| Run | Parent | NLL | Δpre | Δpost | Flags |
|---|---:|---:|---:|---:|---:|
| ST1-dclm-S188-S1-3.2B-lr1e-3 | 2.886706698 | 2.941622756 | +1.125950111 | +0.054916058 | 0 |
| ST1-dclm-S72-S1-3.2B-lr1e-3 | 2.908575296 | 2.958319402 | +0.945968926 | +0.049744105 | 0 |
| Y3-dclm-P1T-L3-3.2B | 2.686908992 | 2.755552615 | +0.323383519 | +0.068643623 | 0 |
| Y3-dclm-P1T-L9-3.2B | 2.686908992 | 2.871552250 | +3.249153015 | +0.184643258 | 0 |
| Y3-dclm-P300-L3-3.2B | 2.737793032 | 2.789057126 | +0.273158845 | +0.051264094 | 0 |
| Y3-dclm-P300-L9-3.2B | 2.737793032 | 2.897751017 | +2.581002209 | +0.159957985 | 0 |

Paired sequence bootstrap: 10,000 resamples, seed 0; packed rows have 4095 scored targets each. Evaluation noise only. Full addendum quantities, provenance, missing runs and preserved flags:

```json
{
  "F8": {
    "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
    "pairs": {
      "F8-L3": {
        "E": 0.01737952924731561,
        "E_pre": 0.05022467417177427,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3415516750251005,
        "bootstrap": {
          "E_ci95": [
            0.016971670586036408,
            0.017802500551044386
          ],
          "E_over_G_ci95": [
            0.3347886794692613,
            0.3485257816926494
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.6584483249748996,
        "E_over_short_delta": 0.33901953410046853
      },
      "F8-L9": {
        "E": 0.0246852738633684,
        "E_pre": 0.668150805442168,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.4851280213927016,
        "bootstrap": {
          "E_ci95": [
            0.024083792822566712,
            0.025277338515471134
          ],
          "E_over_G_ci95": [
            0.4761762078153706,
            0.49406437528951674
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.5148719786072984,
        "E_over_short_delta": 0.1543234863876951
      }
    },
    "four_cut_table": {
      "6": {
        "E": 0.021670884569823023,
        "G": 0.050884040448750056,
        "parent_gain_kept": 0.5741123468438057,
        "node": "6y3o",
        "origin": "Stage1",
        "source": {
          "path": "resources/a2_depth_stage1_references_20261101.json",
          "sha256": "78afb6090646a84afc59d498911d37590ebce745a8e0a4d637d0411406651da5"
        }
      },
      "12": {
        "E": 0.02332383536097593,
        "G": 0.050884040448750056,
        "parent_gain_kept": 0.5416276861019422,
        "node": "wc69",
        "origin": "Why",
        "historical_flags": [
          [
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            }
          ],
          [
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 0
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 1
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 2
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 3
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 4
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 5
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 6
            },
            {
              "test": "preparation:stream:fwe:2031:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2031:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            },
            {
              "test": "preparation:stream:fwe:2032:YT2_CulturaX_heldout",
              "passed": false,
              "detail": {
                "test": "stream:fwe:2032:YT2_CulturaX_heldout",
                "passed": false,
                "detail": {
                  "content_hash_overlap": 3,
                  "heldout_documents": 16862,
                  "id_disjoint": "Dataset-qualified ids: corpus/revision/source/row versus CulturaX split/index"
                },
                "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea"
              },
              "spec_sha256": "87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea",
              "rank": 7
            }
          ]
        ],
        "qualification": "DCLM cleared by independent Why overlap/row audit; original historical flags retained"
      },
      "3": {
        "E": 0.01737952924731561,
        "E_pre": 0.05022467417177427,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.3415516750251005,
        "bootstrap": {
          "E_ci95": [
            0.016971670586036408,
            0.017802500551044386
          ],
          "E_over_G_ci95": [
            0.3347886794692613,
            0.3485257816926494
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.6584483249748996,
        "E_over_short_delta": 0.33901953410046853
      },
      "9": {
        "E": 0.0246852738633684,
        "E_pre": 0.668150805442168,
        "G": 0.050884040448750244,
        "dropped_sequences": 0,
        "dropped_prediction_tokens": 0,
        "kept_sequences": 2461,
        "kept_prediction_tokens": 10077795,
        "E_over_G": 0.4851280213927016,
        "bootstrap": {
          "E_ci95": [
            0.024083792822566712,
            0.025277338515471134
          ],
          "E_over_G_ci95": [
            0.4761762078153706,
            0.49406437528951674
          ],
          "resamples": 10000,
          "seed": 0,
          "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
          "estimator": "token weighted",
          "zero_token_draws": 0,
          "undefined_ratio_draws": 0
        },
        "node": "wc69",
        "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
        "flags": [],
        "unverified": false,
        "parent_gain_kept": 0.5148719786072984,
        "E_over_short_delta": 0.1543234863876951
      }
    },
    "changes": {
      "3": {
        "E08": 0.018639719725659365,
        "E32": 0.01737952924731561,
        "difference": -0.0012601904783437548,
        "label": "flat",
        "direction": "down"
      },
      "9": {
        "E08": 0.022253106709633208,
        "E32": 0.0246852738633684,
        "difference": 0.0024321671537351937,
        "label": "flat",
        "direction": "up"
      }
    },
    "FY3_32B": "mixed",
    "tau": 0.003,
    "secondary": true,
    "original_Why_08_reading_unchanged": "A not supported",
    "slope_per_layer": 0.000694910254484212,
    "E12_minus_E3": 0.005944306113660321,
    "unverified": false,
    "caveat": "DCLM only; within-node E; historical L6 from 6y3o, other cuts wc69. Evaluation noise only, not document independence."
  },
  "F7": {
    "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
    "secondary": true,
    "series": {
      "dclm": {
        "0.2B": {
          "E": 0.011131974404904479,
          "node": "6y3o"
        },
        "0.8B": {
          "E": 0.008627344665466826,
          "E_pre": 0.17998118465799937,
          "G": 0.021868598063990462,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.3945083557812922,
          "bootstrap": {
            "E_ci95": [
              0.008146503158218799,
              0.009101567875214807
            ],
            "E_over_G_ci95": [
              0.37560361078697274,
              0.4137477241031917
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences (native A2 S estimator)",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "E_over_short_delta": 0.09672714000895767,
          "node": "wc69",
          "origin": "F6"
        },
        "3.2B": {
          "E": 0.005171952894106886,
          "E_pre": 0.17998118465799937,
          "G": 0.021868598063990462,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.2365013467700607,
          "bootstrap": {
            "E_ci95": [
              0.004717248259710383,
              0.00562745525299936
            ],
            "E_over_G_ci95": [
              0.21880310598180264,
              0.2540003650582174
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "node": "wc69",
          "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
          "flags": [],
          "unverified": false,
          "parent_gain_kept": 0.7634986532299393,
          "E_over_short_delta": 0.10397117098349956
        }
      },
      "fwe": {
        "0.2B": {
          "E": 0.01478074443100589,
          "node": "6y3o"
        },
        "0.8B": {
          "E": 0.008610998730387487,
          "E_pre": 0.21037520704597804,
          "G": 0.018344410320462974,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.469407224323914,
          "bootstrap": {
            "E_ci95": [
              0.008100929041604121,
              0.009123613133210277
            ],
            "E_over_G_ci95": [
              0.4446942800383174,
              0.4946839052677514
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences (native A2 S estimator)",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "E_over_short_delta": 0.10643185358641802,
          "node": "wc69",
          "origin": "F6"
        },
        "3.2B": {
          "E": 0.005028513427941408,
          "E_pre": 0.21037520704597804,
          "G": 0.018344410320462974,
          "dropped_sequences": 0,
          "dropped_prediction_tokens": 0,
          "kept_sequences": 2461,
          "kept_prediction_tokens": 10077795,
          "E_over_G": 0.2741169293587028,
          "bootstrap": {
            "E_ci95": [
              0.004554093812499127,
              0.005517363521021101
            ],
            "E_over_G_ci95": [
              0.251801136453651,
              0.2968443200702413
            ],
            "resamples": 10000,
            "seed": 0,
            "unit": "paired fixed-length packed sequences (native A2 S estimator)",
            "estimator": "token weighted",
            "zero_token_draws": 0,
            "undefined_ratio_draws": 0
          },
          "E_over_short_delta": 0.1259760226717792,
          "node": "wc69",
          "origin": "F6"
        }
      }
    },
    "original_FE6_and_joint_reading_unchanged": true,
    "pair": {
      "E": 0.005171952894106886,
      "E_pre": 0.17998118465799937,
      "G": 0.021868598063990462,
      "dropped_sequences": 0,
      "dropped_prediction_tokens": 0,
      "kept_sequences": 2461,
      "kept_prediction_tokens": 10077795,
      "E_over_G": 0.2365013467700607,
      "bootstrap": {
        "E_ci95": [
          0.004717248259710383,
          0.00562745525299936
        ],
        "E_over_G_ci95": [
          0.21880310598180264,
          0.2540003650582174
        ],
        "resamples": 10000,
        "seed": 0,
        "unit": "paired fixed-length packed sequences; native Why/F6 estimator",
        "estimator": "token weighted",
        "zero_token_draws": 0,
        "undefined_ratio_draws": 0
      },
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209",
      "flags": [],
      "unverified": false,
      "parent_gain_kept": 0.7634986532299393,
      "E_over_short_delta": 0.10397117098349956
    },
    "FE6_DCLM_32B": {
      "label": "falls",
      "difference": -0.00345539177135994
    },
    "change02_to32": -0.005960021510797593,
    "unverified": false
  },
  "rows": [
    {
      "run_id": "ST1-dclm-S188-S1-3.2B-lr1e-3",
      "arm": "S188",
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F7",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.88670669804388,
      "NLL": 2.9416227563970634,
      "delta_pre": 1.1259501105510854,
      "delta_post": 0.05491605835318274,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "ST1-dclm-S72-S1-3.2B-lr1e-3",
      "arm": "S72",
      "removed": [
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F7",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.908575296107871,
      "NLL": 2.9583194015669463,
      "delta_pre": 0.9459689258930865,
      "delta_post": 0.049744105459075864,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P1T-L3-3.2B",
      "arm": "P1T",
      "removed": [
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.6869089917639344,
      "NLL": 2.755552614962147,
      "delta_pre": 0.32338351898639756,
      "delta_post": 0.06864362319821224,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P1T-L9-3.2B",
      "arm": "P1T",
      "removed": [
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.6869089917639344,
      "NLL": 2.8715522502554713,
      "delta_pre": 3.249153014701919,
      "delta_post": 0.18464325849153734,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P300-L3-3.2B",
      "arm": "P300",
      "removed": [
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.7377930322126844,
      "NLL": 2.789057126163582,
      "delta_pre": 0.2731588448146232,
      "delta_post": 0.05126409395089665,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    },
    {
      "run_id": "Y3-dclm-P300-L9-3.2B",
      "arm": "P300",
      "removed": [
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21
      ],
      "phase": "F8",
      "dataset": "dclm",
      "budget": "3.2B",
      "steps": 12207,
      "tokens": 3199991808,
      "lr": 0.001,
      "stream_seed": 2030,
      "seed": 0,
      "variant": "drop",
      "parent": 2.7377930322126844,
      "NLL": 2.8977510168408527,
      "delta_pre": 2.5810022092597524,
      "delta_post": 0.15995798462816896,
      "flags": [],
      "node": "wc69",
      "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
    }
  ],
  "missing_or_invalid": [],
  "missing_runs": [],
  "coverage_complete": true,
  "events": [
    {
      "name": "Y3-dclm-P300-L3-3.2B",
      "returncode": 0,
      "seconds": 13123.10453401506,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P300-L3-3.2B.log"
    },
    {
      "name": "Y3-dclm-P1T-L3-3.2B",
      "returncode": 0,
      "seconds": 13120.96041374607,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P1T-L3-3.2B.log"
    },
    {
      "name": "Y3-dclm-P300-L9-3.2B",
      "returncode": 0,
      "seconds": 10599.895367615856,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P300-L9-3.2B.log"
    },
    {
      "name": "Y3-dclm-P1T-L9-3.2B",
      "returncode": 0,
      "seconds": 10601.89562015608,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/Y3-dclm-P1T-L9-3.2B.log"
    },
    {
      "name": "ST1-dclm-S72-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11871.421059853863,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/ST1-dclm-S72-S1-3.2B-lr1e-3.log"
    },
    {
      "name": "ST1-dclm-S188-S1-3.2B-lr1e-3",
      "returncode": 0,
      "seconds": 11866.271983564831,
      "log": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107/logs/ST1-dclm-S188-S1-3.2B-lr1e-3.log"
    }
  ],
  "pause": {
    "state": "paused",
    "arm": "P1T",
    "saved_state": {
      "path": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01/pause/resume_state.json",
      "sha256": "b5597baa07d79c6bc82e41a5905969759e9375ef55ccf0794a3110ffe10ccc7b"
    },
    "supervisor": {
      "started_at": "2026-11-06T19:20:32.451384+00:00",
      "training_status": "failed",
      "training_error": "KeyboardInterrupt('Supervisor received signal 15')",
      "free_after_training": [
        {
          "index": 0,
          "uuid": "GPU-08f82f92-6e9a-1b21-56cb-4b359c898e3c",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 1,
          "uuid": "GPU-2909c891-748d-a78f-f72c-181231c00e7a",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 2,
          "uuid": "GPU-d5679196-f3b9-f2fd-2a95-fc23676711ba",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 3,
          "uuid": "GPU-8ec230a4-1d3c-13c5-3681-51bceba7df9a",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 4,
          "uuid": "GPU-cdb11306-b5c0-52f0-d871-bba3d94e34d7",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 5,
          "uuid": "GPU-d6a96000-88e7-7653-6d72-9a8e3bddf198",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 6,
          "uuid": "GPU-96e1284a-b396-0253-6511-1d6b2ce193fa",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        },
        {
          "index": 7,
          "uuid": "GPU-817e7cb4-510c-c53f-9db1-c5752571469f",
          "name": "NVIDIA B200",
          "memory_total_mib": 183359.0,
          "memory_used_mib": 0.0,
          "utilization_percent": 0.0,
          "pids": []
        }
      ],
      "burn": {
        "at": "2026-11-08T17:07:24.755173+00:00",
        "session": "a2_family_20261107_a01_burn",
        "gpus": [
          {
            "index": 0,
            "uuid": "GPU-08f82f92-6e9a-1b21-56cb-4b359c898e3c",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108745
            ]
          },
          {
            "index": 1,
            "uuid": "GPU-2909c891-748d-a78f-f72c-181231c00e7a",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108746
            ]
          },
          {
            "index": 2,
            "uuid": "GPU-d5679196-f3b9-f2fd-2a95-fc23676711ba",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108747
            ]
          },
          {
            "index": 3,
            "uuid": "GPU-8ec230a4-1d3c-13c5-3681-51bceba7df9a",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108748
            ]
          },
          {
            "index": 4,
            "uuid": "GPU-cdb11306-b5c0-52f0-d871-bba3d94e34d7",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108749
            ]
          },
          {
            "index": 5,
            "uuid": "GPU-d6a96000-88e7-7653-6d72-9a8e3bddf198",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108750
            ]
          },
          {
            "index": 6,
            "uuid": "GPU-96e1284a-b396-0253-6511-1d6b2ce193fa",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108751
            ]
          },
          {
            "index": 7,
            "uuid": "GPU-817e7cb4-510c-c53f-9db1-c5752571469f",
            "name": "NVIDIA B200",
            "memory_total_mib": 183359.0,
            "memory_used_mib": 155212.0,
            "utilization_percent": 100.0,
            "pids": [
              108752
            ]
          }
        ],
        "collective_progress_verified": true
      },
      "finished_at": "2026-11-08T17:07:24.755480+00:00"
    },
    "checkpoint": "/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01/pause/checkpoint-15000",
    "global_step": 15000,
    "handoff_verified": true,
    "spec_sha256": "70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209"
  }
}
```
