# A2 Q — quantization trên cùng 12 parents

Spec SHA256: `a7fb04042d54f977cff4ea5c2275b6b2af3a7c00cce4b2898ffc3dc9383c44da`. Frozen source: `422bb7f118fd875ea2d9392e5da0cefbfbfb8d30`.

Trạng thái thực thi: {"spec_sha256": "a7fb04042d54f977cff4ea5c2275b6b2af3a7c00cce4b2898ffc3dc9383c44da", "execution_complete": true, "registered_models": 80, "nll_files": 160, "downstream_models": 16, "elapsed_seconds": 8926.140910940245, "node_hours": 2.4794835865861065, "coverage_complete": true, "flagged_checks": 9610, "calibration_disjointness_passed": false, "decision": {"path": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/decision_quant.json", "sha256": "71a126fd980fe7d491ef45581053be1395de62a2804969c90ac4a340913170fb"}, "tests": {"path": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/tests.json", "sha256": "244a0ccd663294b30339d568f8eedbc25a4cf0f56d7284f7a4fe5b8bcc2ed39f"}}.

Final CulturaX là decision set; own-corpus chỉ diagnostic. Trọng số dequantized được làm tròn BF16, giữ trong tensor FP32; forward BF16 autocast và CE FP32. Mọi NLL/CI xuất phát từ file per-sequence, batch validation 1, downstream batch 2. Không healing, không thêm architecture/LR/exposure test.

## Damage và contrasts

Nats: compressed NLL − parent NLL. C/D(R) là relative contrast; D/val(parent) là relative damage và được ghi đầy đủ trong decision_quant.json. Bootstrap 10.000 lần, seed 0, giữ cùng draws cho mọi model trong dataset. R_long so sánh 1T với 776B nên chỉ descriptive; I cũng chỉ descriptive.

### dclm

| Compression | L_ann: nats / C÷D(R) / label | L_stab | R_short | R_long |
|---|---|---|---|---|
| pre-S1 | +0.291830382 / 0.284440 / more damage; CI [+0.288080788, +0.295605191] | +0.179981185 / 0.190261 / more damage; CI [+0.177190642, +0.182841707] | +0.080013186 / 0.084583 / grey; CI [+0.078213162, +0.081827300] | +0.191862383 / 0.170400 / descriptive only; CI [+0.188840202, +0.194864185] |
| rtn4 | +0.001039562 / 0.097636 / no effect; CI [+0.000854571, +0.001220510] | +0.001215074 / 0.130847 / no effect; CI [+0.001055137, +0.001373104] | +0.001361130 / 0.146575 / no effect; CI [+0.001190271, +0.001530085] | +0.001185617 / 0.112902 / descriptive only; CI [+0.001014360, +0.001360015] |
| rtn3 | +0.011184158 / 0.170112 / more damage; CI [+0.010752879, +0.011621517] | -0.006845503 / -0.125129 / less damage; CI [-0.007241054, -0.006450499] | +0.011038314 / 0.201769 / more damage; CI [+0.010641147, +0.011436267] | +0.029067975 / 0.607327 / descriptive only; CI [+0.028645457, +0.029500312] |
| gptq4 | +0.001443580 / 0.209627 / no effect; CI [+0.001295633, +0.001582855] | +0.000263952 / 0.042945 / no effect; CI [+0.000136700, +0.000393228] | +0.000740105 / 0.120414 / no effect; CI [+0.000610620, +0.000870638] | +0.001919734 / 0.299477 / descriptive only; CI [+0.001779936, +0.002057829] |
| gptq3 | +0.020258419 / 0.573416 / more damage; CI [+0.019900407, +0.020613748] | +0.000445813 / 0.014048 / no effect; CI [+0.000144404, +0.000741646] | +0.003595412 / 0.113299 / more damage; CI [+0.003286911, +0.003899501] | +0.023408019 / 0.727414 / descriptive only; CI [+0.023048669, +0.023778450] |

Healed pruning 3.2B: E=+0.021670884569823; A_init=+0.012113131873957 nats. A_init so sánh hai student, không phải hai damage.

| Parent | Compression | Parent NLL | Compressed NLL | D | D/parent |
|---|---|---:|---:|---:|---:|
| S72 | pre-S1 | 2.908575296108 | 3.854544222001 | +0.945968925893 | +0.325234463 |
| P300 | pre-S1 | 2.737793032213 | 3.763775143788 | +1.025982111575 | +0.374747872 |
| S188 | pre-S1 | 2.886706698044 | 4.012656808595 | +1.125950110551 | +0.390046592 |
| P1T | pre-S1 | 2.686908991764 | 4.004721485197 | +1.317812493433 | +0.490456691 |
| S72 | gptq3-cal1 | 2.908575296108 | 2.939855563304 | +0.031280267196 | +0.010754498 |
| P300 | gptq3-cal1 | 2.737793032213 | 2.774439660804 | +0.036646628591 | +0.013385463 |
| S188 | gptq3-cal1 | 2.886706698044 | 2.918769706328 | +0.032063008284 | +0.011107124 |
| P1T | gptq3-cal1 | 2.686908991764 | 2.743838014036 | +0.056929022272 | +0.021187551 |
| S72 | gptq3 | 2.908575296108 | 2.940309249585 | +0.031733953477 | +0.010910480 |
| P300 | gptq3 | 2.737793032213 | 2.773122398046 | +0.035329365834 | +0.012904323 |
| S188 | gptq3 | 2.886706698044 | 2.918886464814 | +0.032179766770 | +0.011147571 |
| P1T | gptq3 | 2.686908991764 | 2.742496777064 | +0.055587785300 | +0.020688377 |
| S72 | gptq4 | 2.908575296108 | 2.914721623197 | +0.006146327089 | +0.002113174 |
| P300 | gptq4 | 2.737793032213 | 2.744679464673 | +0.006886432461 | +0.002515323 |
| S188 | gptq4 | 2.886706698044 | 2.893116977124 | +0.006410279080 | +0.002220620 |
| P1T | gptq4 | 2.686908991764 | 2.695239004446 | +0.008330012682 | +0.003100221 |
| S72 | rtn3 | 2.908575296108 | 2.963282927930 | +0.054707631822 | +0.018809082 |
| P300 | rtn3 | 2.737793032213 | 2.803538977965 | +0.065745945752 | +0.024014213 |
| S188 | rtn3 | 2.886706698044 | 2.934568827327 | +0.047862129283 | +0.016580184 |
| P1T | rtn3 | 2.686908991764 | 2.763839095581 | +0.076930103817 | +0.028631451 |
| S72 | rtn4 | 2.908575296108 | 2.917861542017 | +0.009286245909 | +0.003192713 |
| P300 | rtn4 | 2.737793032213 | 2.748440407820 | +0.010647375608 | +0.003889036 |
| S188 | rtn4 | 2.886706698044 | 2.897208018378 | +0.010501320334 | +0.003637820 |
| P1T | rtn4 | 2.686908991764 | 2.698595929261 | +0.011686937497 | +0.004349584 |

### fwe

| Compression | L_ann: nats / C÷D(R) / label | L_stab | R_short | R_long |
|---|---|---|---|---|
| pre-S1 | +0.354417524 / 0.368472 / more damage; CI [+0.351273121, +0.357529106] | +0.210375207 / 0.217970 / more damage; CI [+0.207566710, +0.213225269] | -0.003298669 / -0.003418 / no effect; CI [-0.005585537, -0.000905559] | +0.140743647 / 0.119728 / descriptive only; CI [+0.138150970, +0.143293198] |
| rtn4 | +0.000498817 / 0.047367 / no effect; CI [+0.000310691, +0.000690550] | +0.001806636 / 0.247828 / no effect; CI [+0.001646226, +0.001964913] | +0.003241011 / 0.444591 / more damage; CI [+0.003071174, +0.003404554] | +0.001933193 / 0.212520 / descriptive only; CI [+0.001748783, +0.002117436] |
| rtn3 | +0.002594577 / 0.047756 / no effect; CI [+0.002117734, +0.003063890] | +0.007769165 / 0.198648 / more damage; CI [+0.007239331, +0.008294300] | +0.015220164 / 0.389160 / more damage; CI [+0.014829803, +0.015605396] | +0.010045577 / 0.214285 / descriptive only; CI [+0.009495119, +0.010576626] |
| gptq4 | +0.000294480 / 0.040604 / no effect; CI [+0.000140034, +0.000450778] | +0.000228833 / 0.048438 / no effect; CI [+0.000097787, +0.000360809] | +0.002528251 / 0.535170 / no effect; CI [+0.002389871, +0.002662818] | +0.002593898 / 0.523698 / descriptive only; CI [+0.002445226, +0.002746957] |
| gptq3 | +0.010195175 / 0.345459 / more damage; CI [+0.009876725, +0.010508563] | +0.004798875 / 0.202445 / more damage; CI [+0.004498679, +0.005100016] | +0.005807322 / 0.244987 / more damage; CI [+0.005508324, +0.006115979] | +0.011203622 / 0.393061 / descriptive only; CI [+0.010865892, +0.011535211] |

Healed pruning 3.2B: E=+0.023461150403908; A_init=+0.012993651189637 nats. A_init so sánh hai student, không phải hai damage.

| Parent | Compression | Parent NLL | Compressed NLL | D | D/parent |
|---|---|---:|---:|---:|---:|
| S72 | pre-S1 | 3.042152001946 | 4.007307971499 | +0.965155969553 | +0.317260929 |
| P300 | pre-S1 | 2.860631325857 | 3.822488626156 | +0.961857300299 | +0.336239519 |
| S188 | pre-S1 | 3.023807591626 | 4.199338768225 | +1.175531176599 | +0.388758590 |
| P1T | pre-S1 | 2.808707333602 | 4.124982157487 | +1.316274823885 | +0.468640790 |
| S72 | gptq3-cal1 | 3.042152001946 | 3.065879005527 | +0.023727003581 | +0.007799414 |
| P300 | gptq3-cal1 | 2.860631325857 | 2.890623471489 | +0.029992145632 | +0.010484450 |
| S188 | gptq3-cal1 | 3.023807591626 | 3.052447145528 | +0.028639553902 | +0.009471355 |
| P1T | gptq3-cal1 | 2.808707333602 | 2.848210106751 | +0.039502773150 | +0.014064396 |
| S72 | gptq3 | 3.042152001946 | 3.065856613834 | +0.023704611887 | +0.007792054 |
| P300 | gptq3 | 2.860631325857 | 2.890143259673 | +0.029511933816 | +0.010316581 |
| S188 | gptq3 | 3.023807591626 | 3.052311078385 | +0.028503486760 | +0.009426356 |
| P1T | gptq3 | 2.808707333602 | 2.848414442146 | +0.039707108544 | +0.014137147 |
| S72 | gptq4 | 3.042152001946 | 3.046876206406 | +0.004724204459 | +0.001552915 |
| P300 | gptq4 | 2.860631325857 | 2.867883781259 | +0.007252455403 | +0.002535264 |
| S188 | gptq4 | 3.023807591626 | 3.028760628998 | +0.004953037373 | +0.001638013 |
| P1T | gptq4 | 2.808707333602 | 2.816254268545 | +0.007546934943 | +0.002686978 |
| S72 | rtn3 | 3.042152001946 | 3.081262267948 | +0.039110266002 | +0.012856118 |
| P300 | rtn3 | 2.860631325857 | 2.914961756164 | +0.054330430307 | +0.018992461 |
| S188 | rtn3 | 3.023807591626 | 3.070687022585 | +0.046879430959 | +0.015503444 |
| P1T | rtn3 | 2.808707333602 | 2.865632341260 | +0.056925007658 | +0.020267333 |
| S72 | rtn4 | 3.042152001946 | 3.049441876500 | +0.007289874553 | +0.002396289 |
| P300 | rtn4 | 2.860631325857 | 2.871162211870 | +0.010530886013 | +0.003681315 |
| S188 | rtn4 | 3.023807591626 | 3.032904102262 | +0.009096510636 | +0.003008297 |
| P1T | rtn4 | 2.808707333602 | 2.819737037068 | +0.011029703467 | +0.003926968 |

### nem

| Compression | L_ann: nats / C÷D(R) / label | L_stab | R_short | R_long |
|---|---|---|---|---|
| pre-S1 | +0.300869945 / 0.275450 / more damage; CI [+0.297420992, +0.304376660] | +0.314278303 / 0.349028 / more damage; CI [+0.311382091, +0.317237365] | +0.191844385 / 0.213056 / more damage; CI [+0.190334490, +0.193430743] | +0.178436027 / 0.146895 / descriptive only; CI [+0.175693562, +0.181216712] |
| rtn4 | +0.001168387 / 0.101376 / no effect; CI [+0.000988339, +0.001350418] | +0.000471532 / 0.062460 / no effect; CI [+0.000309035, +0.000630848] | +0.003975890 / 0.526654 / more damage; CI [+0.003816732, +0.004132913] | +0.004672744 / 0.582573 / descriptive only; CI [+0.004493171, +0.004849163] |
| rtn3 | +0.006042037 / 0.103600 / more damage; CI [+0.005653291, +0.006438250] | +0.000604966 / 0.014638 / no effect; CI [+0.000242331, +0.000968716] | +0.016991840 / 0.411136 / more damage; CI [+0.016647573, +0.017340733] | +0.022428912 / 0.534863 / descriptive only; CI [+0.022026379, +0.022837982] |
| gptq4 | +0.001126334 / 0.150629 / no effect; CI [+0.000984238, +0.001272154] | +0.000384071 / 0.067749 / no effect; CI [+0.000261745, +0.000512302] | +0.001808539 / 0.319024 / no effect; CI [+0.001680682, +0.001935738] | +0.002550802 / 0.421408 / descriptive only; CI [+0.002403247, +0.002695375] |
| gptq3 | +0.005998788 / 0.165109 / more damage; CI [+0.005680269, +0.006323101] | +0.000588799 / 0.020844 / no effect; CI [+0.000307237, +0.000872198] | +0.008084883 / 0.286218 / more damage; CI [+0.007805785, +0.008364788] | +0.013494872 / 0.467985 / descriptive only; CI [+0.013174685, +0.013828629] |

| Parent | Compression | Parent NLL | Compressed NLL | D | D/parent |
|---|---|---:|---:|---:|---:|
| S72 | pre-S1 | 2.952463783417 | 3.852902816181 | +0.900439032764 | +0.304978858 |
| P300 | pre-S1 | 2.774250937160 | 3.866534354761 | +1.092283417601 | +0.393721924 |
| S188 | pre-S1 | 2.926708960676 | 4.141426296314 | +1.214717335637 | +0.415045483 |
| P1T | pre-S1 | 2.723382862027 | 4.116536224985 | +1.393153362958 | +0.511552519 |
| S72 | gptq3 | 2.952463783417 | 2.980711117567 | +0.028247334150 | +0.009567377 |
| P300 | gptq3 | 2.774250937160 | 2.810583154749 | +0.036332217589 | +0.013096226 |
| S188 | gptq3 | 2.926708960676 | 2.955545094299 | +0.028836133622 | +0.009852751 |
| P1T | gptq3 | 2.723382862027 | 2.765713868086 | +0.042331006059 | +0.015543538 |
| S72 | gptq4 | 2.952463783417 | 2.958132764744 | +0.005668981327 | +0.001920085 |
| P300 | gptq4 | 2.774250937160 | 2.781728457187 | +0.007477520027 | +0.002695329 |
| S188 | gptq4 | 2.926708960676 | 2.932762012587 | +0.006053051910 | +0.002068211 |
| P1T | gptq4 | 2.723382862027 | 2.731986715881 | +0.008603853854 | +0.003159252 |
| S72 | rtn3 | 2.952463783417 | 2.993792744874 | +0.041328961457 | +0.013998126 |
| P300 | rtn3 | 2.774250937160 | 2.832571738604 | +0.058320801444 | +0.021022180 |
| S188 | rtn3 | 2.926708960676 | 2.968642887806 | +0.041933927129 | +0.014328014 |
| P1T | rtn3 | 2.723382862027 | 2.787745700865 | +0.064362838838 | +0.023633416 |
| S72 | rtn4 | 2.952463783417 | 2.960013122666 | +0.007549339249 | +0.002556963 |
| P300 | rtn4 | 2.774250937160 | 2.785776165979 | +0.011525228819 | +0.004154357 |
| S188 | rtn4 | 2.926708960676 | 2.934729832262 | +0.008020871585 | +0.002740577 |
| P1T | rtn4 | 2.723382862027 | 2.736076477795 | +0.012693615768 | +0.004660974 |

## Labels, calibration và kiểm tra

T=max(0.003,0.10D(R)); t=max(0.003,0.03D(R)). Thứ tự: more damage, less damage, no effect, grey. Mọi clause boolean, I, cross-dataset agreement và cal0/cal1 flag được ghi trong decision_quant.json.

QT6 calibration disjointness: False. Overlaps: `{"cal0:cal1": {"ids": [["train", 25978874]], "content_hashes": ["cd0f63b818a2b11ebc4783a1dfd49700266833e76b0b0538b4dd2639edb40d7c"]}, "cal0:tuning": {"ids": [], "content_hashes": []}, "cal0:final": {"ids": [], "content_hashes": []}, "cal1:tuning": {"ids": [], "content_hashes": []}, "cal1:final": {"ids": [], "content_hashes": []}}`.

Nếu hai prefix calibration chạm cùng document ở ranh giới đóng gói, giữ đúng first128/next128 theo spec, ghi QT6 FAIL và đánh dấu mọi kết quả phụ thuộc calibration là unverified. Không thay prefix âm thầm.

| Check | PASS | FLAG |
|---|---:|---:|
| GPTQ_layer | 792 | 0 |
| GPTQ_matrix | 5544 | 0 |
| QT1 | 12 | 0 |
| QT2 | 6 | 0 |
| QT3 | 0 | 9578 |
| QT4 | 3 | 0 |
| QT5 | 1 | 0 |
| QT6 | 0 | 1 |
| QT6_inherited | 0 | 32 |
| QT7 | 68 | 0 |
| accelerate_configuration | 1 | 0 |
| accelerate_env | 1 | 0 |
| all_nll_statistics | 1 | 0 |
| baseline:dclm:P1T | 1 | 0 |
| baseline:dclm:P300 | 1 | 0 |
| baseline:dclm:S188 | 1 | 0 |
| baseline:dclm:S72 | 1 | 0 |
| baseline:fwe:P1T | 1 | 0 |
| baseline:fwe:P300 | 1 | 0 |
| baseline:fwe:S188 | 1 | 0 |
| baseline:fwe:S72 | 1 | 0 |
| baseline:nem:P1T | 1 | 0 |
| baseline:nem:P300 | 1 | 0 |
| baseline:nem:S188 | 1 | 0 |
| baseline:nem:S72 | 1 | 0 |
| cal1:dclm:P1T | 1 | 0 |
| cal1:dclm:P300 | 1 | 0 |
| cal1:dclm:S188 | 1 | 0 |
| cal1:dclm:S72 | 1 | 0 |
| cal1:fwe:P1T | 1 | 0 |
| cal1:fwe:P300 | 1 | 0 |
| cal1:fwe:S188 | 1 | 0 |
| cal1:fwe:S72 | 1 | 0 |
| data:/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/packed/dclm:tuning | 1 | 0 |
| data:/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/packed/fineweb_edu:tuning | 1 | 0 |
| data:/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/data:final | 1 | 0 |
| data:/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/secondary/nem:final | 1 | 0 |
| downstream_CI:dclm | 1 | 0 |
| downstream_CI:fwe | 1 | 0 |
| gptq:dclm:P1T | 1 | 0 |
| gptq:dclm:P300 | 1 | 0 |
| gptq:dclm:S188 | 1 | 0 |
| gptq:dclm:S72 | 1 | 0 |
| gptq:fwe:P1T | 1 | 0 |
| gptq:fwe:P300 | 1 | 0 |
| gptq:fwe:S188 | 1 | 0 |
| gptq:fwe:S72 | 1 | 0 |
| gptq:nem:P1T | 1 | 0 |
| gptq:nem:P300 | 1 | 0 |
| gptq:nem:S188 | 1 | 0 |
| gptq:nem:S72 | 1 | 0 |
| independent_NLL_recomputation | 1 | 0 |
| independent_paired:dclm | 1 | 0 |
| independent_paired:fwe | 1 | 0 |
| independent_replay:dclm/P1T-parent | 1 | 0 |
| independent_replay:dclm/P1T-quantized | 1 | 0 |
| independent_replay:dclm/P300-parent | 1 | 0 |
| independent_replay:dclm/P300-quantized | 1 | 0 |
| independent_replay:dclm/S188-parent | 1 | 0 |
| independent_replay:dclm/S188-quantized | 1 | 0 |
| independent_replay:dclm/S72-parent | 1 | 0 |
| independent_replay:dclm/S72-quantized | 1 | 0 |
| independent_replay:fwe/P1T-parent | 1 | 0 |
| independent_replay:fwe/P1T-quantized | 1 | 0 |
| independent_replay:fwe/P300-parent | 1 | 0 |
| independent_replay:fwe/P300-quantized | 1 | 0 |
| independent_replay:fwe/S188-parent | 1 | 0 |
| independent_replay:fwe/S188-quantized | 1 | 0 |
| independent_replay:fwe/S72-parent | 1 | 0 |
| independent_replay:fwe/S72-quantized | 1 | 0 |
| independent_statistics:dclm | 1 | 0 |
| independent_statistics:fwe | 1 | 0 |
| interim_labels | 1 | 0 |
| pooled_followup | 1 | 0 |
| primary_replay:dclm/P1T-quantized | 1 | 0 |
| primary_replay:dclm/P300-quantized | 1 | 0 |
| primary_replay:dclm/S188-parent | 1 | 0 |
| primary_replay:dclm/S188-quantized | 1 | 0 |
| primary_replay:dclm/S72-quantized | 1 | 0 |
| primary_replay:fwe/P1T-quantized | 1 | 0 |
| primary_replay:fwe/P300-quantized | 1 | 0 |
| primary_replay:fwe/S188-parent | 1 | 0 |
| primary_replay:fwe/S188-quantized | 1 | 0 |
| primary_replay:fwe/S72-quantized | 1 | 0 |
| process:logs/baseline/dclm-P1T.log | 1 | 0 |
| process:logs/baseline/dclm-P300.log | 1 | 0 |
| process:logs/baseline/dclm-S188.log | 1 | 0 |
| process:logs/baseline/dclm-S72.log | 1 | 0 |
| process:logs/baseline/fwe-P1T.log | 1 | 0 |
| process:logs/baseline/fwe-P300.log | 1 | 0 |
| process:logs/baseline/fwe-S188.log | 1 | 0 |
| process:logs/baseline/fwe-S72.log | 1 | 0 |
| process:logs/baseline/nem-P1T.log | 1 | 0 |
| process:logs/baseline/nem-P300.log | 1 | 0 |
| process:logs/baseline/nem-S188.log | 1 | 0 |
| process:logs/baseline/nem-S72.log | 1 | 0 |
| process:logs/cal1/dclm-P1T.log | 1 | 0 |
| process:logs/cal1/dclm-P300.log | 1 | 0 |
| process:logs/cal1/dclm-S188.log | 1 | 0 |
| process:logs/cal1/dclm-S72.log | 1 | 0 |
| process:logs/cal1/fwe-P1T.log | 1 | 0 |
| process:logs/cal1/fwe-P300.log | 1 | 0 |
| process:logs/cal1/fwe-S188.log | 1 | 0 |
| process:logs/cal1/fwe-S72.log | 1 | 0 |
| process:logs/downstream/dclm-P1T-quantized.log | 1 | 0 |
| process:logs/downstream/dclm-P300-quantized.log | 1 | 0 |
| process:logs/downstream/dclm-S188-parent.log | 1 | 0 |
| process:logs/downstream/dclm-S188-quantized.log | 1 | 0 |
| process:logs/downstream/dclm-S72-quantized.log | 1 | 0 |
| process:logs/downstream/fwe-P1T-quantized.log | 1 | 0 |
| process:logs/downstream/fwe-P300-quantized.log | 1 | 0 |
| process:logs/downstream/fwe-S188-parent.log | 1 | 0 |
| process:logs/downstream/fwe-S188-quantized.log | 1 | 0 |
| process:logs/downstream/fwe-S72-quantized.log | 1 | 0 |
| process:logs/gptq/dclm-P1T.log | 1 | 0 |
| process:logs/gptq/dclm-P300.log | 1 | 0 |
| process:logs/gptq/dclm-S188.log | 1 | 0 |
| process:logs/gptq/dclm-S72.log | 1 | 0 |
| process:logs/gptq/fwe-P1T.log | 1 | 0 |
| process:logs/gptq/fwe-P300.log | 1 | 0 |
| process:logs/gptq/fwe-S188.log | 1 | 0 |
| process:logs/gptq/fwe-S72.log | 1 | 0 |
| process:logs/gptq/nem-P1T.log | 1 | 0 |
| process:logs/gptq/nem-P300.log | 1 | 0 |
| process:logs/gptq/nem-S188.log | 1 | 0 |
| process:logs/gptq/nem-S72.log | 1 | 0 |
| process:logs/independent_replays/dclm-P1T-parent.log | 1 | 0 |
| process:logs/independent_replays/dclm-P1T-quantized.log | 1 | 0 |
| process:logs/independent_replays/dclm-P300-parent.log | 1 | 0 |
| process:logs/independent_replays/dclm-P300-quantized.log | 1 | 0 |
| process:logs/independent_replays/dclm-S188-parent.log | 1 | 0 |
| process:logs/independent_replays/dclm-S188-quantized.log | 1 | 0 |
| process:logs/independent_replays/dclm-S72-parent.log | 1 | 0 |
| process:logs/independent_replays/dclm-S72-quantized.log | 1 | 0 |
| process:logs/independent_replays/fwe-P1T-parent.log | 1 | 0 |
| process:logs/independent_replays/fwe-P1T-quantized.log | 1 | 0 |
| process:logs/independent_replays/fwe-P300-parent.log | 1 | 0 |
| process:logs/independent_replays/fwe-P300-quantized.log | 1 | 0 |
| process:logs/independent_replays/fwe-S188-parent.log | 1 | 0 |
| process:logs/independent_replays/fwe-S188-quantized.log | 1 | 0 |
| process:logs/independent_replays/fwe-S72-parent.log | 1 | 0 |
| process:logs/independent_replays/fwe-S72-quantized.log | 1 | 0 |
| process:logs/replays/dclm-P1T-quantized.log | 1 | 0 |
| process:logs/replays/dclm-P300-quantized.log | 1 | 0 |
| process:logs/replays/dclm-S188-parent.log | 1 | 0 |
| process:logs/replays/dclm-S188-quantized.log | 1 | 0 |
| process:logs/replays/dclm-S72-quantized.log | 1 | 0 |
| process:logs/replays/fwe-P1T-quantized.log | 1 | 0 |
| process:logs/replays/fwe-P300-quantized.log | 1 | 0 |
| process:logs/replays/fwe-S188-parent.log | 1 | 0 |
| process:logs/replays/fwe-S188-quantized.log | 1 | 0 |
| process:logs/replays/fwe-S72-quantized.log | 1 | 0 |
| process:logs/rtn/dclm-P1T.log | 1 | 0 |
| process:logs/rtn/dclm-P300.log | 1 | 0 |
| process:logs/rtn/dclm-S188.log | 1 | 0 |
| process:logs/rtn/dclm-S72.log | 1 | 0 |
| process:logs/rtn/fwe-P1T.log | 1 | 0 |
| process:logs/rtn/fwe-P300.log | 1 | 0 |
| process:logs/rtn/fwe-S188.log | 1 | 0 |
| process:logs/rtn/fwe-S72.log | 1 | 0 |
| process:logs/rtn/nem-P1T.log | 1 | 0 |
| process:logs/rtn/nem-P300.log | 1 | 0 |
| process:logs/rtn/nem-S188.log | 1 | 0 |
| process:logs/rtn/nem-S72.log | 1 | 0 |
| quantized_reload | 8 | 0 |
| registered_NLL_coverage | 1 | 0 |
| registered_downstream_coverage | 1 | 0 |
| representative_gate | 10 | 0 |
| rtn:dclm:P1T | 1 | 0 |
| rtn:dclm:P300 | 1 | 0 |
| rtn:dclm:S188 | 1 | 0 |
| rtn:dclm:S72 | 1 | 0 |
| rtn:fwe:P1T | 1 | 0 |
| rtn:fwe:P300 | 1 | 0 |
| rtn:fwe:S188 | 1 | 0 |
| rtn:fwe:S72 | 1 | 0 |
| rtn:nem:P1T | 1 | 0 |
| rtn:nem:P300 | 1 | 0 |
| rtn:nem:S188 | 1 | 0 |
| rtn:nem:S72 | 1 | 0 |
| score:dclm/P1T-quantized | 1 | 0 |
| score:dclm/P300-quantized | 1 | 0 |
| score:dclm/S188-parent | 1 | 0 |
| score:dclm/S188-quantized | 1 | 0 |
| score:dclm/S72-quantized | 1 | 0 |
| score:fwe/P1T-quantized | 1 | 0 |
| score:fwe/P300-quantized | 1 | 0 |
| score:fwe/S188-parent | 1 | 0 |
| score:fwe/S188-quantized | 1 | 0 |
| score:fwe/S72-quantized | 1 | 0 |
| spec_identity | 1 | 0 |
| twelve_parent_manifest | 1 | 0 |

Failures không bị đổi thành PASS. GPTQ Cholesky failure dùng RTN cho đúng matrix lỗi và flag model; các parent/method khác tiếp tục. Chi tiết theo matrix/model ở tests.json và worker logs.

## Downstream

Bit-width selection: `{"spec_sha256": "a7fb04042d54f977cff4ea5c2275b6b2af3a7c00cce4b2898ffc3dc9383c44da", "mean_damage": {"3": 0.034532251548411184, "4": 0.0065312104361074505}, "selected_bits": 3, "rule": "closest to 0.10 over all eight DCLM/FWE parents; exact tie -> lower bitwidth"}`.

### dclm

| Metric | Contrast | Estimate (pp) | 95% CI (pp) |
|---|---|---:|---|
| mc_mean | S72_drop | +0.741996991 | [+0.339932273, +1.139088745] |
| mc_mean | P300_drop | +0.949778545 | [+0.478924054, +1.413171784] |
| mc_mean | S188_drop | +0.251334689 | [-0.186911099, +0.683839748] |
| mc_mean | P1T_drop | +0.413160386 | [-0.048860092, +0.872171505] |
| mc_mean | L_ann | -0.536618159 | [-1.201208438, +0.120880126] |
| mc_mean | L_stab | -0.490662303 | [-1.076387819, +0.092887458] |
| mc_mean | R_short | +0.207781554 | [-0.413830808, +0.836680673] |
| lambada_accuracy | S72_drop | +2.561614593 | [+1.843586260, +3.299049098] |
| lambada_accuracy | P300_drop | +2.891519503 | [+2.154084999, +3.648360179] |
| lambada_accuracy | S188_drop | +2.503396080 | [+1.765961576, +3.260236755] |
| lambada_accuracy | P1T_drop | +2.755676305 | [+2.017756647, +3.531923152] |
| lambada_accuracy | L_ann | -0.135843198 | [-1.164370270, +0.931496216] |
| lambada_accuracy | L_stab | -0.058218513 | [-1.067339414, +0.950902387] |
| lambada_accuracy | R_short | +0.329904910 | [-0.679701145, +1.339025810] |

| Task | S72-parent | P300-parent | S188-parent | P1T-parent | S72-quantized | P300-quantized | S188-quantized | P1T-quantized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 34.129693 | 37.116041 | 33.617747 | 40.358362 | 33.020478 | 36.348123 | 33.276451 | 38.395904 |
| arc_easy | 61.237374 | 67.003367 | 61.447811 | 68.434343 | 60.353535 | 65.488215 | 60.563973 | 68.518519 |
| boolq | 63.088685 | 62.354740 | 64.434251 | 65.902141 | 62.262997 | 60.611621 | 64.434251 | 65.688073 |
| hellaswag | 58.076080 | 65.056762 | 58.763195 | 68.591914 | 57.000597 | 63.981279 | 57.916750 | 67.177853 |
| lambada_openai | 57.054143 | 65.476422 | 58.373763 | 67.649913 | 54.492529 | 62.584902 | 55.870367 | 64.894236 |
| openbookqa | 36.800000 | 39.200000 | 35.800000 | 39.200000 | 36.200000 | 38.400000 | 35.600000 | 39.400000 |
| piqa | 72.415669 | 75.680087 | 72.959739 | 76.550598 | 72.578890 | 74.972797 | 72.415669 | 75.625680 |
| sciq | 88.500000 | 90.800000 | 89.800000 | 93.000000 | 88.000000 | 90.600000 | 89.500000 | 92.900000 |
| winogrande | 59.273875 | 62.036306 | 58.247830 | 63.535912 | 58.168903 | 61.247040 | 59.352802 | 64.561957 |

### fwe

| Metric | Contrast | Estimate (pp) | 95% CI (pp) |
|---|---|---:|---|
| mc_mean | S72_drop | +0.468074413 | [+0.038847987, +0.902310242] |
| mc_mean | P300_drop | +1.442148380 | [+0.998051647, +1.885433525] |
| mc_mean | S188_drop | +0.749417897 | [+0.342600367, +1.150273850] |
| mc_mean | P1T_drop | +1.857060863 | [+1.380365077, +2.329756575] |
| mc_mean | L_ann | +0.414912483 | [-0.220656117, +1.061314372] |
| mc_mean | L_stab | +0.281343483 | [-0.318571531, +0.870973890] |
| mc_mean | R_short | +0.974073967 | [+0.347209953, +1.596645535] |
| lambada_accuracy | S72_drop | +1.843586260 | [+1.203182612, +2.483989909] |
| lambada_accuracy | P300_drop | +3.357267611 | [+2.619833107, +4.133514458] |
| lambada_accuracy | S188_drop | +2.425771395 | [+1.746555405, +3.104987386] |
| lambada_accuracy | P1T_drop | +4.967979818 | [+4.172326800, +5.783039006] |
| lambada_accuracy | L_ann | +1.610712206 | [+0.523966621, +2.678051620] |
| lambada_accuracy | L_stab | +0.582185135 | [-0.368717252, +1.533087522] |
| lambada_accuracy | R_short | +1.513681351 | [+0.523966621, +2.503396080] |

| Task | S72-parent | P300-parent | S188-parent | P1T-parent | S72-quantized | P300-quantized | S188-quantized | P1T-quantized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 37.542662 | 42.662116 | 35.750853 | 42.918089 | 37.457338 | 40.017065 | 36.092150 | 40.187713 |
| arc_easy | 66.666667 | 72.474747 | 63.425926 | 70.959596 | 65.025253 | 71.127946 | 62.289562 | 69.402357 |
| boolq | 59.266055 | 62.354740 | 61.559633 | 63.914373 | 60.000000 | 60.428135 | 61.100917 | 60.764526 |
| hellaswag | 54.381597 | 61.422027 | 55.247959 | 64.469229 | 53.505278 | 60.217088 | 54.650468 | 63.025294 |
| lambada_openai | 42.014361 | 51.775665 | 44.187852 | 54.317873 | 40.170774 | 48.418397 | 41.762080 | 49.349893 |
| openbookqa | 39.800000 | 41.800000 | 39.400000 | 42.400000 | 39.000000 | 41.600000 | 37.800000 | 41.600000 |
| piqa | 71.545158 | 75.136017 | 71.436344 | 75.244831 | 71.001088 | 74.211099 | 70.729053 | 74.211099 |
| sciq | 89.400000 | 93.200000 | 88.800000 | 93.200000 | 89.500000 | 92.200000 | 87.200000 | 91.900000 |
| winogrande | 57.458564 | 61.089187 | 57.063931 | 63.851618 | 56.827151 | 58.800316 | 56.827151 | 61.010260 |

## Own-corpus NLL (diagnostic only)

| Run | Mean NLL | Per-sequence artifact |
|---|---:|---|
| Q-dclm-P1T-parent | 2.452078286400 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/P1T/parent-own_corpus.json` |
| Q-dclm-P1T-pre-S1 | 3.744879166399 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/P1T/pre-S1-own_corpus.json` |
| Q-dclm-P300-parent | 2.506344140808 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/P300/parent-own_corpus.json` |
| Q-dclm-P300-pre-S1 | 3.555026044526 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/P300/pre-S1-own_corpus.json` |
| Q-dclm-S188-parent | 2.656850420321 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/S188/parent-own_corpus.json` |
| Q-dclm-S188-pre-S1 | 3.774380817661 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/S188/pre-S1-own_corpus.json` |
| Q-dclm-S72-parent | 2.677280175405 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/S72/parent-own_corpus.json` |
| Q-dclm-S72-pre-S1 | 3.618580657740 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/dclm/S72/pre-S1-own_corpus.json` |
| Q-fwe-P1T-parent | 2.256209722936 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/P1T/parent-own_corpus.json` |
| Q-fwe-P1T-pre-S1 | 3.628595580846 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/P1T/pre-S1-own_corpus.json` |
| Q-fwe-P300-parent | 2.308842448625 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/P300/parent-own_corpus.json` |
| Q-fwe-P300-pre-S1 | 3.313459735705 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/P300/pre-S1-own_corpus.json` |
| Q-fwe-S188-parent | 2.474318023053 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/S188/parent-own_corpus.json` |
| Q-fwe-S188-pre-S1 | 3.682665113220 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/S188/pre-S1-own_corpus.json` |
| Q-fwe-S72-parent | 2.489572923114 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/S72/parent-own_corpus.json` |
| Q-fwe-S72-pre-S1 | 3.475696347842 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/fwe/S72/pre-S1-own_corpus.json` |
| Q-nem-P1T-parent | 2.513666708896 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/P1T/parent-own_corpus.json` |
| Q-nem-P1T-pre-S1 | 3.859129403094 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/P1T/pre-S1-own_corpus.json` |
| Q-nem-P300-parent | 2.564980981101 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/P300/parent-own_corpus.json` |
| Q-nem-P300-pre-S1 | 3.624701690704 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/P300/pre-S1-own_corpus.json` |
| Q-nem-S188-parent | 2.703932035754 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/S188/parent-own_corpus.json` |
| Q-nem-S188-pre-S1 | 3.894054993124 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/S188/pre-S1-own_corpus.json` |
| Q-nem-S72-parent | 2.726567278888 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/S72/parent-own_corpus.json` |
| Q-nem-S72-pre-S1 | 3.581908400996 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/baseline/nem/S72/pre-S1-own_corpus.json` |
| Q-dclm-P1T-gptq3-cal1 | 2.506320198900 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/dclm/P1T/gptq3-cal1-own_corpus.json` |
| Q-dclm-P300-gptq3-cal1 | 2.545540430444 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/dclm/P300/gptq3-cal1-own_corpus.json` |
| Q-dclm-S188-gptq3-cal1 | 2.689660631459 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/dclm/S188/gptq3-cal1-own_corpus.json` |
| Q-dclm-S72-gptq3-cal1 | 2.708407059902 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/dclm/S72/gptq3-cal1-own_corpus.json` |
| Q-fwe-P1T-gptq3-cal1 | 2.301494866363 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/fwe/P1T/gptq3-cal1-own_corpus.json` |
| Q-fwe-P300-gptq3-cal1 | 2.344949916431 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/fwe/P300/gptq3-cal1-own_corpus.json` |
| Q-fwe-S188-gptq3-cal1 | 2.506400804345 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/fwe/S188/gptq3-cal1-own_corpus.json` |
| Q-fwe-S72-gptq3-cal1 | 2.516814918902 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/cal1/fwe/S72/gptq3-cal1-own_corpus.json` |
| Q-dclm-P1T-gptq4 | 2.461780393669 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/P1T/gptq4-own_corpus.json` |
| Q-dclm-P1T-gptq3 | 2.505405524092 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/P1T/gptq3-own_corpus.json` |
| Q-dclm-P300-gptq4 | 2.513603601711 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/P300/gptq4-own_corpus.json` |
| Q-dclm-P300-gptq3 | 2.545049747713 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/P300/gptq3-own_corpus.json` |
| Q-dclm-S188-gptq4 | 2.663330429120 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/S188/gptq4-own_corpus.json` |
| Q-dclm-S188-gptq3 | 2.689443025393 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/S188/gptq3-own_corpus.json` |
| Q-dclm-S72-gptq4 | 2.683276003361 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/S72/gptq4-own_corpus.json` |
| Q-dclm-S72-gptq3 | 2.708690945517 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/dclm/S72/gptq3-own_corpus.json` |
| Q-fwe-P1T-gptq4 | 2.265083299242 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/P1T/gptq4-own_corpus.json` |
| Q-fwe-P1T-gptq3 | 2.301690841677 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/P1T/gptq3-own_corpus.json` |
| Q-fwe-P300-gptq4 | 2.316616441846 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/P300/gptq4-own_corpus.json` |
| Q-fwe-P300-gptq3 | 2.345282039707 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/P300/gptq3-own_corpus.json` |
| Q-fwe-S188-gptq4 | 2.480116320201 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/S188/gptq4-own_corpus.json` |
| Q-fwe-S188-gptq3 | 2.506051567083 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/S188/gptq3-own_corpus.json` |
| Q-fwe-S72-gptq4 | 2.495016027250 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/S72/gptq4-own_corpus.json` |
| Q-fwe-S72-gptq3 | 2.516507273615 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/fwe/S72/gptq3-own_corpus.json` |
| Q-nem-P1T-gptq4 | 2.522386311283 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/P1T/gptq4-own_corpus.json` |
| Q-nem-P1T-gptq3 | 2.556887620872 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/P1T/gptq3-own_corpus.json` |
| Q-nem-P300-gptq4 | 2.572609546946 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/P300/gptq4-own_corpus.json` |
| Q-nem-P300-gptq3 | 2.601012286072 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/P300/gptq3-own_corpus.json` |
| Q-nem-S188-gptq4 | 2.709947834625 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/S188/gptq4-own_corpus.json` |
| Q-nem-S188-gptq3 | 2.732324765481 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/S188/gptq3-own_corpus.json` |
| Q-nem-S72-gptq4 | 2.732621140549 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/S72/gptq4-own_corpus.json` |
| Q-nem-S72-gptq3 | 2.753848859934 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/gptq/nem/S72/gptq3-own_corpus.json` |
| Q-dclm-P1T-rtn4 | 2.463309738901 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/P1T/rtn4-own_corpus.json` |
| Q-dclm-P1T-rtn3 | 2.520261692584 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/P1T/rtn3-own_corpus.json` |
| Q-dclm-P300-rtn4 | 2.517155620321 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/P300/rtn4-own_corpus.json` |
| Q-dclm-P300-rtn3 | 2.577815253551 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/P300/rtn3-own_corpus.json` |
| Q-dclm-S188-rtn4 | 2.664637261459 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/S188/rtn4-own_corpus.json` |
| Q-dclm-S188-rtn3 | 2.696517687200 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/S188/rtn3-own_corpus.json` |
| Q-dclm-S72-rtn4 | 2.685495062703 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/S72/rtn4-own_corpus.json` |
| Q-dclm-S72-rtn3 | 2.736557386035 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/dclm/S72/rtn3-own_corpus.json` |
| Q-fwe-P1T-rtn4 | 2.267294164962 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/P1T/rtn4-own_corpus.json` |
| Q-fwe-P1T-rtn3 | 2.311612434122 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/P1T/rtn3-own_corpus.json` |
| Q-fwe-P300-rtn4 | 2.318388304380 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/P300/rtn4-own_corpus.json` |
| Q-fwe-P300-rtn3 | 2.361591023277 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/P300/rtn3-own_corpus.json` |
| Q-fwe-S188-rtn4 | 2.482109825923 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/S188/rtn4-own_corpus.json` |
| Q-fwe-S188-rtn3 | 2.521801332673 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/S188/rtn3-own_corpus.json` |
| Q-fwe-S72-rtn4 | 2.496581067537 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/S72/rtn4-own_corpus.json` |
| Q-fwe-S72-rtn3 | 2.527525301334 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/fwe/S72/rtn3-own_corpus.json` |
| Q-nem-P1T-rtn4 | 2.525644473305 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/P1T/rtn4-own_corpus.json` |
| Q-nem-P1T-rtn3 | 2.573731474224 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/P1T/rtn3-own_corpus.json` |
| Q-nem-P300-rtn4 | 2.575194962006 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/P300/rtn4-own_corpus.json` |
| Q-nem-P300-rtn3 | 2.616314741545 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/P300/rtn3-own_corpus.json` |
| Q-nem-S188-rtn4 | 2.711332261603 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/S188/rtn4-own_corpus.json` |
| Q-nem-S188-rtn3 | 2.741509890573 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/S188/rtn3-own_corpus.json` |
| Q-nem-S72-rtn4 | 2.733826043871 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/S72/rtn4-own_corpus.json` |
| Q-nem-S72-rtn3 | 2.762555057315 | `/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01/results/a2_quant_20261031/workers/rtn/nem/S72/rtn3-own_corpus.json` |

## Bổ sung post hoc: pooled downstream DCLM + FineWeb-Edu

Phân tích được định nghĩa sau khi thấy kết quả từng dataset. Bootstrap 10.000 lần, seed 0; cùng lượt lấy mẫu item áp dụng cho cả hai dataset, giữ pairing giữa các model. Đơn vị bảng: điểm phần trăm (pp). CI chỉ phản ánh sampling benchmark item.

| Metric | Contrast | DCLM | FWE | Pooled | 95% CI | Tương quan draws |
|---|---|---:|---:|---:|---|---:|
| mc_mean | P300_drop | 3.472035404 | 2.757858272 | 3.114946838 | [2.680669771, 3.543912518] | 0.077349 |
| mc_mean | P1T_drop | 3.315287576 | 3.187083642 | 3.251185609 | [2.830996678, 3.672330266] | 0.038843 |
| mc_mean | X72_drop_from_S72 | 0.659318837 | -0.868594621 | -0.104637892 | [-0.575700089, 0.357834720] | -0.014275 |
| mc_mean | P1T_minus_P300_drop | -0.156747828 | 0.429225370 | 0.136238771 | [-0.417750278, 0.686094129] | 0.022334 |
| mc_mean | X72_minus_P300_accuracy | -0.403024228 | -0.633313744 | -0.518168986 | [-0.860637835, -0.181002262] | -0.021306 |
| mc_mean | X72_minus_P300_drop | -2.812716567 | -3.626452893 | -3.219584730 | [-3.777794579, -2.638681556] | 0.035375 |
| mc7_without_boolq | P300_drop | 2.919547671 | 3.086307226 | 3.002927448 | [2.531974455, 3.459722972] | 0.083833 |
| mc7_without_boolq | P1T_drop | 3.352028090 | 3.485107387 | 3.418567739 | [2.963731641, 3.880104117] | 0.045965 |
| mc7_without_boolq | X72_drop_from_S72 | -0.251298349 | -0.599494770 | -0.425396559 | [-0.930109196, 0.072702750] | 0.020727 |
| mc7_without_boolq | P1T_minus_P300_drop | 0.432480420 | 0.398800161 | 0.415640290 | [-0.190583147, 1.016291138] | 0.025676 |
| mc7_without_boolq | X72_minus_P300_accuracy | -0.609135596 | -0.741262015 | -0.675198806 | [-1.040536299, -0.318187370] | -0.029856 |
| mc7_without_boolq | X72_minus_P300_drop | -3.170846020 | -3.685801996 | -3.428324008 | [-4.038018914, -2.813584641] | 0.045271 |
| lambada_accuracy | P300_drop | 4.405200854 | 4.327576169 | 4.366388512 | [3.716281778, 5.035901417] | 0.003420 |
| lambada_accuracy | P1T_drop | 5.297884727 | 5.453134097 | 5.375509412 | [4.676887250, 6.074131574] | 0.019134 |
| lambada_accuracy | X72_drop_from_S72 | -3.842421890 | -4.599262565 | -4.220842228 | [-5.006792160, -3.444595381] | 0.013609 |
| lambada_accuracy | P1T_minus_P300_drop | 0.892683873 | 1.125557927 | 1.009120900 | [0.116437027, 1.862992432] | -0.001075 |
| lambada_accuracy | X72_minus_P300_accuracy | -0.174655540 | -0.834465360 | -0.504560450 | [-1.048175820, 0.058218513] | 0.015352 |
| lambada_accuracy | X72_minus_P300_drop | -8.247622744 | -8.926838735 | -8.587230739 | [-9.547836212, -7.655734524] | 0.013262 |

MC7 bỏ BoolQ là sensitivity báo cạnh MC8, không thay thế kết quả chính; lý do preregistration: parents ở mức majority-class khoảng 62.2%.
Đối chiếu point estimates với các bảng đã công bố: True.

## Bổ sung T17 (text only)

T17 Stage1 = 3.0618885176067017 và Stage1b = 3.0617311959690636 nats; Stage1b − Stage1 = −0.0001573216376380 nats. Các phiên bản được ghi nhận (Stage1 / Stage1b) khớp nhau: torch 2.14.1+cu130 / 2.14.1+cu130, CUDA build tag cu130 / cu130, transformers 4.57.1 / 4.57.1, accelerate 1.13.0 / 1.13.0, Python 3.11.15 / 3.11.15, NumPy 2.4.6 / 2.4.6, safetensors 0.8.0 / 0.8.0. Hai lần dùng cùng tám UUID GPU B200. Exact loaded CUDA runtime, cuDNN, NCCL, driver package và SDPA backend thực tế không được ghi đầy đủ cho từng T17, nên không khẳng định chúng giống nhau. Nguyên nhân: chưa xác định (not identified). Không label nào phụ thuộc sai khác này. Nguồn: docs/A2_STAGE1B_FOLLOWUP_REPLY_20261030.md; đây là đối chiếu text-only, không chạy thêm T17.

## Provenance, ledger và phần còn thiếu

Parent/calibration/native implementation: manifests/. Per-sequence NLL: workers/. Raw scores, primary replay và fresh independent replay có SHA trong downstream.json. runs.jsonl là index của 160 file NLL dự kiến.

Ước lượng preregistration: khoảng 2 node-hours; đây không phải deadline. Ledger đo được trong complete.json chỉ bao gồm coordinator; thời gian chuẩn bị/handoff bổ sung từ supervisor receipt khi xác minh terminal.

Coverage complete: True. Coverage đầy đủ không biến diagnostic FAIL thành PASS. Các flag/unverified vẫn được giữ.

