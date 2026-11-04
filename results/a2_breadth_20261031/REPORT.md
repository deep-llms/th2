# A2 Breadth / stable-state healing
Spec SHA256: `dd24ef489e2fb3cb4f6e16fd7db0df4b412e96679786315cf44bb3229c88fd0b`

Coverage complete: True. Technical/scientific checks: 603/603 PASS. Independent local CPU replay: PASS. Preregistered predictions: SM1, SM2, SP1 supported; SM3 not supported. Prediction outcomes are distinct from implementation checks.


## Kết quả chính và trạng thái cuối cùng

Đã hoàn tất 4 lượt heal và 44 cấu hình SmolLM3 (11 checkpoint × parent/3 block), lưu đủ 56 file NLL. Đánh giá SmolLM3 gồm 11.717 tài liệu, 8.821.351 token được dự đoán; mỗi tài liệu là một sequence riêng.

- **SM1 đạt:** B_S1 tăng damage **7,646%** giữa cặp 283B → 755B; CI 95% bootstrap theo cặp tài liệu **[7,378%; 7,973%]** (10.000 resample, seed 0).
- **SM2 đạt:** Spearman giữa log tokens và B_S1 damage trên 9 checkpoint stage1 là **1,0**.
- **SM3 không đạt:** B_early không tăng tương đối mạnh hơn B_S1 từ checkpoint đầu đến cuối stage1.
- **SP1 đạt ở cả hai dataset:** E_stab = **0,0111319744** (DCLM; **7,027%**) và **0,0147807444** (FW-EDU; **10,055%**), đều ≥ 0,008.

Đây là bằng chứng về ảnh hưởng thời lượng huấn luyện ở trạng thái stable trong một họ model thứ hai, cùng penalty sau healing ở hai dataset. Không diễn giải cặp stage2 → stage3 thành hiệu ứng annealing thuần túy: data mix cũng đổi. Các giá trị damage lớn của checkpoint cuối vẫn giữ nguyên trong bảng và JSON.

## SmolLM3 pruning damage

| Revision | Tokens | Parent NLL | B_early D | B_S1 D | B_end D |
|---|---:|---:|---:|---:|---:|
| stage1-step-40000 | 94371840000 | 2.854040833 | 8.490563700 | 0.862828185 | 2.738050643 |
| stage1-step-80000 | 188743680000 | 2.789947857 | 7.458192883 | 0.959114163 | 3.304393276 |
| stage1-step-120000 | 283115520000 | 2.761608417 | 7.003091704 | 0.984422885 | 3.535325481 |
| stage1-step-160000 | 377487360000 | 2.746321298 | 8.906022942 | 1.003983743 | 3.875203154 |
| stage1-step-320000 | 754974720000 | 2.712024100 | 6.196922574 | 1.059695416 | 3.806817854 |
| stage1-step-640000 | 1509949440000 | 2.686776794 | 6.226735959 | 1.097666853 | 10.498298487 |
| stage1-step-1280000 | 3019898880000 | 2.666146954 | 3.232600436 | 1.218820865 | 9.583291785 |
| stage1-step-2560000 | 6039797760000 | 2.650000282 | 3.063048733 | 2.040374105 | 13.291351634 |
| stage1-step-3440000 | 8115978240000 | 2.646539841 | 3.894191991 | 3.576476490 | 16.111062330 |
| stage2-step-4200000 | 9909043200000 | 2.651628023 | 3.554741637 | 5.547769185 | 17.508578223 |
| stage3-step-4720000 | 11135877120000 | 2.550939193 | 3.369186604 | 41.800051274 | 25.780313599 |

## Matched 283B → 755B pair

| Block | Relative change | Paired 95% CI | Spearman, nine stage1 points |
|---|---:|---|---:|
| B_early | -0.11511617492196784 | [-0.1178116818359194, -0.11250528833552918] | -0.8333333333333333 |
| B_S1 | 0.07646361297284264 | [0.07378225762381461, 0.0797257828414644] | 1.0 |
| B_end | 0.0767941663091935 | [0.07505207151843943, 0.07861392337161893] | 0.9666666666666667 |

First-to-last relative changes and descriptive stage2→stage3 pair (confounded data mix):

```json
{
  "relative_to_first": {
    "stage1-step-40000": {
      "B_early": 0.0,
      "B_S1": 0.0,
      "B_end": 0.0
    },
    "stage1-step-80000": {
      "B_early": -0.12159037408900486,
      "B_S1": 0.11159345439121782,
      "B_end": 0.2068415480706307
    },
    "stage1-step-120000": {
      "B_early": -0.17519119447886766,
      "B_S1": 0.1409257390022678,
      "B_end": 0.2911833791247316
    },
    "stage1-step-160000": {
      "B_early": 0.048931879745071516,
      "B_S1": 0.16359636787071863,
      "B_end": 0.41531463781607925
    },
    "stage1-step-320000": {
      "B_early": -0.27014002921241775,
      "B_S1": 0.22816504314009167,
      "B_end": 0.3903387302769028
    },
    "stage1-step-640000": {
      "B_early": -0.26662867400949375,
      "B_S1": 0.272173153338817,
      "B_end": 2.834223634866448
    },
    "stage1-step-1280000": {
      "B_early": -0.6192713993753691,
      "B_S1": 0.41258814374890185,
      "B_end": 2.5000418311208805
    },
    "stage1-step-2560000": {
      "B_early": -0.6392408276218409,
      "B_S1": 1.3647513366869073,
      "B_end": 3.854311833281714
    },
    "stage1-step-3440000": {
      "B_early": -0.5413505947503354,
      "B_S1": 3.14506219219676,
      "B_end": 4.884135990371224
    },
    "stage2-step-4200000": null,
    "stage3-step-4720000": null
  },
  "end_pair": {
    "B_early": {
      "difference": -0.18555503297899767,
      "relative": -0.0521993022021959,
      "confounded_data_mix": true
    },
    "B_S1": {
      "difference": 36.25228208881151,
      "relative": 6.534569280276742,
      "confounded_data_mix": true
    },
    "B_end": {
      "difference": 8.271735376653794,
      "relative": 0.47243901083108336,
      "confounded_data_mix": true
    }
  }
}
```

## Stable-state heals

| Dataset | S72 Δpost | S188 Δpost | E_stab | E_stab / S72 | E_ann | E_ann / P300 | P300 − S72 Δpost |
|---|---:|---:|---:|---:|---:|---:|---:|
| dclm | 0.158411477 | 0.169543452 | 0.011131974 | 0.070272524 | 0.021096293 | 0.103028841 | 0.046349569 |
| fwe | 0.146999744 | 0.161780488 | 0.014780744 | 0.100549457 | 0.026224222 | 0.133987100 | 0.048722245 |

## Preregistered clauses

{"SM1": true, "SM2": true, "SM3": false, "SP1": true}

## Deviations and limitations

- Depth A-B prerequisite/spec absent from workspace; execute authorized Breadth now; Depth C cannot be reconstructed.
- Separate-document scoring uses token-weighted NLL and document-paired resampling (10,000, seed0); no BOS/template, explicit EOS, truncate4096.
- The stage3 end contrast changes data mix and is descriptive only.
- Scientific checks do not stop registered work; failed checks remain in tests.json and decision flags.

Missing results: [].

## Compute ledger

```json
{
  "elapsed_seconds": 17024.501652726904,
  "download_wait_seconds": 10530.816530371085,
  "estimated_node_hours": 2,
  "policy": "report-only",
  "events": [
    {
      "name": "ST1-dclm-S72-S1-0.2B-lr1e-3",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/ST1-dclm-S72-S1-0.2B-lr1e-3.log"
    },
    {
      "name": "ST1-dclm-S188-S1-0.2B-lr1e-3",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/ST1-dclm-S188-S1-0.2B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S72-S1-0.2B-lr1e-3",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/ST1-fwe-S72-S1-0.2B-lr1e-3.log"
    },
    {
      "name": "ST1-fwe-S188-S1-0.2B-lr1e-3",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/ST1-fwe-S188-S1-0.2B-lr1e-3.log"
    },
    {
      "name": "prepare_documents",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/prepare_documents.log"
    },
    {
      "name": "stage1-step-40000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-40000.log"
    },
    {
      "name": "stage1-step-3440000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-3440000.log"
    },
    {
      "name": "stage1-step-80000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-80000.log"
    },
    {
      "name": "stage2-step-4200000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage2-step-4200000.log"
    },
    {
      "name": "stage1-step-120000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-120000.log"
    },
    {
      "name": "stage3-step-4720000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage3-step-4720000.log"
    },
    {
      "name": "stage1-step-160000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-160000.log"
    },
    {
      "name": "stage1-step-320000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-320000.log"
    },
    {
      "name": "stage1-step-640000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-640000.log"
    },
    {
      "name": "stage1-step-1280000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-1280000.log"
    },
    {
      "name": "stage1-step-2560000",
      "returncode": 0,
      "log": "/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01/results/a2_breadth_20261031/logs/stage1-step-2560000.log"
    }
  ],
  "spec_sha256": "dd24ef489e2fb3cb4f6e16fd7db0df4b412e96679786315cf44bb3229c88fd0b"
}
```

Expected 2 node-hours plus download (estimate only). Actual execution, scientific coverage and independent native handoff are separate receipts.

Breadth registered work is complete, including independent saved-array verification and native GPU handoff. Depth C is not run because `docs/A2_depth_spec.md` is absent; no Depth result is claimed.

## Independent verification, provenance and retrieval

- Execution source: `60e5461cbbf21ae437149a45923371cbc6dd74a2`; all 158 frozen source files matched both node hashes and the recorded Git tree.
- Launch: 31 October 2026, 21:49:30.998 SGT. Coordinator/readout terminal: 1 November, 02:33:22.215 SGT. Supervisor exit 0 and native burn handoff: 02:34:28.106 SGT.
- Independent terminal observation: 07:11:52–07:12:04 SGT. Eight burn workers 753407–753414 under launcher 753338 had matching PID/start/command identities; collective cycles advanced 19120 → 19130, payload 21229.92 → 21241.03 GiB. Guard released; experiment session ended. See [terminal verification](terminal_0712/verification.json).
- Health receipt SHA256: `d3fe08902aa1b1e0f35194373fe5239c21eb377d4e078cfe9f3848787e2a8abe`. Earlier runner DNS/SSH failures are preserved in [transport failure record](health_a02_transport_failure/receipt.json); later successful delivery of the same health command established recovery, without an experiment restart.
- Export commit: `a98de30119c3e0856b818249e513174403f1649f`; controller acknowledged 2 files pulled, 0 skipped. The archive contains **159 original files**, **7,772,377 compressed bytes**, SHA256 `9821604f83b263cf01008844b632e61d4bba1f5801e25581d9f1724ae720a774`. All member sizes/hashes were verified; all 56 per-sequence hashes also match the separate health receipt.
- [Independent CPU replay](final/independent_cpu_replay.json.gz) uses separate NumPy formulas, without importing the production readout: 56 means; all checkpoint damages/relative changes; three fresh paired bootstrap intervals (10,000, seed 0); correlations; stable/annealed contrasts; all four prediction clauses; rank checks/LR/order coverage. All comparisons passed; maximum absolute discrepancy **1.1368683772161603e-13** (tolerance1e-12 for statistical quantities).
- Original node report and all exported arrays/JSON are preserved locally in `artifacts/results/a2_breadth_20261031/` under this results directory. These bulk originals are Git-ignored to respect the runner source-size limit; they must be copied separately for a portable backup. [Archive member manifest](final/export_manifest.json) and [download verification](final/download_verification.json) record their paths and checksums. No model weights were included in this small result export; original B200 checkpoints remain in place.
- Rerun the CPU audit from the project root with `OPENBLAS_NUM_THREADS=1 python results/a2_breadth_20261031/final/replay_saved_arrays.py results/a2_breadth_20261031/artifacts/results/a2_breadth_20261031 temp/breadth_replay.json`.

## Measured compute and remaining scope

Coordinator time before readout was **4.7290 node-hours**, including **2.9252 hours waiting for downloads**; nonwaiting time was **1.8038 hours**. Full coordinator/readout execution was **4.7307 hours**, and launch-to-native-handoff was **4.7492 hours**. These figures exclude background download time before launch and subsequent local verification; none is a measured GPU-utilization integral. The preregistered estimate was approximately 2 hours plus downloads; all time policies are report-only.

No remaining registered Breadth computation, export, replay, or GPU-handoff step. The ordering deviation remains: Depth A–B evidence/spec was absent when Breadth was authorized, and Depth C/DT6 cannot be reconstructed from the available specification. No additional experiment was launched.
