"""Render the registered tables from saved receipts; never invent absent cells."""
import json
from pathlib import Path
from collections import Counter

T17_NOTE = ('T17 Stage1 = 3.0618885176067017 và Stage1b = 3.0617311959690636 nats; '
    'Stage1b − Stage1 = −0.0001573216376380 nats. Các phiên bản được ghi nhận '
    '(Stage1 / Stage1b) khớp nhau: torch 2.14.1+cu130 / 2.14.1+cu130, CUDA build tag '
    'cu130 / cu130, transformers 4.57.1 / 4.57.1, accelerate 1.13.0 / 1.13.0, '
    'Python 3.11.15 / 3.11.15, NumPy 2.4.6 / 2.4.6, safetensors 0.8.0 / 0.8.0. '
    'Hai lần dùng cùng tám UUID GPU B200. Exact loaded CUDA runtime, cuDNN, NCCL, '
    'driver package và SDPA backend thực tế không được ghi đầy đủ cho từng T17, '
    'nên không khẳng định chúng giống nhau. Nguyên nhân: chưa xác định (not identified). '
    'Không label nào phụ thuộc sai khác này. Nguồn: docs/A2_STAGE1B_FOLLOWUP_REPLY_20261030.md; '
    'đây là đối chiếu text-only, không chạy thêm T17.')


def read(root,name,default=None):
    p=Path(root)/name
    return json.loads(p.read_text()) if p.exists() else default


def pooled_markdown(result):
    lines=['## Bổ sung post hoc: pooled downstream DCLM + FineWeb-Edu','',
        'Phân tích được định nghĩa sau khi thấy kết quả từng dataset. Bootstrap 10.000 lần, seed 0; '
        'cùng lượt lấy mẫu item áp dụng cho cả hai dataset, giữ pairing giữa các model. '
        'Đơn vị bảng: điểm phần trăm (pp). CI chỉ phản ánh sampling benchmark item.','',
        '| Metric | Contrast | DCLM | FWE | Pooled | 95% CI | Tương quan draws |',
        '|---|---|---:|---:|---:|---|---:|']
    for metric,rows in result['metrics'].items():
        for name,r in rows.items():
            corr='undefined (zero variance)' if r['correlation'] is None else f"{r['correlation']:.6f}"
            lines.append(f"| {metric} | {name} | {100*r['dclm']:.9f} | {100*r['fwe']:.9f} | "
                         f"{100*r['pooled']:.9f} | [{100*r['ci95'][0]:.9f}, {100*r['ci95'][1]:.9f}] | {corr} |")
    lines+=['','MC7 bỏ BoolQ là sensitivity báo cạnh MC8, không thay thế kết quả chính; '
             'lý do preregistration: parents ở mức majority-class khoảng 62.2%.',
             'Đối chiếu point estimates với các bảng đã công bố: '+str(result['passed'])+'.','',
             '## Bổ sung T17 (text only)','',T17_NOTE,'']
    return '\n'.join(lines)


def render(root,cfg):
    root=Path(root);decision=read(root,'decision_quant.json',{});done=read(root,'complete.json',{})
    tests=read(root,'tests.json',{});cal=read(root,'manifests/calibration.json',{})
    lines=['# A2 Q — quantization trên cùng 12 parents','',
        'Spec SHA256: `'+cfg['spec_sha256']+'`. Frozen source: `'+cfg['code_commit']+'`.','',
        'Trạng thái thực thi: '+json.dumps(done,ensure_ascii=False)+'.','',
        'Final CulturaX là decision set; own-corpus chỉ diagnostic. Trọng số dequantized được '
        'làm tròn BF16, giữ trong tensor FP32; forward BF16 autocast và CE FP32. '
        'Mọi NLL/CI xuất phát từ file per-sequence, batch validation 1, downstream batch 2. '
        'Không healing, không thêm architecture/LR/exposure test.','',
        '## Damage và contrasts','',
        'Nats: compressed NLL − parent NLL. C/D(R) là relative contrast; D/val(parent) '
        'là relative damage và được ghi đầy đủ trong decision_quant.json. '
        'Bootstrap 10.000 lần, seed 0, giữ cùng draws cho mọi model trong dataset. '
        'R_long so sánh 1T với 776B nên chỉ descriptive; I cũng chỉ descriptive.','']
    for ds,table in decision.get('datasets',{}).items():
        lines += ['### '+ds,'','| Compression | L_ann: nats / C÷D(R) / label | L_stab | R_short | R_long |',
                  '|---|---|---|---|---|']
        for compression in ('pre-S1','rtn4','rtn3','gptq4','gptq3'):
            cells=[]
            for name in ('L_ann','L_stab','R_short','R_long'):
                v=table.get(compression,{}).get('contrasts',{}).get(name)
                if v is None:cells.append('MISSING');continue
                rel='undefined' if v['relative_to_reference_damage'] is None else f"{v['relative_to_reference_damage']:.6f}"
                cells.append(f"{v['estimate']:+.9f} / {rel} / {v['label']}; CI [{v['ci95'][0]:+.9f}, {v['ci95'][1]:+.9f}]")
            lines.append('| '+compression+' | '+' | '.join(cells)+' |')
        if ds in ('dclm','fwe'):
            e,a=(.021670884569823023,.01211313187395735) if ds=='dclm' else (.023461150403907727,.01299365118963669)
            lines+=['',f'Healed pruning 3.2B: E={e:+.15f}; A_init={a:+.15f} nats. '
                    'A_init so sánh hai student, không phải hai damage.']
        lines+=['','| Parent | Compression | Parent NLL | Compressed NLL | D | D/parent |',
                '|---|---|---:|---:|---:|---:|']
        for compression,row in table.items():
            for parent,v in row['damage'].items():
                lines.append(f"| {parent} | {compression} | {v['parent_nll']:.12f} | {v['compressed_nll']:.12f} | {v['damage']:+.12f} | {v['relative_damage']:+.9f} |")
        lines+=['']
    lines+=['## Labels, calibration và kiểm tra','',
        'T=max(0.003,0.10D(R)); t=max(0.003,0.03D(R)). Thứ tự: more damage, less damage, '
        'no effect, grey. Mọi clause boolean, I, cross-dataset agreement và cal0/cal1 flag '
        'được ghi trong decision_quant.json.','',
        'QT6 calibration disjointness: '+str(cal.get('passed'))+'. Overlaps: `'+json.dumps(cal.get('overlaps',{}))+'`.','',
        'Nếu hai prefix calibration chạm cùng document ở ranh giới đóng gói, giữ đúng '
        'first128/next128 theo spec, ghi QT6 FAIL và đánh dấu mọi kết quả phụ thuộc calibration là unverified. '
        'Không thay prefix âm thầm.','', '| Check | PASS | FLAG |','|---|---:|---:|']
    counts={}
    for group in tests.values():
        if not isinstance(group,list):continue
        for row in group:
            name=row.get('test',row.get('check','unknown'));counts.setdefault(name,Counter())[bool(row['passed'])]+=1
    for name,c in sorted(counts.items()):lines.append(f'| {name} | {c[True]} | {c[False]} |')
    lines+=['','Failures không bị đổi thành PASS. GPTQ Cholesky failure dùng RTN cho đúng matrix lỗi '
        'và flag model; các parent/method khác tiếp tục. Chi tiết theo matrix/model ở tests.json và worker logs.','',
        '## Downstream','', 'Bit-width selection: `'+json.dumps(read(root,'bitwidth.json',{}))+'`.','']
    downstream=read(root,'downstream_paired.json',{})
    for ds,rows in downstream.get('datasets',{}).items():
        lines+=['### '+ds,'','| Metric | Contrast | Estimate (pp) | 95% CI (pp) |','|---|---|---:|---|']
        for metric in ('mc_mean','lambada_accuracy'):
            for name,v in rows[metric].items():
                lines.append(f"| {metric} | {name} | {100*v['estimate']:+.9f} | [{100*v['ci95'][0]:+.9f}, {100*v['ci95'][1]:+.9f}] |")
        lines+=['']
        model_names=list(next(iter(rows['tasks'].values()))['accuracy'])
        lines+=['| Task | '+' | '.join(model_names)+' |','|---|'+'---:|'*len(model_names)]
        for task,row in rows['tasks'].items():
            lines.append('| '+task+' | '+' | '.join(f"{100*row['accuracy'][m]:.6f}" for m in model_names)+' |')
        lines+=['']
    lines+=['## Own-corpus NLL (diagnostic only)','',
        '| Run | Mean NLL | Per-sequence artifact |','|---|---:|---|']
    index=root/'runs.jsonl'
    if index.exists():
        for line in index.read_text().splitlines():
            row=json.loads(line)
            if row['set']=='own_corpus':
                lines.append(f"| {row['run_id']} | {row['nll']:.12f} | `{row['per_sequence_file']['path']}` |")
    lines+=['']
    pooled=read(root,'pooled_downstream.json')
    if pooled:lines.append(pooled_markdown(pooled))
    else:lines+=['## Bổ sung CPU','', 'Pooled downstream chưa có artifact hợp lệ; xem checks.','',T17_NOTE,'']
    lines+=['## Provenance, ledger và phần còn thiếu','',
        'Parent/calibration/native implementation: manifests/. Per-sequence NLL: workers/. '
        'Raw scores, primary replay và fresh independent replay có SHA trong downstream.json. '
        'runs.jsonl là index của 160 file NLL dự kiến.','',
        'Ước lượng preregistration: khoảng 2 node-hours; đây không phải deadline. '
        'Ledger đo được trong complete.json chỉ bao gồm coordinator; thời gian chuẩn bị/handoff '
        'bổ sung từ supervisor receipt khi xác minh terminal.','',
        'Coverage complete: '+str(done.get('coverage_complete'))+'. '
        'Coverage đầy đủ không biến diagnostic FAIL thành PASS. Các flag/unverified vẫn được giữ.', '']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
