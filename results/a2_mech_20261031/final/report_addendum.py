"""Render the owner-requested descriptive addendum from locked saved JSON; CPU only."""
import hashlib,json
from pathlib import Path
import numpy as np

base=Path('results/a2_mech_20261031')
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=load(base/'decision_mech.json');b=load(base/'buckets.json');n=load(base/'noise_paired.json');layers=load(base/'layers.json')
qpath=Path('results/a2_quant_20261031/batches/b01/decision_quant.json');q=load(qpath)
assert sha(qpath)==load(base/'config/plan.json')['quant_decision']['sha256']
manifest=load(base/'artifacts/manifest.json')
for name in ('decision_mech.json','buckets.json','noise_paired.json','layers.json'):
 assert sha(base/name)==manifest['files']['results/a2_mech_20261031/readout_frequency_revision/'+name]['sha256']
dslist=('dclm','fwe','nem');parents=('S72','P300','S188','P1T');cnames=('L_ann','L_stab','R_short')
metric_names=list(d['metrics']);assert len(metric_names)==83 and set(metric_names)==set(d['correlations_dataset_centered'])
floatfmt=lambda x:'NA' if x is None else f'{x:.6f}'
pct=lambda x:'NA' if x is None else f'{100*x:+.3f}%'
lines=['<!-- A2_M_ADDENDUM_START -->','## Phụ lục theo yêu cầu — bảng đầy đủ từ JSON đã lưu','',
 'Bổ sung bằng CPU từ các kết quả canonical đã khóa SHA; không chạy lại model, quantization hoặc GPU. Các điểm noise là trung bình seed 0 và 1. Damage/gap/A_init có đơn vị nats/token; retention và correlation là số không thứ nguyên. Giá trị bảng làm tròn để đọc; JSON giữ đầy đủ độ chính xác.','',
 '`L_ann = (P1T − P300) / P300`, `L_stab = (S188 − S72) / S72`, `R_short = (P300 − S72) / S72`, áp dụng riêng cho metric đang xét. Các relative contrasts hiển thị dưới dạng %. Mẫu số là giá trị metric của parent tham chiếu, không phải NLL parent.','',
 '### 1. N-rtn3: damage từng parent và tỷ số so với RTN-3','',
 '| Dataset | Parent | N-rtn3 damage | RTN-3 damage | N-rtn3 / RTN-3 |','|---|---|---|---|---|']
for ds in dslist:
 for p in parents:
  value=d['metrics']['N-rtn3']['values'][ds][p];ref=q['datasets'][ds]['rtn3']['damage'][p]['damage'];ratio=d['N_rtn3_over_rtn3'][ds+'/'+p]
  assert np.isclose(value,n['datasets'][ds]['N-rtn3']['damage'][p]['damage'],rtol=0,atol=1e-12)
  assert np.isclose(value/ref,ratio,rtol=0,atol=1e-12)
  lines.append(f'| {ds} | {p} | {floatfmt(value)} | {floatfmt(ref)} | {floatfmt(ratio)} |')
def contrasts(metric):
 lines.extend(['','| Dataset | L_ann | L_stab | R_short |','|---|---|---|---|'])
 for ds in dslist:
  t=d['metrics'][metric]['relative_contrasts'][ds]
  lines.append('| '+ds+' | '+' | '.join(pct(t[c]['relative']) for c in cnames)+' |')
 lines.append('')
lines.extend(['','**Relative contrasts của N-rtn3 damage:**']);contrasts('N-rtn3')
lines.extend(['### 2. N-iso(0.1) và N-iso(0.3)',''])
for metric in ('N-iso-0.1','N-iso-0.3'):
 lines.append('**'+metric+'**');contrasts(metric)
lines.extend(['Các ordering flips giữa hai seed được giữ dưới đây; `đổi` nghĩa là mệnh đề có truth value khác giữa seed 0 và 1, không phải điểm trung bình không xác định.','',
 '| Probe | Dataset | Mệnh đề đổi giữa seed 0 và 1 |','|---|---|---|'])
for metric in ('N-rtn3','N-iso-0.1','N-iso-0.3'):
 for ds in dslist:
  flips=[k for k,v in d['noise_seed_readout'][metric]['ordering_flips'][ds].items() if v]
  lines.append('| '+metric+' | '+ds+' | '+(', '.join(flips) if flips else 'Không')+' |')
lines.extend(['','### 3. Correlation sau khi trừ mean từng dataset','',
 'Với mỗi metric và mỗi loại damage, trừ mean riêng của bốn parent trong từng dataset, rồi tính Pearson r trên 12 residual (3 dataset × 4 parent). Không có p-value hoặc suy luận nhân quả. GPTQ-3 dùng cal0 theo JSON đã đăng ký; cờ calibration sensitivity của A2 Q vẫn áp dụng.','',
 '| Metric | r với pre-S1 damage | r với rtn3 damage | r với gptq3 damage |','|---|---|---|---|'])
max_error=0
for metric in metric_names:
 saved=d['correlations_dataset_centered'][metric]
 for compression in ('pre-S1','rtn3','gptq3'):
  x=np.array([[d['metrics'][metric]['values'][ds][p] for p in parents] for ds in dslist]);y=np.array([[q['datasets'][ds][compression]['damage'][p]['damage'] for p in parents] for ds in dslist])
  x=x-x.mean(axis=1,keepdims=True);y=y-y.mean(axis=1,keepdims=True)
  value=float(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y)))
  assert np.isclose(value,saved[compression],atol=1e-12,rtol=0),(metric,compression)
  max_error=max(max_error,abs(value-saved[compression]))
 lines.append('| '+metric+' | '+' | '.join(floatfmt(saved[c]) for c in ('pre-S1','rtn3','gptq3'))+' |')
lines.extend(['','### 4. Profile single-layer bypass: đủ 24 lớp','',
 'Chỉ số lớp bắt đầu từ 0; S1 là lớp 16–21. Bảng dùng damage của việc bypass riêng từng lớp trên small set. Các giá trị relative có thể lớn khi damage parent tham chiếu nhỏ; không cộng hoặc lấy trung bình các tỷ lệ này để suy ra damage của cả block.',''])
for ds in dslist:
 lines.extend(['**'+ds+'**','','| Lớp | Thuộc S1 (16–21) | L_ann | L_stab | R_short |','|---|---|---|---|---|'])
 for i in range(24):
  t=d['metrics'][f'single_damage_{i}']['relative_contrasts'][ds]
  for c,pa,pb in [('L_ann','P1T','P300'),('L_stab','S188','S72'),('R_short','P300','S72')]:
   va=layers['parents'][ds+'/'+pa]['single_damage'][i];vb=layers['parents'][ds+'/'+pb]['single_damage'][i]
   assert np.isclose((va-vb)/vb,t[c]['relative'],atol=1e-12,rtol=0)
  lines.append(f'| {i} | '+('Có' if 16<=i<=21 else '—')+' | '+' | '.join(pct(t[c]['relative']) for c in cnames)+' |')
 lines.append('')
lines.extend(['### 5. Đủ 10 bucket của mỗi loại, ở DCLM và FWE','',
 'Bucket đánh số 0–9. Frequency 0 → 9 là hiếm → phổ biến; difficulty 0 → 9 là dễ → khó. `g_par = P300 − P1T`; `g_stu = student P300 − student P1T`; `retention = g_stu/g_par`; `A_init = student X72 − student P300`. Cờ mẫu số nhỏ dùng đúng |g_par| < 0.005.',''])
for ds in ('dclm','fwe'):
 for kind in ('frequency','difficulty'):
  rows=b['datasets'][ds]['buckets'][kind];assert len(rows)==10
  lines.extend(['**'+ds+' / '+kind+'**','','| Bucket | g_par | g_stu | Retention | A_init | Cờ |','|---|---|---|---|---|---|'])
  for row in rows:
   means=row['means'];assert np.isclose(means['P300']-means['P1T'],row['parent_gap'],atol=1e-12,rtol=0)
   assert np.isclose(means['student-P300']-means['student-P1T'],row['student_gap'],atol=1e-12,rtol=0)
   assert np.isclose(means['student-X72']-means['student-P300'],row['A_init'],atol=1e-12,rtol=0)
   assert np.isclose(row['student_gap']/row['parent_gap'],row['retention'],atol=1e-12,rtol=0)
   flags=[]
   if row['low_denominator']:flags.append('|g_par| < 0.005')
   if row.get('bootstrap_denominator_crosses_zero'):flags.append('bootstrap g_par qua 0')
   lines.append('| '+str(row['bucket'])+' | '+' | '.join(floatfmt(row[k]) for k in ('parent_gap','student_gap','retention','A_init'))+' | '+('; '.join(flags).replace('|','\\|') if flags else '—')+' |')
  lines.append('')
lines.extend(['Phụ lục này không thay đổi decision đã công bố. Nguồn: [decision_mech.json](decision_mech.json), [noise_paired.json](noise_paired.json), [layers.json](layers.json), [buckets.json](buckets.json) và A2 Q `decision_quant.json` đã khóa SHA trong config.','', '<!-- A2_M_ADDENDUM_END -->'])
report=base/'REPORT.md';text=report.read_text();start='<!-- A2_M_ADDENDUM_START -->'
if start in text:text=text.split(start)[0].rstrip()+'\n'
report.write_text(text.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
receipt=dict(spec_sha256=d['spec_sha256'],source_sha256={str(p):sha(p) for p in [base/'decision_mech.json',base/'noise_paired.json',base/'layers.json',base/'buckets.json',qpath]},n_rtn3_parents=12,n_rtn3_contrast_rows=3,n_iso_extra_contrast_rows=6,correlation_metrics=83,correlation_values=249,correlation_recomputed_max_abs=max_error,layer_dataset_rows=72,layer_contrasts=216,bucket_rows=40,cpu_only=True,no_model_forward=True,decision_unchanged=True)
(base/'final/addendum_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
