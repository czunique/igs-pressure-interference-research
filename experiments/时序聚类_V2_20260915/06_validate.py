"""Independent checks, source immutability receipt and reproducibility notes."""
import sys,json,hashlib,importlib.util,pickle
from pathlib import Path
import numpy as np,pandas as pd
from tslearn.metrics import dtw as independent_dtw
from sklearn.preprocessing import RobustScaler
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915')
d=pd.read_pickle(B/'final_labels.pkl');raw=pd.read_pickle(B/'raw_curves.pkl');X=np.load(B/'model_shapes101.npy');Ds=np.load(B/'shape_DTW51.npy');checks={}
checks['unique_sample_ids']=len(d)==d.sample_id.nunique();checks['four_classes_exhaustive']=set(d.cluster)=={1,2,3,4};checks['no_nonpressure_variables']=not d.header.str.contains('砂|排量|液量|温度|15分钟压降|停泵压力').any();checks['no_known_self_pressure']=not ((d.source_well==d.monitor_well)&(d.platform_id==d.monitor_platform)&d.event_identity_known).any();checks['finite_curves']=bool(np.isfinite(X).all());checks['sample_matrix_correspondence']=len(d)==len(X)==len(Ds);checks['all_positive_duration']=bool((d.duration_min>0).all());checks['all_at_least_six_native_points']=bool((d.native_n>=6).all());checks['native_interval_positive']=bool((d.native_dt_s>0).all())
rng=np.random.default_rng(724);errors=[]
for i,j in rng.integers(0,len(d),size=(12,2)):
 a=np.clip(X[i,::2],-4,4);b=np.clip(X[j,::2],-4,4);dist=independent_dtw(a,b,global_constraint='sakoe_chiba',sakoe_chiba_radius=5)/np.sqrt(51);errors.append(abs(dist-Ds[i,j]))
checks['dtw_independent_max_error_12_pairs']=float(max(errors));checks['dtw_independent_pass']=max(errors)<1e-9
feat_errors=[];rows=[]
for i in rng.choice(len(d),24,replace=False):
 r=d.iloc[i];o=raw[int(r.matrix_row)];tt=(o['smooth_t']-o['smooth_t'][0])/60;p=o['smooth_p'];u=tt/tt[-1];mask=u<=.25
 if mask.sum()>=3:
  # Independent centered covariance slope, instead of pipeline polyfit.
  t=tt[mask];y=p[mask];slope=np.sum((t-t.mean())*(y-y.mean()))/np.sum((t-t.mean())**2);err=abs(slope-r.early_rate_mpa_min);feat_errors.append(err)
 amplitude=float(np.percentile(p,95)-np.percentile(p,5));rows.append(dict(sample_id=r.sample_id,amplitude_error=abs(amplitude-r.amplitude_range_mpa)))
checks['early_slope_independent_max_error_24_samples']=float(max(feat_errors));checks['amplitude_independent_max_error_24_samples']=float(max(x['amplitude_error'] for x in rows));checks['feature_independent_pass']=max(feat_errors)<1e-9 and max(x['amplitude_error'] for x in rows)<1e-9
inv=pd.read_csv(O/'01_全量文件覆盖.csv');unchanged=[]
for r in inv.itertuples():
 p=Path(r.path);st=p.stat();unchanged.append(st.st_size==r.size and st.st_mtime_ns==r.mtime)
checks['source_files_size_mtime_unchanged']=all(unchanged);checks['source_files_checked']=len(unchanged)
hash_checks=[]
for i in rng.choice(len(inv),20,replace=False):
 r=inv.iloc[i];hash_checks.append(hashlib.sha256(Path(r.path).read_bytes()).hexdigest()==r.sha256)
checks['source_hash_spotcheck_20_files']=all(hash_checks)
# Save every candidate assignment in a directly inspectable table.
labels=np.load(B/'candidate_labels.npz');tab=d[['sample_id','platform_id','source_well','stage_id','monitor_platform','monitor_well','cluster','morphology_type']].copy()
for key in labels.files:tab[key]=labels[key]+1
tab.to_csv(O/'14_全部候选方案逐曲线标签.csv',index=False,encoding='utf-8-sig')
# Build a fixed-training transformer bundle for future repeat use (do not refit on inference rows).
s=importlib.util.spec_from_file_location('cmp',B/'03_compare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);a=m.signedlog(d[m.AMP].to_numpy(float));sh=d[m.SHP].to_numpy(float);v=d[m.KIN].to_numpy(float,copy=True);v[:,:3]=m.signedlog(v[:,:3],.01);v[:,3]=m.signedlog(v[:,3],.0001);scalers=[RobustScaler(quantile_range=(10,90)).fit(z) for z in [sh,a,v]];F=np.concatenate([np.sqrt(w)*np.clip(sc.transform(z),-3,3)/np.sqrt(z.shape[1]) for sc,z,w in zip(scalers,[sh,a,v],[.6,.25,.15])],axis=1);checks['stored_transform_reproduces_features']=bool(np.allclose(F,np.load(B/'balanced_features.npy')))
with open(B/'final_kmeans.pkl','rb') as f:bundle=pickle.load(f)
bundle['scalers']=scalers;bundle['feature_blocks']=[m.SHP,m.AMP,m.KIN];bundle['weights']=[.6,.25,.15];bundle['note']='Fixed training scalers: shape direct; amplitude signedlog scale1; kinetics signedlog scales .01,.01,.01,.0001. Transform, clip +/-3, divide sqrt(block width), multiply sqrt(weight), concatenate.'
with open(B/'final_kmeans_with_scalers.pkl','wb') as f:pickle.dump(bundle,f)
# Avoid interpreting derived onset fields as validated when dates or observation completeness are unknown.
delay_ok=(d.window_basis=='absolute')&(~d.fragment_only)&(d.construction_coverage>=.95)&(d.baseline_method=='pre_pump_last180s')&d.true_pump_response_delay_min.notna()
pd.DataFrame({'sample_id':d.sample_id,'computed_pump_relative_delay_min':d.true_pump_response_delay_min,'date_and_baseline_supported_delay':delay_ok,'caution':'Onset censoring and source timestamp validity still require review; clock-only values are not verified response delays.'}).to_csv(O/'15_响应时间适用性审查.csv',index=False,encoding='utf-8-sig')
checks['delay_date_and_baseline_supported_n']=int(delay_ok.sum());checks={k:(bool(v) if isinstance(v,np.bool_) else v) for k,v in checks.items()};checks['pass']=all(v for k,v in checks.items() if isinstance(v,bool));(B/'validation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
(B/'复算说明.md').write_text('''# V2复算说明

原始E盘文件只读。所有写出文件限于本实验过程/结论目录。V1全部保留。

## 环境

使用 `../时序聚类_20260915/venv/Scripts/python.exe`。依赖版本见 `environment_versions.json`。

## 缓存与运行顺序

1. `01_recover.py`：读取V1的1764文件清单，解析原始工作表；缓存目录cache。`01b_union.py` 补充旧解析器有效通道。
2. `01c_fix_identity.py` 和 `01d_pump_template.py`：纠正已核查的源井/监测井身份及施工模板字段。
3. `01e_header_retry.py`、`01f_final_retry.py`：保留特殊表头/通道字典补读证据，再运行01c、01d。
4. `02_broad_features.py`：重复一致性核对、连续片段回收、观察窗选择和原生尺度特征提取。
5. `03_compare.py`：四算法×K2–6，共20方案；20次筛选重采样。DTW缓存由形态矩阵SHA256指纹确认。
6. `04_finalize.py`：K3/4/5各100次施工段分组重采样，输出四类标签和质量/平台敏感性结果。
7. `05_report.py`：报告、科学图、逐平台全曲线图、本地查询HTML及最终文件利用表。
8. `06_validate.py`：独立DTW/特征抽查、源文件大小/修改时间全检和20文件哈希抽查、全部候选标签CSV及固定训练尺度模型。

初次从原始文件完整重读因解析器版本改进，补读路径与缓存路径应保持一致。交付时的 `pressure_all.pkl` 是已完成多轮人工抽查与补读的最终通道快照；推荐从步骤4开始精确复现本轮结果。不要把清空缓存后的新解析结果未经核查直接当成本轮快照。

## 核心中间文件

- pressure_all.pkl：逐源文件、逐通道原始时间压力及来源；只读源文件的本地快照。
- broad_samples.pkl、raw_curves.pkl、curves101.npy、shapes101.npy：全候选样本、原生曲线、统一时间曲线及形态。
- model_samples.pkl：按sample_id排序的最终建模样本，matrix_row映射上述数组。
- balanced_features.npy、hybrid_distance.npy、shape_DTW51.npy：实际输入与评价距离。
- candidate_labels.npz：所有候选算法分配，内部从0编号；结论CSV中候选编号从1开始。
- final_labels.pkl：最终C1–C4编号及质量标记。
- final_kmeans_with_scalers.pkl：已保存训练尺度的模型。后续新曲线必须使用保存的scaler，禁止在待预测集合重新拟合尺度。

## 关键口径

样本是压力曲线观测，不一定是独立压窜事件。同段多个邻井有相关性；源井/段号不明时仅能作形态探索。未知事件按平台及记录日粗分组。相同原始数值重复导出只计一次；同事件读数不一致则保留变体并标记。

低频阈值是原生采样间隔>120秒。全零占位记录不作为无风险样本。观察段少于6点、小于5分钟或无法可靠切分的跨日长记录，不强行插值。长缺口仅回收最长满足条件片段。

幅度为平滑P95−P05。rate字段单位MPa/min，acceleration字段MPa/min²。native_rate_timescale_min、acceleration_timescale_min供分辨率审查。early_to_late_acceleration是早尾段斜率变化/时间，并非瞬时裂缝增长加速度。

onset_fraction为观察窗内位置；未检出时编码1且onset_censored=True。true_pump_response_delay_min是历史字段名，仅为有基线时按匹配窗口计算的延迟，并非所有行均经验证；必须结合结论15号响应时间适用性审查使用。

当前形态标签不是微地震验证后的压窜风险标签。candidate_for_later_supervised_model是质量筛选候选，不是已通过事件匹配和风险验证的最终训练集。
''',encoding='utf-8')
import importlib.metadata
versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib','tslearn','dtaidistance','python-calamine','xlrd']};(B/'environment_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8');print(json.dumps(checks,ensure_ascii=False,indent=2));assert checks['pass']

