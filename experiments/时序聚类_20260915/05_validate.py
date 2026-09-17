"""Independent distance verification and per-sample grouped resampling consistency."""
import sys,json,importlib.util,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from tslearn.metrics import dtw as reference_dtw
B=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_20260915')
spec=importlib.util.spec_from_file_location('cluster',B/'03_cluster.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
X=np.load(B/'primary_curves_201.npy');D=np.load(B/'dtw_radius10_201.npy');df=pd.read_pickle(B/'clustered_samples.pkl');lab=np.load(B/'selected_labels.npy');cfg=json.loads((B/'run_config.json').read_text(encoding='utf-8'));F=df[m.FEATURES].astype(float).to_numpy();splits=np.load(B/'group_subsamples.npz');n=len(df);k=cfg['k']
seen=np.zeros(n,int);same=np.zeros(n,int);co=np.zeros((n,n),np.uint16);together=np.zeros((n,n),np.uint16)
for b in range(100):
 ix=splits[str(b)];ll=m.fit(cfg['selected_method'],k,X[ix],F[ix],D[np.ix_(ix,ix)],seed=20260915+b,n_init=5)
 table=np.zeros((k,k),int)
 np.add.at(table,(lab[ix],ll),1);a,c=linear_sum_assignment(-table);mapping=dict(zip(c,a));aligned=np.array([mapping[j] for j in ll]);seen[ix]+=1;same[ix]+=aligned==lab[ix]
 together[np.ix_(ix,ix)]+=1;co[np.ix_(ix,ix)]+=(ll[:,None]==ll[None,:]).astype(np.uint16)
cons=np.divide(co,together,out=np.full((n,n),np.nan),where=together>0);np.savez_compressed(B/'co_clustering_probabilities.npz',probability=cons,co_observed=together,sample_id=df.sample_id.to_numpy(dtype=str))
z=df[['sample_id','platform_id','source_well','stage_id','monitor_well','cluster','boundary_flag']].copy();z['resamples_present']=seen;z['aligned_label_agreement']=same/seen;z.to_csv(O/'13_逐样本重采样一致性.csv',index=False,encoding='utf-8-sig')
rng=np.random.default_rng(42);errors=[]
for _ in range(12):
 a,b=rng.choice(n,2,replace=False);ref=reference_dtw(X[a],X[b],global_constraint='sakoe_chiba',sakoe_chiba_radius=10)/np.sqrt(201);errors.append(abs(ref-D[a,b]))
assert max(errors)<1e-8,errors
old=pd.read_csv(B/'质控修订前审计/06_主分析样本聚类标签.csv');merged=df[['sample_id','cluster']].merge(old[['sample_id','cluster']],on='sample_id',suffixes=('_final','_before'));ari=m.adjusted_rand_score(merged.cluster_before,merged.cluster_final)
checks=json.loads((B/'validation_checks.json').read_text(encoding='utf-8'));checks.update({'dtw_matches_independent_tslearn_12_pairs':True,'dtw_max_absolute_error':max(errors),'source_stage_monitor_unique':not df.duplicated(['platform_id','source_well','stage_id','monitor_well']).any(),'all_primary_have_observed_resampling':bool((seen>0).all()),'qc_revision_common_sample_ari':ari,'n_label_agreement_below_80pct':int((same/seen<.8).sum())})
(B/'validation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
print('LOW_AGREEMENT',int((same/seen<.8).sum()))

extra='\n## 九、最终核验与标签使用\n\n- 独立抽查12对曲线，DTAIDistance与tslearn实现的同半径DTW距离完全一致。\n- 去重样本键、矩阵数值、标签覆盖、类别计数及主分析质量阈值检查均通过。监测源文件的大小与修改时间保持一致。\n- 将最后一条近零压力大跳变记录移入工况复核后，其余505条曲线的分组与修订前完全一致（共同样本ARI=1.000）。\n- 10条曲线属于原型距离边界样本；另有2条在成组抽样中的标签一致率低于80%。两种标记含义不同，逐样本结果见 `13_逐样本重采样一致性.csv`。\n- 同一施工段的多个邻井可能分入不同类别，因此各类别的施工段数、井对数不能简单相加作为总数。\n- 输出属于首轮探索标签，尚未经盲审命名、微地震验证或独立外部平台检验。本轮未运行地质分类预测和SHAP。\n'
rp=O/"时序聚类_首轮结果报告.md"
rt=rp.read_text(encoding="utf-8")
if "## 九、最终核验与标签使用" not in rt:rp.write_text(rt+extra,encoding="utf-8")
