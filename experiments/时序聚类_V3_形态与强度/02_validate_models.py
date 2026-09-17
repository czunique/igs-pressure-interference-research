"""Sensitivity and free clustering baseline for the reference-guided V3 classification."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,importlib.util,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import adjusted_rand_score,silhouette_score
from threadpoolctl import threadpool_limits
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore');threadpool_limits(4)
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V3_形态与强度');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V3_形态与强度');OLD=B.parent/'时序聚类_V2_20260915'
s=importlib.util.spec_from_file_location('v3',B/'01_classify.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pd.read_pickle(B/'results.pkl');Y=np.load(B/'shape101.npy');base=d.morphology_code_v3.to_numpy();sens=[];same=[]
for threshold in [.1,.15,.25,.3]:
 lab,*_=m.assign(d,Y,weak_threshold=threshold);sens.append(dict(test='微弱幅度界限_'+str(threshold)+'MPa',agreement=float(np.mean(lab==base)),ari=adjusted_rand_score(base,lab)));same.append(lab==base)
for penalty in [.0003,.0012]:
 lab,*_=m.assign(d,Y,complexity_penalty=penalty);sens.append(dict(test='每参数惩罚_'+str(penalty),agreement=float(np.mean(lab==base)),ari=adjusted_rand_score(base,lab)));same.append(lab==base)
raw=pd.read_pickle(OLD/'raw_curves.pkl')
for factor in [.5,2.]:
 feat=[];yy=[]
 for i,r in d.iterrows():
  f,_,y=m.extract(raw[int(r.matrix_row)],r,factor);feat.append(f);yy.append(y)
 lab,*_=m.assign(pd.DataFrame(feat),np.array(yy));sens.append(dict(test='趋势平滑尺度_'+str(factor),agreement=float(np.mean(lab==base)),ari=adjusted_rand_score(base,lab)));same.append(lab==base)
print('MORPHOLOGY',pd.DataFrame(sens).to_string(index=False),flush=True);pd.DataFrame(sens).to_csv(O/'04_形态参数敏感性.csv',index=False,encoding='utf-8-sig');d['morphology_parameter_agreement']=np.mean(same,axis=0);d['morphology_uncertain']=d.morphology_uncertain|(d.morphology_parameter_agreement<.75)
# Free clustering in the proposed expanded feature space. No label forcing or curve relabeling.
features=['amplitude_mpa','positive_rise_p95_mpa','overall_rate_mpa_min','early_rate_mpa_min_v3','middle_rate_mpa_min_v3','late_rate_mpa_min_v3','early_middle_acceleration_mpa_min2','middle_late_acceleration_mpa_min2','positive_auc_mean_mpa','positive_auc_mpa_min','rising_duration_min','longest_rising_duration_min','early_shape_rate','middle_shape_rate','late_shape_rate','onset_fraction_v3','dip_depth_fraction','return_fraction']
a=d[features].to_numpy(float);a=np.sign(a)*np.log1p(np.abs(a)/np.array([1,1,.01,.01,.01,.01,.0001,.0001,1,100,60,60,1,1,1,1,1,1]));a=np.clip(RobustScaler(quantile_range=(10,90)).fit_transform(a),-3,3);np.save(B/'free_clustering_features.npy',a)
res=[];alllab={};rng=np.random.default_rng(m.SEED);ix=rng.choice(len(d),1600,replace=False)
for k in range(5,10):
 km=KMeans(k,n_init=30,random_state=m.SEED).fit(a);lab=km.labels_;alllab['k'+str(k)]=lab+1;ar=[]
 for b in range(30):
  groups=[]
  for pl,z in d.groupby('platform_id'):
   g=z.stage_group.unique();groups.extend(rng.choice(g,max(1,int(np.ceil(.8*len(g)))),replace=False))
  sub=np.flatnonzero(d.stage_group.isin(groups));l=KMeans(k,n_init=10,random_state=m.SEED+b).fit_predict(a[sub]);ar.append(adjusted_rand_score(lab[sub],l))
 res.append(dict(k=k,silhouette=float(silhouette_score(a[ix],lab[ix])),min_cluster_n=int(np.bincount(lab).min()),conditional_scale_group_ari_median=float(np.median(ar)),ari_to_guided_types=float(adjusted_rand_score(base,lab))));print('FREE',res[-1],flush=True)
pd.DataFrame(res).to_csv(O/'05_新增特征自由聚类对照.csv',index=False,encoding='utf-8-sig');tab=d[['sample_id','morphology_v3']].copy()
for key,val in alllab.items():tab[key]=val
tab.to_csv(O/'06_自由聚类逐条标签.csv',index=False,encoding='utf-8-sig')
# Intensity score sensitivity. Refit three score groups under each stated alternative.
cfg=json.loads((B/'config.json').read_text(encoding='utf-8'));cols=cfg['intensity_features'];z=d[['strength_component_'+c for c in cols]].to_numpy(float);baselevel=d.response_strength_grade.map({'弱':1,'中':2,'强':3}).to_numpy();risks=[];votes=[]
weights={'等指标权重':[1/6]*6,'幅度优先':[.5,.15,.05,.05,.05,.2],'持续时间面积优先':[.2,.2,.2,.15,.1,.15],'无累计面积时长':[.45,.3,0,0,0,.25]}
for name,w in weights.items():
 score=z@np.array(w);km=KMeans(3,n_init=30,random_state=m.SEED).fit(score[:,None]);c=np.sort(km.cluster_centers_.ravel());cut=(c[:-1]+c[1:])/2;lab=np.searchsorted(cut,score)+1;votes.append(lab==baselevel);risks.append(dict(test=name,n=len(d),agreement=float(np.mean(lab==baselevel)),ari=adjusted_rand_score(baselevel,lab),weak=int(sum(lab==1)),medium=int(sum(lab==2)),strong=int(sum(lab==3))))
for name,mask in {'排除操作突变':~d.operational_flag,'排除低频':~d.low_resolution,'排除片段和冲突':~d.fragment_only&~d.alternate_record_flag}.items():
 sub=np.flatnonzero(mask);v=np.log1p(d.iloc[sub][cols].to_numpy(float)/np.array(cfg['intensity_log_scales']));norm=np.quantile(v,.9,axis=0);zz=np.clip(v/np.maximum(norm,1e-6),0,1.5);score=zz@np.array(cfg['intensity_weights']);km=KMeans(3,n_init=30,random_state=m.SEED).fit(score[:,None]);c=np.sort(km.cluster_centers_.ravel());lab=np.searchsorted((c[:-1]+c[1:])/2,score)+1;risks.append(dict(test=name,n=len(sub),agreement=float(np.mean(lab==baselevel[sub])),ari=adjusted_rand_score(baselevel[sub],lab),weak=int(sum(lab==1)),medium=int(sum(lab==2)),strong=int(sum(lab==3))))
pd.DataFrame(risks).to_csv(O/'07_响应强度权重敏感性.csv',index=False,encoding='utf-8-sig');d['strength_weight_agreement']=np.mean(votes,axis=0)
d['risk_mapping_review_required']=(d.morphology_code_v3>=7)|d.morphology_uncertain|d.operational_flag|d.alternate_record_flag|d.fragment_only|~d.event_identity_known|~d.well_identity_known|(d.strength_weight_agreement<.75)
d['risk_grade_provisional']=np.where(d.risk_mapping_review_required,'待核查',d.response_strength_grade+'（暂定）');d['risk_training_candidate']=~d.risk_mapping_review_required
# Representative actual curves prefer supported parameters and low template error; no synthetic curves stand in for observations.
proto=[]
for code,g in d.groupby('morphology_code_v3'):
 target=g[g.morphology_parameter_agreement>=.75]
 if target.empty:target=g
 clean=target[~target.operational_flag&~target.fragment_only&target.event_identity_known&target.well_identity_known]
 if len(clean):target=clean
 # Select clear, actual examples for illustration; population medians remain calculated on all class members.
 clear=target
 if code==1:clear=target[(target.positive_rise_p95_mpa>=.03)&target.amplitude_mpa.between(.04,.15)]
 if code==3:clear=target[target.middle_shape_rate>1.7*target[['early_shape_rate','late_shape_rate']].max(axis=1)]
 if code==5:clear=target[(target.minimum_dp_mpa<-.15)&(target.dip_depth_fraction>.15)&(target.positive_rise_p95_mpa>.2)]
 if len(clear):target=clear
 mid=np.median(g.positive_rise_p95_mpa);cost=target.template_normalized_rmse+.025*np.abs(np.log1p(target.positive_rise_p95_mpa)-np.log1p(mid));idx=cost.idxmin();proto.append(dict(code=int(code),name=m.NAMES[code],model_row=int(idx),sample_id=d.iloc[idx].sample_id,file=d.iloc[idx].file,sheet=d.iloc[idx].sheet))
pd.DataFrame(proto).to_csv(O/'08_典型曲线来源.csv',index=False,encoding='utf-8-sig');d.to_pickle(B/'results.pkl');d.to_csv(O/'01_全量曲线形态与强度标签.csv',index=False,encoding='utf-8-sig');print('FINAL_UNCERTAIN',int(d.morphology_uncertain.sum()),'REVIEW',int(d.risk_mapping_review_required.sum()),flush=True)
print('RISK_SENS',pd.DataFrame(risks).to_string(index=False),flush=True)
