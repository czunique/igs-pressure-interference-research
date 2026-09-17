"""Final four-type interpretation and conditional-label reliability checks."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,importlib.util,json,pickle,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score,silhouette_samples
from sklearn.cluster import KMeans
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore')
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915')
s=importlib.util.spec_from_file_location('cmp',B/'03_compare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def main():
 d=pd.read_pickle(B/'model_samples.pkl');F=np.load(B/'balanced_features.npy');D=np.load(B/'hybrid_distance.npy');labs=np.load(B/'candidate_labels.npz');ref=labs['Balanced_feature_KMeans__4'];splits=np.load(B/'group_splits.npz');hist=[];votes=np.zeros((len(d),4));seen=np.zeros(len(d))
 for k in [3,4,5]:
  rk=labs['Balanced_feature_KMeans__'+str(k)];aris=[]
  for b in range(100):
   ix=splits[str(b)];f,*_=m.representation(d.iloc[ix]);lab=KMeans(k,n_init=10,random_state=m.SEED+b).fit_predict(f);ari=adjusted_rand_score(rk[ix],lab);aris.append(ari);hist.append(dict(k=k,replicate=b,ari=ari,n=len(ix)))
   if k==4:
    cross=np.zeros((4,4),int)
    for j in range(4):
     for l in range(4):cross[j,l]=sum((ref[ix]==j)&(lab==l))
    rows,cols=linear_sum_assignment(-cross);mapping=dict(zip(cols,rows));aligned=np.array([mapping[x] for x in lab]);votes[ix,aligned]+=1;seen[ix]+=1
  print('100_GROUP_STABILITY',k,'median',np.median(aris),'p10',np.quantile(aris,.1),flush=True)
 pd.DataFrame(hist).to_csv(O/'07_100次施工段分组稳定性.csv',index=False,encoding='utf-8-sig');d['resampling_seen']=seen.astype(int);d['label_agreement']=votes[np.arange(len(d)),ref]/seen
 # Label order is based on computed full-data profiles, not manual curve relabeling.
 prof=d.assign(raw_cluster=ref).groupby('raw_cluster').median(numeric_only=True);weak=int(prof.amplitude_range_mpa.idxmin());decline=int(prof.end_dp_mpa.idxmin());up=[j for j in range(4) if j not in [weak,decline]];rapid=max(up,key=lambda j:prof.loc[j,'early_slope_shape']-prof.loc[j,'late_slope_shape']);late=next(j for j in up if j!=rapid);mapping={weak:1,rapid:2,late:3,decline:4};names={1:'低幅波动型',2:'早期上升趋缓型',3:'迟缓响应持续上升型',4:'降压主导型'}
 d['cluster']=np.array([mapping[j] for j in ref]);d['morphology_type']=d.cluster.map(names);d['hybrid_silhouette']=silhouette_samples(D,ref,metric='precomputed');d['boundary_flag']=(d.label_agreement<.8)|(d.hybrid_silhouette<0)
 d['risk_label_validated']=False;d['candidate_for_later_supervised_model']=d.event_identity_known&d.well_identity_known&~d.fragment_only&~d.alternate_record_flag&~d.operational_flag&~d.boundary_flag
 d['initial_drop_then_rise_descriptor']=(d.initial_relative_drop_mpa<=-.2)&(d.tail_rate_mpa_min>.001)&(d.end_dp_mpa-d.initial_relative_drop_mpa>.5)
 # Real observed medoids in the shared distance; alternative exemplars favor usable metadata.
 proto=[]
 for cl in range(1,5):
  ix=np.flatnonzero(d.cluster.to_numpy()==cl);med=ix[np.argmin(D[np.ix_(ix,ix)].sum(axis=1))];proto.append(dict(cluster=cl,model_row=int(med),sample_id=d.iloc[med].sample_id,morphology_type=names[cl]))
 pd.DataFrame(proto).to_csv(O/'09_真实典型曲线索引.csv',index=False,encoding='utf-8-sig')
 d.to_pickle(B/'final_labels.pkl');d.to_csv(O/'08_全量曲线聚类标签.csv',index=False,encoding='utf-8-sig')
 sens=[]
 masks={'去除低频记录':~d.low_resolution,'去除操作突变标记':~d.operational_flag,'去除重复记录冲突':~d.alternate_record_flag,'去除局部片段':~d.fragment_only,'仅日期可靠施工窗口':d.window_basis=='absolute','仅已知源井和段号':d.event_identity_known}
 for name,mask in masks.items():
  ix=np.flatnonzero(mask);f,*_=m.representation(d.iloc[ix]);pred=KMeans(4,n_init=20,random_state=m.SEED).fit_predict(f);sens.append(dict(test=name,n=len(ix),ari=adjusted_rand_score(ref[ix],pred)))
 for plat,z in d.groupby('platform_id'):
  if len(z)<20:continue
  ix=np.flatnonzero(d.platform_id!=plat);f,*_=m.representation(d.iloc[ix]);pred=KMeans(4,n_init=12,random_state=m.SEED).fit_predict(f);sens.append(dict(test='留一平台_'+plat,n=len(ix),ari=adjusted_rand_score(ref[ix],pred)))
 _,A,S,V=m.representation(d)
 for weight in [.4,.5,.7]:
  f=np.c_[np.sqrt(weight)*S,np.sqrt((1-weight)*.625)*A,np.sqrt((1-weight)*.375)*V];pred=KMeans(4,n_init=20,random_state=m.SEED).fit_predict(f);sens.append(dict(test='形态权重_'+str(weight),n=len(d),ari=adjusted_rand_score(ref,pred)))
 pd.DataFrame(sens).to_csv(O/'10_质量与平台敏感性.csv',index=False,encoding='utf-8-sig')
 model=KMeans(4,n_init=12,random_state=m.SEED).fit(F);assert adjusted_rand_score(ref,model.labels_)>.999
 with open(B/'final_kmeans.pkl','wb') as f:pickle.dump(dict(model=model,raw_to_cluster=mapping,cluster_names=names,feature_training_data=d[m.AMP+m.SHP+m.KIN],note='Recreate signed transformations and full-training RobustScaler using 03_compare.representation; never refit scaling to prediction data.'),f)
 summary=[]
 for cl,z in d.groupby('cluster'):
  r=dict(cluster=int(cl),name=names[cl],n=len(z),platforms=z.platform_id.nunique(),stages=z.stage_group.nunique(),low_resolution_n=int(z.low_resolution.sum()),boundary_n=int(z.boundary_flag.sum()),alternate_n=int(z.alternate_record_flag.sum()),fragment_n=int(z.fragment_only.sum()))
  for feat in ['amplitude_range_mpa','end_dp_mpa','early_rate_mpa_min','middle_rate_mpa_min','tail_rate_mpa_min','early_to_late_acceleration_mpa_min2','onset_fraction','true_pump_response_delay_min','label_agreement','hybrid_silhouette']:
   r[feat+'_median']=float(z[feat].median());r[feat+'_q25']=float(z[feat].quantile(.25));r[feat+'_q75']=float(z[feat].quantile(.75))
  summary.append(r)
 pd.DataFrame(summary).to_csv(O/'11_四类形态特征汇总.csv',index=False,encoding='utf-8-sig');print('FINAL',json.dumps(summary,ensure_ascii=False),flush=True)
 (B/'selected_model.json').write_text(json.dumps(dict(method='Balanced_feature_KMeans',k=4,decision='3 groups merge two rising response kinetics; 4 groups retain weak, declining, early-fast and late-rising patterns; 5 groups further split the late-rising group mainly by magnitude. Four is the interpretable compromise, not the global silhouette maximum.',names=names,raw_mapping=mapping,seed=m.SEED,observation_count=len(d),ambiguous_count=int(d.boundary_flag.sum()),candidate_supervised_count=int(d.candidate_for_later_supervised_model.sum())),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
