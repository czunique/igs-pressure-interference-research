"""Full-coverage morphology clustering. K=3..5 prespecified interpretive range; K=2,6 sensitivity."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,warnings,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,cut_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score,adjusted_rand_score
from dtaidistance import dtw
from threadpoolctl import threadpool_limits
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore');threadpool_limits(4)
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915');SEED=20260915
AMP=['amplitude_range_mpa','initial_relative_rise_mpa','initial_relative_drop_mpa','end_dp_mpa']
SHP=['shape_level_10','shape_level_25','shape_level_50','shape_level_75','shape_level_90','shape_end','early_slope_shape','middle_slope_shape','late_slope_shape','dip_fraction','recovery_fraction','peak_time_fraction','trough_time_fraction','onset_fraction']
KIN=['early_rate_mpa_min','middle_rate_mpa_min','tail_rate_mpa_min','early_to_late_acceleration_mpa_min2']
def signedlog(x,scale=1):return np.sign(x)*np.log1p(np.abs(x)/scale)
def scale_block(x):return np.clip(RobustScaler(quantile_range=(10,90)).fit_transform(x),-3,3)/np.sqrt(x.shape[1])
def representation(d):
 a=scale_block(signedlog(d[AMP].to_numpy(float)));s=scale_block(d[SHP].to_numpy(float));q=d[KIN].to_numpy(float,copy=True);q[:,:3]=signedlog(q[:,:3],.01);q[:,3]=signedlog(q[:,3],.0001);v=scale_block(q)
 return np.c_[np.sqrt(.6)*s,np.sqrt(.25)*a,np.sqrt(.15)*v],a,s,v

def medoids(D,k,n_init=5,seed=SEED,max_iter=30):
 rng=np.random.default_rng(seed);best=None;n=len(D)
 for trial in range(n_init):
  meds=[int(rng.integers(n))]
  while len(meds)<k:
   ds=np.min(D[:,meds],axis=1)**2;ds[meds]=0;meds.append(int(rng.choice(n,p=ds/ds.sum())) if ds.sum()>0 else next(x for x in range(n) if x not in meds))
  meds=np.array(meds)
  for _ in range(max_iter):
   lab=np.argmin(D[:,meds],axis=1);lab[meds]=np.arange(k);new=meds.copy()
   for j in range(k):
    ix=np.flatnonzero(lab==j)
    if len(ix):new[j]=ix[np.argmin(D[np.ix_(ix,ix)].sum(axis=1))]
   if np.array_equal(new,meds):break
   meds=new
  lab=np.argmin(D[:,meds],axis=1);lab[meds]=np.arange(k);loss=np.min(D[:,meds],axis=1).sum()
  if best is None or loss<best[0]:best=(loss,lab.copy(),meds.copy())
 return best[1],best[2]

def fit(method,k,F,X,D,seed=SEED,n_init=10):
 if method=='Hybrid_DTW_medoid':return medoids(D,k,n_init,seed)[0]
 if method=='Balanced_feature_KMeans':return KMeans(k,n_init=n_init,random_state=seed).fit_predict(F)
 if method=='Balanced_feature_Ward':return cut_tree(linkage(F,method='ward'),n_clusters=k).ravel()
 if method=='Shape_KMeans':return KMeans(k,n_init=n_init,random_state=seed).fit_predict(X)
 raise ValueError(method)

def main():
 d=pd.read_pickle(B/'broad_samples.pkl');d=d[d.included].sort_values('sample_id').reset_index(drop=True);d.to_pickle(B/'model_samples.pkl');X=np.load(B/'shapes101.npy')[d.matrix_row.astype(int)];F,A,S,V=representation(d);np.save(B/'balanced_features.npy',F);np.save(B/'model_shapes101.npy',X)
 sh=np.ascontiguousarray(np.clip(X[:,::2],-4,4),dtype=np.double)
 fp=hashlib.sha256(sh.tobytes()).hexdigest();fpfile=B/'shape_DTW51.sha256'
 if not (B/'shape_DTW51.npy').exists() or not fpfile.exists() or fpfile.read_text()!=fp:
  ds=dtw.distance_matrix_fast(sh,window=6,parallel=True,use_pruning=False)/np.sqrt(51);np.fill_diagonal(ds,0);np.save(B/'shape_DTW51.npy',ds);fpfile.write_text(fp)
 else:ds=np.load(B/'shape_DTW51.npy')
 df=squareform(pdist(F));normal=lambda a:a/np.median(a[np.triu_indices(len(a),1)][a[np.triu_indices(len(a),1)]>0])
 D=.5*normal(ds)+.5*normal(df);np.fill_diagonal(D,0);assert np.isfinite(D).all();np.save(B/'hybrid_distance.npy',D)
 rng=np.random.default_rng(SEED);evalix=np.sort(rng.choice(len(d),min(1600,len(d)),replace=False));np.save(B/'silhouette_indices.npy',evalix)
 methods=['Balanced_feature_KMeans','Balanced_feature_Ward','Shape_KMeans','Hybrid_DTW_medoid'];results=[];labels={};supports=[]
 for method in methods:
  for k in range(2,7):
   st=time.time();lab=fit(method,k,F,sh,D,n_init=12);key=method+'__'+str(k);labels[key]=lab
   sup=[dict(cluster=int(j),n=int(sum(lab==j)),platforms=d.iloc[np.flatnonzero(lab==j)].platform_id.nunique(),stage_groups=d.iloc[np.flatnonzero(lab==j)].stage_group.nunique()) for j in np.unique(lab)]
   sil=float(silhouette_score(D[np.ix_(evalix,evalix)],lab[evalix],metric='precomputed'))
   row=dict(method=method,k=k,common_hybrid_silhouette=sil,min_n=min(z['n'] for z in sup),min_platforms=min(z['platforms'] for z in sup),min_stage_groups=min(z['stage_groups'] for z in sup),max_fraction=max(z['n'] for z in sup)/len(d),seconds=time.time()-st)
   row['support_gate']=row['min_n']>=30 and row['min_platforms']>=3 and row['min_stage_groups']>=15
   results.append(row);supports.extend([dict(method=method,k=k,**z) for z in sup]);print('FIT',row,flush=True)
 np.savez(B/'candidate_labels.npz',**labels);pd.DataFrame(results).to_csv(O/'05_算法与类别数比较.csv',index=False,encoding='utf-8-sig');pd.DataFrame(supports).to_csv(O/'06_候选类别支持度.csv',index=False,encoding='utf-8-sig')
 # 20 preliminary grouped repetitions for K=3..5; preprocessing refitted on each subset.
 splits=[]
 for b in range(100):
  chosen=[]
  for plat,z in d.groupby('platform_id'):
   g=z.stage_group.unique();chosen.extend(rng.choice(g,max(1,int(np.ceil(.8*len(g)))),replace=False))
  splits.append(np.flatnonzero(d.stage_group.isin(chosen)))
 np.savez(B/'group_splits.npz',**{str(i):x for i,x in enumerate(splits)})
 stability=[]
 for row in results:
  if row['k'] not in [3,4,5]:continue
  ref=labels[row['method']+'__'+str(row['k'])];aris=[]
  for b,ix in enumerate(splits[:20]):
   subF,*_=representation(d.iloc[ix]);subD=D[np.ix_(ix,ix)]
   lab=fit(row['method'],row['k'],subF,sh[ix],subD,SEED+b,5);ari=adjusted_rand_score(ref[ix],lab);aris.append(ari);stability.append(dict(method=row['method'],k=row['k'],replicate=b,ari=ari,n=len(ix)))
  row['ari_median']=float(np.median(aris));row['ari_p10']=float(np.quantile(aris,.1));print('STABILITY',row['method'],row['k'],row['ari_median'],row['ari_p10'],flush=True)
  pd.DataFrame(results).to_csv(O/'05_算法与类别数比较.csv',index=False,encoding='utf-8-sig');pd.DataFrame(stability).to_csv(B/'screen_stability.csv',index=False)
 (B/'method_config.json').write_text(json.dumps(dict(seed=SEED,amp=AMP,shape=SHP,kinematics=KIN,weights=[.6,.25,.15],hybrid='0.5 normalized shape DTW + 0.5 normalized balanced-feature Euclidean',dtw_length=51,dtw_radius=5,silhouette_sample_size=len(evalix),preferred_k=[3,4,5],diagnostic_k=[2,6],raw_acceleration='exported for inspection only; sampling resolution differs; primary acceleration uses early-to-late slope change',stability='80 percent construction-stage groups within each platform; feature model preprocessing refitted; hybrid distance conditional on full-data scaling'),ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',len(d),flush=True)
if __name__=='__main__':main()

