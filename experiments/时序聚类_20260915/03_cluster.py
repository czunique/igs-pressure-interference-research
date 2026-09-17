"""Comparable clustering on common DTW dissimilarities, grouped resampling and sensitivity."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,warnings,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage,cut_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score,silhouette_samples,adjusted_rand_score
from dtaidistance import dtw
from threadpoolctl import threadpool_limits
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore');threadpool_limits(4)
BASE=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915');OUT=Path('F:/论文库/IGS/实验/结论/时序聚类_20260915')
FEATURES=['a60_mpa','end_dp_mpa','min_dp_mpa','area_mean_mpa','max_rate60_mpa_min','onset_fraction','onset_censored']

def medoids(D,k,n_init=20,seed=20260915):
 rng=np.random.default_rng(seed);best=None;n=len(D)
 for trial in range(n_init):
  meds=[int(rng.integers(n))]
  while len(meds)<k:
   ds=np.min(D[:,meds],axis=1)**2;ds[meds]=0
   nxt=int(rng.choice(n,p=ds/ds.sum())) if ds.sum()>0 else next(x for x in range(n) if x not in meds)
   meds.append(nxt)
  meds=np.array(meds)
  for _ in range(100):
   lab=np.argmin(D[:,meds],axis=1);lab[meds]=np.arange(k);new=meds.copy()
   for j in range(k):
    ix=np.where(lab==j)[0]
    if len(ix):new[j]=ix[np.argmin(D[np.ix_(ix,ix)].sum(axis=1))]
   if np.array_equal(new,meds):break
   meds=new
  lab=np.argmin(D[:,meds],axis=1);lab[meds]=np.arange(k)
  loss=np.min(D[:,meds],axis=1).sum()
  if best is None or loss<best[0]:best=(loss,lab.copy(),meds.copy())
 return best[1],best[2]

def fit(method,k,X,F,D,seed=20260915,n_init=20):
 if method=='DTW_kmedoids':return medoids(D,k,n_init,seed)[0]
 if method=='Euclidean_kmeans':return KMeans(k,n_init=n_init,max_iter=300,random_state=seed).fit_predict(X)
 if method=='Feature_Ward':return cut_tree(linkage(RobustScaler().fit_transform(F),method='ward'),n_clusters=k).ravel()
 raise ValueError(method)

def distances(X,radius=10):
 # dtaidistance window=1 is the diagonal; +1 implements inclusive radius.
 D=dtw.distance_matrix_fast(np.ascontiguousarray(X,dtype=np.double),window=radius+1,parallel=True,use_pruning=False)
 np.fill_diagonal(D,0);assert np.all(np.isfinite(D));assert np.allclose(D,D.T)
 return D/np.sqrt(X.shape[1])

def metrics(df,D,labels):
 counts=np.bincount(labels)
 supports=[]
 for k in np.unique(labels):
  z=df.iloc[np.flatnonzero(labels==k)]
  supports.append({'cluster':int(k),'n':len(z),'platforms':z.platform_id.nunique(),'pairs':z.physical_pair_id.nunique(),'stages':z.stage_group.nunique()})
 mature=all(s['stages']>=20 and s['pairs']>=2 and s['platforms']>=2 for s in supports)
 return {'common_dtw_silhouette':float(silhouette_score(D,labels,metric='precomputed')),'min_cluster_n':int(counts.min()),'max_cluster_fraction':float(counts.max()/len(df)),'min_cluster_stages':min(s['stages'] for s in supports),'min_cluster_pairs':min(s['pairs'] for s in supports),'min_cluster_platforms':min(s['platforms'] for s in supports),'support_gate':mature},supports

def main():
 full=pd.read_pickle(BASE/'samples.pkl');df=full[full.status=='primary'].sort_values('sample_id').copy().reset_index(drop=True)
 X=np.load(BASE/'all_curves_201.npy')[df.matrix_row.astype(int)]
 assert len(df)>=20,'Insufficient primary samples for multi-method clustering'
 assert len(df)==df.sample_id.nunique();assert np.isfinite(X).all()
 F=df[FEATURES].astype(float).to_numpy();assert np.isfinite(F).all()
 np.save(BASE/'primary_curves_201.npy',X);df.to_csv(BASE/'primary_sample_order.csv',index=False,encoding='utf-8-sig')
 print('PRIMARY',len(df),'platforms',df.platform_id.nunique(),'stages',df.stage_group.nunique(),flush=True)
 dp=BASE/'dtw_radius10_201.npy'
 fingerprint=hashlib.sha256(X.tobytes()).hexdigest();fp=BASE/'dtw_radius10_201.sha256'
 if dp.exists() and fp.exists() and fp.read_text()==fingerprint:D=np.load(dp)
 else:
  D=distances(X);np.save(dp,D);fp.write_text(fingerprint)
 methods=['DTW_kmedoids','Euclidean_kmeans','Feature_Ward'];results=[];all_labels={};supports=[]
 for m in methods:
  for k in range(2,7):
   t=time.time();labels=fit(m,k,X,F,D);res,sup=metrics(df,D,labels);key=m+'__'+str(k)
   results.append({'method':m,'k':k,**res,'fit_seconds':time.time()-t});all_labels[key]=labels
   supports.extend([{'method':m,'k':k,**s} for s in sup]);print('FIT',m,k,res,flush=True)
 pd.DataFrame(supports).to_csv(OUT/'04_各候选方案类别支持度.csv',index=False,encoding='utf-8-sig')
 np.savez(BASE/'all_method_labels.npz',**all_labels)
 # 100 group subsamples, stratified within platform; entire source-stage groups travel together.
 rng=np.random.default_rng(20260915);splits=[]
 for b in range(100):
  chosen=[]
  for plat,z in df.groupby('platform_id'):
   groups=z.stage_group.unique();chosen.extend(rng.choice(groups,size=max(1,int(np.ceil(.8*len(groups)))),replace=False))
  splits.append(np.flatnonzero(df.stage_group.isin(chosen)))
 np.savez(BASE/'group_subsamples.npz',**{str(i):x for i,x in enumerate(splits)})
 stability=[]
 for res in results:
  m,k=res['method'],res['k'];ref=all_labels[m+'__'+str(k)];aris=[];pairaris=[];prev=None
  for b,ix in enumerate(splits):
   lab=fit(m,k,X[ix],F[ix],D[np.ix_(ix,ix)],seed=20260915+b,n_init=5)
   ari=adjusted_rand_score(ref[ix],lab);aris.append(ari)
   if prev is not None:
    common,a,bb=np.intersect1d(prev[0],ix,return_indices=True);pari=adjusted_rand_score(prev[1][a],lab[bb]);pairaris.append(pari)
   else:pari=np.nan
   stability.append({'method':m,'k':k,'replicate':b,'n':len(ix),'ari_to_full':ari,'ari_adjacent_overlap':pari});prev=(ix,lab)
  res.update(ari_median=float(np.median(aris)),ari_p10=float(np.quantile(aris,.1)),overlap_ari_median=float(np.median(pairaris)))
  print('STABILITY',m,k,res['ari_median'],flush=True)
  pd.DataFrame(stability).to_csv(OUT/'05_分组重采样稳定性.csv',index=False,encoding='utf-8-sig')
 tab=pd.DataFrame(results);tab['stable_gate']=tab.ari_p10>=.7;tab['eligible']=tab.support_gate&tab.stable_gate
 pool=tab[tab.eligible] if tab.eligible.any() else tab
 peak=pool.common_dtw_silhouette.max();near=pool[pool.common_dtw_silhouette>=peak-.02]
 chosen=near.sort_values(['k','ari_median','common_dtw_silhouette'],ascending=[True,False,False]).iloc[0]
 method,k=chosen['method'],int(chosen.k);labels=all_labels[method+'__'+str(k)]
 # Identifier order is descriptive, by median sustained pressure rise, never an asserted risk order.
 order=sorted(np.unique(labels),key=lambda j:df.iloc[np.flatnonzero(labels==j)].a60_mpa.median())
 mapping={int(j):i for i,j in enumerate(order)};labels=np.array([mapping[int(j)] for j in labels])
 tab['selected']=(tab.method==method)&(tab.k==k);tab.to_csv(OUT/'03_算法与类别数比较.csv',index=False,encoding='utf-8-sig')
 meds=np.array([ix[np.argmin(D[np.ix_(ix,ix)].sum(axis=1))] for ix in [np.flatnonzero(labels==j) for j in range(k)]])
 dis=D[:,meds];nearest=dis.argmin(axis=1);sorteddis=np.sort(dis,axis=1)
 margin=(sorteddis[:,1]-sorteddis[:,0])/(sorteddis[:,1]+1e-12)
 df['cluster']=['C'+str(j+1) for j in labels];df['prototype_distance']=dis[np.arange(len(df)),labels];df['nearest_prototype_cluster']=['C'+str(j+1) for j in nearest];df['boundary_margin']=margin;df['boundary_flag']=margin<.1;df['silhouette']=silhouette_samples(D,labels,metric='precomputed');df['provisional_pressure_grade']=pd.cut(df.a60_mpa,[-np.inf,1,5,np.inf],labels=['<1 MPa','1–5 MPa','≥5 MPa'],right=False)
 df.to_csv(OUT/'06_主分析样本聚类标签.csv',index=False,encoding='utf-8-sig')
 df.to_pickle(BASE/'clustered_samples.pkl');np.save(BASE/'selected_labels.npy',labels);np.save(BASE/'prototype_indices.npy',meds)
 summary=[]
 for j in range(k):
  z=df[labels==j];q={'cluster':'C'+str(j+1),'n':len(z),'platforms':z.platform_id.nunique(),'pairs':z.physical_pair_id.nunique(),'stages':z.stage_group.nunique(),'prototype_id':df.iloc[meds[j]].sample_id,'prototype_file':df.iloc[meds[j]].pressure_file,'boundary_n':int(z.boundary_flag.sum())}
  for f in ['a60_mpa','min_dp_mpa','end_dp_mpa','onset_min','max_rate60_mpa_min','duration_min']:
   q.update({f+'_median':float(z[f].median()),f+'_q25':float(z[f].quantile(.25)),f+'_q75':float(z[f].quantile(.75))})
  q.update({'rise_not_detected_n':int(z.onset_censored.sum()),'grade_lt1_fraction':float((z.a60_mpa<1).mean()),'grade_1to5_fraction':float(((z.a60_mpa>=1)&(z.a60_mpa<5)).mean()),'grade_ge5_fraction':float((z.a60_mpa>=5).mean())});summary.append(q)
 pd.DataFrame(summary).to_csv(OUT/'07_类别响应特征汇总.csv',index=False,encoding='utf-8-sig')
 sensitivity=[]
 for name,Y,r in [('radius0',X,0),('radius5',X,5),('radius20',X,20),('length101',X[:,::2],5),('per_curve_zscore',(X-X.mean(axis=1,keepdims=True))/np.maximum(X.std(axis=1,keepdims=True),.01),10)]:
  DD=distances(Y,r);ll=medoids(DD,k,20)[0]
  sensitivity.append({'scenario':name,'method':'DTW_kmedoids','k':k,'ari_vs_selected':adjusted_rand_score(labels,ll),'ari_vs_dtw_same_k':adjusted_rand_score(all_labels['DTW_kmedoids__'+str(k)],ll),'own_silhouette':silhouette_score(DD,ll,metric='precomputed'),'common_original_dtw_silhouette':silhouette_score(D,ll,metric='precomputed')});print('SENS',sensitivity[-1],flush=True)
 raws=pd.read_pickle(BASE/'event_raw_curves.pkl')
 unsmoothed=np.array([np.interp(np.linspace(0,row.duration_min,201),raws[int(row.matrix_row)]['t_min'],raws[int(row.matrix_row)]['p_mpa']-row.baseline_mpa) for _,row in df.iterrows()])
 for name,Y in [('no_median_smoothing',unsmoothed)]:
  DD=distances(Y,10);ll=medoids(DD,k,20)[0]
  sensitivity.append({'scenario':name,'method':'DTW_kmedoids','k':k,'ari_vs_selected':adjusted_rand_score(labels,ll),'ari_vs_dtw_same_k':adjusted_rand_score(all_labels['DTW_kmedoids__'+str(k)],ll),'own_silhouette':silhouette_score(DD,ll,metric='precomputed'),'common_original_dtw_silhouette':silhouette_score(D,ll,metric='precomputed')})
 pd.DataFrame(sensitivity).to_csv(OUT/'08_时间与幅度敏感性.csv',index=False,encoding='utf-8-sig')
 # Leave-one-platform-out checks are sensitivity of this exploratory dictionary, not supervised external validation.
 logo=[]
 for plat in df.platform_id.unique():
  ix=np.flatnonzero(df.platform_id.to_numpy()!=plat)
  if len(ix)<=k:continue
  ll=fit(method,k,X[ix],F[ix],D[np.ix_(ix,ix)],n_init=20)
  logo.append({'left_out_platform':plat,'n_retained':len(ix),'ari_retained':adjusted_rand_score(labels[ix],ll)})
 pd.DataFrame(logo).to_csv(OUT/'09_留一平台稳定性.csv',index=False,encoding='utf-8-sig')
 # Rejected low quality curves keep separate supplementary labels, never alter the fitted model.
 supplement=full[full.status=='supplementary'].copy();SX=np.load(BASE/'all_curves_201.npy')[supplement.matrix_row.astype(int)]
 if len(supplement):
  SD=np.array([[dtw.distance_fast(np.ascontiguousarray(x),np.ascontiguousarray(X[m]),window=11)/np.sqrt(201) for m in meds] for x in SX])
  near=SD.argmin(axis=1);supplement['nearest_cluster']=['C'+str(j+1) for j in near];supplement['nearest_distance']=SD.min(axis=1)
  thresholds=np.array([np.quantile(dis[labels==j,j],.95) for j in range(k)])
  supplement['beyond_primary_p95']=SD.min(axis=1)>thresholds[near];supplement['label_status']='仅供复核_不作正式训练标签'
  supplement.to_csv(OUT/'10_补充曲线临时归类.csv',index=False,encoding='utf-8-sig')
 config={'selected_method':method,'k':k,'n':len(df),'platforms':df.platform_id.nunique(),'stages':df.stage_group.nunique(),'pairs':df.physical_pair_id.nunique(),'primary_all_data_exploratory':True,'support_gate_passed':bool(chosen.support_gate),'stability_gate_passed':bool(chosen.stable_gate),'silhouette':float(chosen.common_dtw_silhouette),'ari_median':float(chosen.ari_median),'ari_p10':float(chosen.ari_p10),'prototype_reassignment_agreement':float((nearest==labels).mean()),'boundary_fraction':float(df.boundary_flag.mean()),'features':FEATURES,'main_initializations':20,'resampling_initializations':5,'group_replicates':100,'random_seed':20260915}
 (BASE/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8');print('FINAL',config,flush=True)
if __name__=='__main__':main()
