"""V3: reference-guided morphology, separate physical response intensity. No source writes."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,warnings,pickle
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
threadpool_limits(4);sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore')
OLD=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');B=Path('F:/论文库/IGS/实验/过程/时序聚类_V3_形态与强度');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V3_形态与强度');SEED=20260915
NAMES={1:'涨幅微弱型',2:'快速稳定型',3:'渐进上升型',4:'稳定缓升型',5:'先降后升型',6:'后期加速上升型',7:'冲高回落型',8:'降压主导型',9:'复杂波动或待判型'}

def extract(o,r,smooth_factor=1.):
 t=np.array(o['smooth_t'],float);p=np.array(o['smooth_p'],float);t=(t-t[0])/60;T=t[-1];u=t/T;dt=np.median(np.diff(t));noise=float(r.baseline_noise_mpa);eps=max(.05,3*noise)
 # Secondary smoothing provides meaningful trends across ~5% of observation duration, at native or coarser resolution.
 w=int(round(max(3.,.05*T)*smooth_factor/dt))|1;w=min(max(3,w),len(p)//2*2-1)
 if w>=5:p=savgol_filter(p,w,2,mode='interp')
 p0=float(np.median(p[u<=.05])) if sum(u<=.05)>=2 else float(p[0]);y=p-p0;base=float(r.baseline_mpa);yb=p-base
 xu=np.linspace(0,1,101);x=np.interp(xu,u,y);amp=float(np.quantile(y,.95)-np.quantile(y,.05));peak=max(0.,float(np.quantile(y,.95)));end=float(np.median(y[u>=.95]));dip=float(np.min(y));trough=float(u[np.argmin(y)]);peak_time=float(u[np.argmax(y)])
 def slope(a,b):
  ii=(u>=a)&(u<=b);tt=t[ii];pp=p[ii]
  if len(tt)>=3:return float(np.sum((tt-tt.mean())*(pp-pp.mean()))/np.sum((tt-tt.mean())**2))
  return float((np.interp(b,u,p)-np.interp(a,u,p))/((b-a)*T))
 early,middle,late=[slope(a,b) for a,b in [(0,1/3),(1/3,2/3),(2/3,1)]];overall=slope(0,1);scale=max(amp,.2,3*noise);den=T/scale
 rates=np.gradient(p,t);acc=np.gradient(rates,t)
 # Total and longest rising episodes on a common 21-knot trend representation; time resolution cannot improve on native observations.
 nknots=min(21,len(p));rt=np.linspace(0,T,nknots);rp=np.interp(rt,t,p);diff=np.diff(rp);interval=np.diff(rt);growing=diff>max(.01,noise);above=(rp[:-1]+rp[1:])/2-p0>eps;run=0.;longest=0.
 for ii,yes in enumerate(growing):run=run+interval[ii] if yes else 0.;longest=max(longest,run)
 rise_time=float(interval[growing].sum());positive_time=float(interval[above].sum());auc=float(np.trapezoid(np.maximum(y,0),t));signed=float(np.trapezoid(y,t));positive_base=float(np.trapezoid(np.maximum(yb,0),t))
 hit=y>eps;on=None
 for j in np.flatnonzero(hit):
  if j+1<len(hit) and hit[j+1]:on=float(u[j]);break
 ef=eps/scale;yy=x/scale
 # Persistent decline, return, and plateau criteria use coarse segment trends, not isolated extrema.
 out=dict(p0_window_mpa=p0,window_duration_min=T,noise_mpa=noise,noise_detection_mpa=eps,trend_smoothing_min=w*dt,amplitude_mpa=amp,positive_rise_p95_mpa=peak,peak_rise_mpa=max(0.,float(np.max(y))),end_rise_mpa=end,minimum_dp_mpa=dip,overall_rate_mpa_min=overall,endpoint_rate_mpa_min=end/T,early_rate_mpa_min_v3=early,middle_rate_mpa_min_v3=middle,late_rate_mpa_min_v3=late,initial10_rate_mpa_min=slope(0,.1),tail10_rate_mpa_min=slope(.9,1),early_middle_acceleration_mpa_min2=(middle-early)/(T/3),middle_late_acceleration_mpa_min2=(late-middle)/(T/3),acceleration_p95_native_mpa_min2=float(np.quantile(acc,.95)),acceleration_p05_native_mpa_min2=float(np.quantile(acc,.05)),positive_auc_mpa_min=auc,signed_auc_mpa_min=signed,positive_auc_mean_mpa=auc/T,positive_auc_prepump_mpa_min=positive_base,positive_duration_min=positive_time,rising_duration_min=rise_time,longest_rising_duration_min=longest,rising_fraction=rise_time/T,positive_fraction=positive_time/T,early_shape_rate=early*den,middle_shape_rate=middle*den,late_shape_rate=late*den,onset_fraction_v3=on if on is not None else 1.,onset_censored_v3=on is None,peak_fraction_v3=peak_time,trough_fraction_v3=trough,return_fraction=(float(np.max(y))-end)/max(amp,.2),dip_depth_fraction=max(0.,-dip)/scale,scale_mpa=scale,shape_end_v3=end/scale,shape_auc_v3=auc/(T*scale),trend_knots=nknots)
 return out,x,yy

def library():
 u=np.linspace(0,1,101);curves=[];labels=[];params=[]
 def add(cl,g,par):curves.append(g);labels.append(cl);params.append(par)
 add(4,u,dict(form='linear',parameters=1))
 for k in [3,4.5,6,9,14]:
  for a in [0.,.04,.08]:
   v=np.maximum(u-a,0)/(1-a);g=(1-np.exp(-k*v))/(1-np.exp(-k));add(2,g,dict(form='early_exponential_plateau',k=k,onset=a,parameters=3))
 for k in [6,8,11,15,20]:
  for c in [.3,.4,.5,.6,.7]:
   g=expit(k*(u-c));g=(g-g[0])/(g[-1]-g[0]);add(3,g,dict(form='logistic',k=k,center=c,parameters=3))
 for a in [.1,.2,.3,.4,.5]:
  for depth in [.08,.15,.3,.5,1.,2.]:
   for power in [1.,1.5,2.]:
    rise=(np.maximum(u-a,0)/(1-a))**power;g=-depth*(1-np.exp(-u/.05))+(1+depth)*rise;add(5,g,dict(form='dip_then_rise',onset=a,depth=depth,power=power,parameters=4))
 for power in [1.5,2.,3.,4.,6.]:
  for a in [0.,.1,.25,.4]:
   g=(np.maximum(u-a,0)/(1-a))**power;add(6,g,dict(form='late_power_rise',power=power,onset=a,parameters=3))
 for peak in [.2,.35,.5,.65,.8]:
  for end in [-.5,0.,.3,.6]:
   g=np.where(u<=peak,u/peak,1+(end-1)*(u-peak)/(1-peak));add(7,g,dict(form='rise_then_fall',peak=peak,end=end,parameters=3))
 for k in [0.,2.,5.]:
  g=-u if k==0 else -(1-np.exp(-k*u))/(1-np.exp(-k));add(8,g,dict(form='decline',k=k,parameters=2))
 return np.array(curves),np.array(labels),params

def assign(feat,Y,weak_threshold=.2,complexity_penalty=.0006):
 L,cl,pars=library();L=L-L.mean(axis=1,keepdims=True);Lnorm=(L*L).sum(axis=1);Yc=Y-Y.mean(axis=1,keepdims=True);dots=Yc@L.T;coef=np.maximum(dots/Lnorm,0);res=np.maximum((Yc*Yc).mean(axis=1)[:,None]-2*coef*dots/101+coef**2*Lnorm/101,0)
 penalties=np.array([p['parameters'] for p in pars])*complexity_penalty;scores=res+penalties
 n=len(feat);valid=np.ones((n,len(L)),bool);ep=feat.early_shape_rate.to_numpy();mp=feat.middle_shape_rate.to_numpy();lp=feat.late_shape_rate.to_numpy();end=feat.shape_end_v3.to_numpy();dip=feat.minimum_dp_mpa.to_numpy();scale=feat.scale_mpa.to_numpy();eps=feat.noise_detection_mpa.to_numpy();tr=feat.trough_fraction_v3.to_numpy();pk=feat.peak_fraction_v3.to_numpy();rr=feat.return_fraction.to_numpy()
 gates={2:(ep>.25)&(lp<.55*np.maximum(ep,mp))&(end>.35),3:(mp>1.15*np.maximum(ep,lp))&(mp>.35)&(end>.35),4:(end>.25)&(ep>-.15)&(mp>0)&(lp>-.15),5:(dip<-np.maximum(eps,.08*scale))&(tr>.025)&(tr<.65)&((feat.end_rise_mpa.to_numpy()-dip)>.4*scale)&(lp>.25),6:(lp>1.25*np.maximum(ep,.05))&(end>.35),7:(rr>.3)&(pk>.05)&(pk<.9)&(lp<-.15),8:(end<-.25)&(feat.positive_rise_p95_mpa.to_numpy()<.5*scale)}
 for k,g in gates.items():valid[:,cl==k]=g[:,None]
 scores[~valid]=np.inf;best=np.argmin(scores,axis=1);bestscore=scores[np.arange(n),best];typ=cl[best].copy();norm_rmse=np.sqrt(res[np.arange(n),best]);no=np.isinf(bestscore)|(norm_rmse>.3);typ[no]=9
 weak=(feat.amplitude_mpa.to_numpy()<=weak_threshold)&(feat.peak_rise_mpa.to_numpy()<=max(.3,weak_threshold*1.5));typ[weak]=1
 second=np.partition(scores,1,axis=1)[:,1];margin=(second-bestscore)/np.maximum(second,1e-10);margin[~np.isfinite(margin)]=0
 # Template-to-template margin should be between types, not neighboring parameters within the same type.
 byclass=np.column_stack([np.min(scores[:,cl==k],axis=1) for k in [2,3,4,5,6,7,8]]);order=np.sort(byclass,axis=1);gap=(order[:,1]-order[:,0])/np.maximum(order[:,1],1e-10);gap[~np.isfinite(gap)]=0
 fitcurves=coef[np.arange(n),best,None]*L[best]+Y.mean(axis=1)[:,None]
 return typ,norm_rmse,gap,best,fitcurves,byclass

def run():
 d=pd.read_pickle(OLD/'final_labels.pkl');raw=pd.read_pickle(OLD/'raw_curves.pkl');rows=[];X=[];Y=[]
 for i,r in d.iterrows():
  f,x,y=extract(raw[int(r.matrix_row)],r);rows.append(f);X.append(x);Y.append(y)
 f=pd.DataFrame(rows);X=np.array(X);Y=np.array(Y);d=pd.concat([d.reset_index(drop=True),f],axis=1);np.save(B/'pressure_delta101.npy',X);np.save(B/'shape101.npy',Y)
 typ,err,gap,best,pred,byclass=assign(f,Y);d['morphology_code_v3']=typ;d['morphology_v3']=pd.Series(typ).map(NAMES);d['template_normalized_rmse']=err;d['template_margin_between_types']=gap;d['best_template_index']=best;d['morphology_uncertain']=(err>.2)|((gap<.1)&~np.isin(typ,[1,8]))|(typ==9);d['reference_type']=typ<=5
 np.save(B/'fitted_shape101.npy',pred);pd.DataFrame(byclass,columns=['quick','sigmoid','linear','dip_rise','accelerating','pulse','decline']).to_csv(B/'template_errors.csv',index=False)
 # Separate pressure-response intensity. All components increase monotonically with measured rise strength.
 # Log transforms suppress extreme operational jumps. PCA/SHAP are not risk calibration.
 cols=['positive_rise_p95_mpa','positive_auc_mean_mpa','positive_auc_mpa_min','positive_duration_min','rising_duration_min','max_segment_rate_mpa_min']
 d['max_segment_rate_mpa_min']=np.maximum(d[['early_rate_mpa_min_v3','middle_rate_mpa_min_v3','late_rate_mpa_min_v3']].max(axis=1),0)
 vals=d[cols].to_numpy(float);v=np.log1p(vals/np.array([1,1,100,60,60,.01]));p90=np.quantile(v,.9,axis=0);z=np.clip(v/np.maximum(p90,1e-6),0,1.5)
 weights=np.array([.35,.2,.1,.1,.05,.2]);score=z@weights;d['response_strength_score']=score
 # 1D KMeans cut points, ordered by increasing intensity; three grades are a requested descriptive partition.
 km=KMeans(3,n_init=30,random_state=SEED).fit(score[:,None]);centers=np.sort(km.cluster_centers_.ravel());cuts=(centers[:-1]+centers[1:])/2;level=np.searchsorted(cuts,score)+1
 d['response_strength_grade']=pd.Series(level).map({1:'弱',2:'中',3:'强'});d['risk_grade_provisional']=d.response_strength_grade+'（暂定）';d['risk_validated']=False
 d['risk_mapping_review_required']=(typ>=7)|d.morphology_uncertain|d.operational_flag|d.alternate_record_flag|d.fragment_only|~d.event_identity_known|~d.well_identity_known
 d['risk_training_candidate']=~d.risk_mapping_review_required
 for j,c in enumerate(cols):d['strength_component_'+c]=z[:,j];d['strength_contribution_'+c]=z[:,j]*weights[j]
 d.to_pickle(B/'results.pkl');d.to_csv(O/'01_全量曲线形态与强度标签.csv',index=False,encoding='utf-8-sig');pd.crosstab(d.morphology_v3,d.response_strength_grade).reindex(columns=['弱','中','强']).to_csv(O/'02_形态与强度交叉表.csv',encoding='utf-8-sig')
 summary=d.groupby(['morphology_code_v3','morphology_v3']).agg(n=('sample_id','size'),platforms=('platform_id','nunique'),uncertain=('morphology_uncertain','sum'),amplitude=('amplitude_mpa','median'),rise=('positive_rise_p95_mpa','median'),early=('early_rate_mpa_min_v3','median'),middle=('middle_rate_mpa_min_v3','median'),late=('late_rate_mpa_min_v3','median'),auc=('positive_auc_mpa_min','median'),rise_time=('rising_duration_min','median'),fit_error=('template_normalized_rmse','median'));summary.to_csv(O/'03_形态特征汇总.csv',encoding='utf-8-sig')
 _,_,pars=library();(B/'template_library.json').write_text(json.dumps(pars,indent=2),encoding='utf-8')
 (B/'config.json').write_text(json.dumps(dict(seed=SEED,mode='reference-guided phenotype classification plus separate response intensity',n=len(d),morphology_names=NAMES,weak_threshold_mpa=.2,weak_peak_limit_mpa=.3,early_middle_late='equal thirds of observed window',template_penalty_per_parameter=.0006,template_reject_rmse=.3,intensity_features=cols,intensity_log_scales=[1,1,100,60,60,.01],intensity_p90=p90.tolist(),intensity_weights=weights.tolist(),intensity_centers=centers.tolist(),intensity_cuts=cuts.tolist(),risk_status='provisional pressure-response grades; not validated harm or operational thresholds'),ensure_ascii=False,indent=2),encoding='utf-8')
 print(summary.to_string(),flush=True);print('INTENSITY',d.response_strength_grade.value_counts().to_dict(),'CUTS',cuts,flush=True);print('CROSS',pd.crosstab(d.morphology_v3,d.response_strength_grade).to_string(),flush=True)
if __name__=='__main__':run()
