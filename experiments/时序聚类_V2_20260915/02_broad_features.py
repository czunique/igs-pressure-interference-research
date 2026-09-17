"""V2 broad curve inclusion, duplicate reconciliation, native-time morphology features."""
import sys,re,json,hashlib,warnings
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore')
OLD=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915');B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915')

def best_window(c,events):
 t=c['t'];pot=events.get((c['platform'],c['source'],c['stage']),[]) if c['source'] and c['stage'] else []
 scored=[]
 for e in pot:
  if c['absolute'] and e['absolute']:a,b=e['t0'],e['t1'];mode='absolute'
  else:
   shift=round((t[0]-e['t0'])/86400)*86400;a,b=e['t0']+shift,e['t1']+shift;mode='same_stage_clock'
  inter=max(0,min(t[-1],b)-max(t[0],a));cov=inter/(b-a)
  if cov>=.45:scored.append((cov,-e['dt'],a,b,mode,e))
 if not scored:return None
 z=max(scored,key=lambda z:(z[0]>=.9,z[1],z[0]));return {'a':z[2],'b':z[3],'mode':z[4],'event':z[5],'coverage':z[0]}

def features(t,p,c,w):
 native_dt=float(np.median(np.diff(t)));first,last=t[0],t[-1];pre_t=np.array([]);pre_p=np.array([])
 if w:
  sel=(t>=w['a'])&(t<=w['b']);pre=(t<w['a'])&(t>=w['a']-600);pre_t,pre_p=t[pre],p[pre]
  if sel.sum()>=6:t,p=t[sel],p[sel]
  else:w=None
 if len(t)<6:return None,'少于6个观测点'
 span=t[-1]-t[0];fragment=False;original_n=len(t)
 native=float(np.median(np.diff(t)))
 if span>18*3600 or np.max(np.diff(t))>max(4*native,.35*span):
  breaks=np.flatnonzero(np.diff(t)>max(4*native,1800))+1
  segments=[z for z in np.split(np.arange(len(t)),breaks) if len(z)>=6 and 300<=t[z[-1]]-t[z[0]]<=18*3600]
  if segments:
   seg=max(segments,key=lambda z:t[z[-1]]-t[z[0]]);t,p=t[seg],p[seg];span=t[-1]-t[0];fragment=True
 if span<300:return None,'观测时长小于5分钟'
 if span>18*3600:return None,'连续记录超过18小时且无法可靠切分事件'
 valid=(p>=0)&(p<=150);bad=1-valid.mean();t,p=t[valid],p[valid]
 if len(t)<6 or bad>.2:return None,'有效压力不足或压力单位待核对'
 gaps=np.diff(t);dt=float(np.median(gaps));span=t[-1]-t[0]
 if span<=0:return None,'无有效时长'
 if np.max(gaps)>max(4*dt,.35*span):return None,'超长缺口超过可辨认形态范围'
 if np.max(p)<.01:return None,'全零占位压力'
 # Resample to native-or-coarser observations before smoothing; derivatives never use 101-point upsampling.
 step=max(30.,dt);tg=np.arange(t[0],t[-1]+.01,step)
 if len(tg)<6:tg=np.linspace(t[0],t[-1],len(t))
 pg=np.interp(tg,t,p);smooth=median_filter(pg,size=3 if len(pg)>=12 else 1,mode='nearest')
 win=min(len(pg)//2*2-1,max(5,int(round(300/step))|1));win=max(3,win)
 if win>=5 and len(pg)>=win:smooth=savgol_filter(smooth,win,2,mode='interp')
 init=float(np.median(smooth[:max(2,int(np.ceil(.05*len(smooth))))]))
 base=init;baseline='observed_start_proxy';baseline_stable=False
 if len(pre_t)>=3 and pre_t[-1]-pre_t[0]>=60:
  tail=pre_t>=max(pre_t[0],(w['a'] if w else t[0])-180);q=pre_p[tail]
  if len(q)>=3 and np.quantile(q,.9)-np.quantile(q,.1)<=1:
   base=float(np.median(q));baseline='pre_pump_last180s';baseline_stable=True
 noise=float(1.4826*np.median(np.abs(np.diff(pg)-np.median(np.diff(pg))))/np.sqrt(2)) if len(pg)>2 else .01
 y=smooth-base;u=(tg-tg[0])/(tg[-1]-tg[0]);target=np.linspace(0,1,101);x=np.interp(target,u,y)
 delta=smooth-init;amp=float(np.quantile(delta,.95)-np.quantile(delta,.05));floor=max(.2,3*noise);shape=(x-x[0])/max(amp,floor)
 minutes=(tg-tg[0])/60;rates=np.gradient(smooth,minutes);accel=np.gradient(rates,minutes)
 def slope(a,b):
  ix=(u>=a)&(u<=b)
  return float(np.polyfit(minutes[ix],smooth[ix],1)[0]) if ix.sum()>=3 else float((np.interp(b,u,smooth)-np.interp(a,u,smooth))/((b-a)*(minutes[-1]-minutes[0])))
 e,m,l=slope(0,.25),slope(.25,.75),slope(.75,1)
 q10,q25,q50,q75,q90=np.interp([.1,.25,.5,.75,.9],u,delta)
 end=float(np.median(y[-max(2,int(np.ceil(.05*len(y)))):]))
 peak=float(np.max(y));trough=float(np.min(y));peak_initial=float(np.max(delta));trough_initial=float(np.min(delta));spanamp=max(amp,floor)
 threshold=max(.1,3*noise);hit=(y if baseline_stable and w else delta)>threshold;on=None
 for j in np.flatnonzero(hit):
  z=(tg>=tg[j])&(tg<=tg[j]+max(60,2*step))
  if z.sum()>=2 and (tg[z][-1]-tg[j])>=min(60,step) and hit[z].all():on=float(minutes[j]);break
 delay=on+(tg[0]-w['a'])/60 if on is not None and w and baseline_stable else np.nan
 norm_rate=span/60/spanamp;norm_acc=(span/60)**2/spanamp
 rec={'fragment_only':fragment,'fragment_discarded_points':original_n-len(t),'native_n':len(t),'native_dt_s':dt,'duration_min':span/60,'max_gap_s':float(np.max(gaps)),'baseline_mpa':base,'baseline_method':baseline,'baseline_noise_mpa':noise,'pressure_peak_mpa':float(np.max(p)),'amplitude_range_mpa':amp,'rise_peak_mpa':peak,'drop_min_mpa':trough,'end_dp_mpa':end,'initial_relative_rise_mpa':peak_initial,'initial_relative_drop_mpa':trough_initial,'initial_10pct_dp_mpa':q10,'early_25pct_dp_mpa':q25,'middle_50pct_dp_mpa':q50,'late_75pct_dp_mpa':q75,'tail_90pct_dp_mpa':q90,'initial_rate_mpa_min':slope(0,.1),'early_rate_mpa_min':e,'middle_rate_mpa_min':m,'tail_rate_mpa_min':l,'max_rate_mpa_min':float(np.quantile(rates,.95)),'min_rate_mpa_min':float(np.quantile(rates,.05)),'max_acceleration_mpa_min2':float(np.quantile(accel,.95)),'min_acceleration_mpa_min2':float(np.quantile(accel,.05)),'early_to_late_acceleration_mpa_min2':(l-e)/(span/60*.75),'observed_onset_min':on,'true_pump_response_delay_min':delay,'onset_reference':'pump' if np.isfinite(delay) else 'observed_window_only','onset_censored':on is None,'onset_fraction':on/(span/60) if on is not None else 1.,'peak_time_fraction':float(u[np.argmax(y)]),'trough_time_fraction':float(u[np.argmin(y)]),'area_mean_mpa':float(np.trapezoid(y,tg)/span),'recovery_fraction':float((peak-end)/max(peak-trough,.2)),'early_slope_shape':e*norm_rate,'middle_slope_shape':m*norm_rate,'late_slope_shape':l*norm_rate,'early_late_slope_change_shape':(l-e)*norm_rate,'acceleration_p95_shape':float(np.quantile(accel,.95))*norm_acc,'acceleration_p05_shape':float(np.quantile(accel,.05))*norm_acc,'dip_fraction':max(0,-trough_initial)/spanamp,'end_peak_ratio':float((end-y[0])/max(abs(peak-y[0]),.2)),'native_rate_timescale_min':step/60,'acceleration_timescale_min':max(win*step/60,2*step/60),'low_resolution':dt>120,'max_native_jump_mpa':float(np.max(np.abs(np.diff(p)))),'operational_flag':bool(np.min(p)<1 and np.max(np.abs(np.diff(p)))>5),'nonstationary_baseline':bool(len(pre_p)>3 and np.quantile(pre_p,.9)-np.quantile(pre_p,.1)>1),'t0_observed_s':float(t[0]),'t1_observed_s':float(t[-1])}
 for a in [10,25,50,75,90]:rec[f'shape_level_{a}']=float(np.interp(a/100,target,shape))
 if len(y)>0:rec['shape_end']=float(shape[-1])
 return (rec,x,shape,{'t':t,'p':p,'smooth_t':tg,'smooth_p':smooth,'pre_t':pre_t,'pre_p':pre_p}),''

def main():
 records=pd.read_pickle(B/'pressure_all.pkl');pw=pd.read_csv(OLD/'pump_windows.csv',dtype={'source':str,'stage':str});events=defaultdict(list)
 for e in pw.to_dict('records'):events[(e['platform'],e['source'],e['stage'])].append(e)
 groups=defaultdict(list);audit=[]
 for r in records:
  for i,c in enumerate(r['channels']):
   provenance={'file':r['path'],'sheet':c['sheet'],'column':c['column'],'platform':c['platform'],'source':c['source'],'stage':c['stage'],'monitor':c['monitor'],'monitor_platform':c['monitor_platform'],'mapping':c['mapping'],'header':c['header'],'unit':c['unit'],'channel_id':hashlib.sha1((r['path']+'|'+c['sheet']+'|'+str(c['column'])).encode()).hexdigest()[:14]}
   if c.get('not_timeseries'):audit.append(provenance|{'status':'阶段统计而非时序','reason':'各施工段统计指标，不能拼为连续压力曲线'});continue
   if any(z in c['header'] for z in ['砂','排量','液量','温度']):audit.append(provenance|{'status':'非压力变量','reason':'仅压力曲线进入聚类'});continue
   if c['self_pressure']:audit.append(provenance|{'status':'施工井自身压力','reason':'源井与监测井相同'});continue
   if len(c['t'])<6:audit.append(provenance|{'status':'过少观测','reason':'少于6点'});continue
   w=best_window(c,events)
   day=str(int((w['a'] if w else c['t'][0])//86400)) if c['absolute'] else 'clock_only'
   if c['source'] and c['stage']:key=(c['platform'],c['source'],c['stage'],c['monitor_platform'],c['monitor'],day)
   else:key=(c['platform'],'unknown',hashlib.sha1((day+'|'+str(int(c['t'][0]%86400/1800))+'|'+str(int(c['t'][-1]%86400/1800))).encode()).hexdigest()[:8],c['monitor_platform'],c['monitor'],day)
   groups[key].append((r,c,w,provenance))
 # Separate incompatible exports, retaining observations and marking event-level ambiguity.
 expanded={}
 for key,items in groups.items():
  items.sort(key=lambda z:(z[1]['dt'],-len(z[1]['t'])))
  cohorts=[]
  for item in items:
   cc=item[1];chosen=None
   for cohort in cohorts:
    ref=cohort[0][1];ix=(cc['t']>=ref['t'][0])&(cc['t']<=ref['t'][-1])
    if ix.sum()>=5:
     dif=float(np.median(np.abs(np.interp(cc['t'][ix],ref['t'],ref['p'])-cc['p'][ix])))
     if dif<=max(.3,.03*np.ptp(ref['p'])):chosen=cohort;break
    elif cc['source'] and cc['stage'] and cc['absolute'] and ref['absolute']:
     chosen=cohort;break
   if chosen is None:cohorts.append([item])
   else:chosen.append(item)
  for j,cohort in enumerate(cohorts):
   for item in cohort:item[3]['alternate_record_flag']=len(cohorts)>1
   expanded[key[:-1]+('variant'+str(j),key[-1])]=cohort
 groups=expanded
 print('RAW_GROUPS',len(groups),flush=True)
 samples=[];curves=[];shapes=[];raws=[]
 for key,items in groups.items():
  # Merge consistent partial exports of the same event and channel. Day-only clock translations do not alter onset.
  # Best coverage/finer observations first; overlapping conflicting exports stay as aliases with conflict audit.
  items.sort(key=lambda z:(-(z[2]['coverage'] if z[2] else .5),z[1]['dt'],-len(z[1]['t'])))
  r,c,w,prov=items[0];tt=c['t'].copy();pp=c['p'].copy();aliases=[];conflicts=[]
  for rr,cc,ww,pv in items[1:]:
   ct=cc['t'].copy();cp=cc['p'];common=(ct>=tt[0])&(ct<=tt[-1]);diff=np.median(np.abs(np.interp(ct[common],tt,pp)-cp[common])) if common.sum()>=5 else 0.
   if diff>max(.3,.03*np.ptp(pp)):
    conflicts.append(pv['channel_id']);audit.append(pv|{'status':'同事件重复记录不一致','reason':f'重叠压力中位差{diff:.3f}MPa'});continue
   outside=(ct<tt[0])|(ct>tt[-1])
   if outside.any():
    tt=np.r_[tt,ct[outside]];pp=np.r_[pp,cp[outside]];order=np.argsort(tt);tt,pp=tt[order],pp[order]
   aliases.append(pv)
  result,reason=features(tt,pp,c,w)
  if result is None:
   for *_,pvv in items:audit.append(pvv|{'status':'无法形成完整形态样本','reason':reason})
   continue
  feat,x,sh,raw=result
  sid=hashlib.sha1('|'.join(key).encode()).hexdigest()[:14]
  rawhash=hashlib.sha256(np.round(np.r_[tt,pp],5).tobytes()).hexdigest()
  stage_group='|'.join([c['platform'],c['source'],c['stage'],key[-1]]) if c['source'] else c['platform']+'|'+key[-1]+'|unknown_source'
  sample={'sample_id':sid,'platform_id':c['platform'],'source_well':c['source'],'stage_id':c['stage'],'monitor_platform':c['monitor_platform'],'monitor_well':c['monitor'],'stage_group':stage_group,'pair_id':c['platform']+'|'+c['source']+'|'+c['monitor_platform']+'|'+c['monitor'],'well_identity_known':not c['monitor'].startswith('sensor:'),'event_identity_known':bool(c['source'] and c['stage']),'window_basis':w['mode'] if w else 'observed_file_window','construction_coverage':w['coverage'] if w else np.nan,'pump_file':w['event']['path'] if w else '',**prov,'rawhash':rawhash,'alias_count':len(aliases),'conflict_count':len(conflicts),'matrix_row':len(curves),**feat}
  samples.append(sample);curves.append(x);shapes.append(sh);raws.append(raw)
  audit.append(prov|{'status':'参与形态聚类候选','sample_id':sid,'reason':sample['window_basis']})
  for al in aliases:audit.append(al|{'status':'重复或补充片段已合并','sample_id':sid,'reason':'相同事件/通道且压力一致，仅计一个样本'})
 df=pd.DataFrame(samples)
 # Exact content exports with different filenames are counted once; identical zero lines are excluded upstream.
 dup=df.duplicated(['platform_id','monitor_platform','monitor_well','rawhash']);df['included']=~dup
 df['duplicate_sample_of']=''
 for h,z in df.groupby(['platform_id','monitor_platform','monitor_well','rawhash']):
  if len(z)>1:df.loc[z.index[1:],'duplicate_sample_of']=z.iloc[0].sample_id
 df.to_pickle(B/'broad_samples.pkl');pd.to_pickle(raws,B/'raw_curves.pkl');np.save(B/'curves101.npy',np.array(curves));np.save(B/'shapes101.npy',np.array(shapes))
 df.to_csv(O/'04_全量曲线特征与质量.csv',index=False,encoding='utf-8-sig');pd.DataFrame(audit).to_csv(O/'03_逐通道利用去向.csv',index=False,encoding='utf-8-sig')
 z=df[df.included];print('SAMPLES',len(z),'PLATFORMS',z.platform_id.nunique(),'EVENT_KNOWN',z.event_identity_known.sum(),'LOWFREQ',z.low_resolution.sum(),'BASELINES',z.baseline_method.value_counts().to_dict(),flush=True);print('WINDOWS',z.window_basis.value_counts().to_dict(),flush=True);print('AUDIT',pd.DataFrame(audit).status.value_counts().to_dict(),flush=True)
if __name__=='__main__':main()
