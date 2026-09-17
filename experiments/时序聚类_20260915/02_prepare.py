"""Construct independent source-stage-monitor samples and explicit quality cohorts."""
import sys,json,hashlib,warnings,re
from pathlib import Path
import numpy as np,pandas as pd
from scipy.ndimage import median_filter
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore')
BASE=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915');OUT=Path('F:/论文库/IGS/实验/结论/时序聚类_20260915')

def normalize(r):
 for c in r['channels']:
  # Legacy first-pass pandas 3 datetime64[us] cache is explicitly migrated to seconds.
  if c['absolute'] and np.nanmedian(c['t'])<1e8:
   c['t']=np.round(c['t']*1000,4);c['dt']=float(np.median(np.diff(c['t'])))
 return r

def runs(mask):
 x=np.diff(np.r_[False,mask,False].astype(int));return zip(np.where(x==1)[0],np.where(x==-1)[0])

def pump_event(r,c):
 t,p=c['t'],c['p'];dt=c['dt']
 if dt<=0 or np.quantile(p,.95)<.5:return None
 th=max(.5,float(np.quantile(p,.95))*.05)
 ranges=[(a,b) for a,b in runs(p>=th) if t[b-1]-t[a]>=60 and b-a>=2]
 if not ranges:return None
 a,b=ranges[0][0],ranges[-1][1]-1
 if not 300<=t[b]-t[a]<=43200:return None
 return {'platform':r['platform'],'source':r['source'],'stage':r['stage'],'path':r['path'],'sheet':c['sheet'],'t0':float(t[a]),'t1':float(t[b]),'absolute':bool(c['absolute']),'dt':dt,'pump_threshold':th,'q95':float(np.quantile(p,.95)),'t':t,'q':p}

def make_sample(r,c,e,alignment):
 t,p=c['t'].copy(),c['p'].copy();start,end=e['t0'],e['t1']
 if alignment!='absolute':
  # Day translation only: preserve observed clock time and all within-event lags.
  shift=round((start-t[0])/86400)*86400;t+=shift
 else:shift=0
 dt=float(np.median(np.diff(t)));dur=end-start
 rawvalid=(p>=0)&(p<=150);invalid_pressure=int((~rawvalid).sum());t=t[rawvalid];p=p[rawvalid]
 key=f"{e['platform']}|{e['source']}|{e['stage']}|{c['monitor']}"
 sid=hashlib.sha1(key.encode()).hexdigest()[:14]
 rec={'sample_id':sid,'platform_id':e['platform'],'source_well':e['source'],'stage_id':e['stage'],'episode_id':'whole_stage','monitor_well':c['monitor'],'physical_pair_id':e['platform']+'|'+'-'.join(sorted([e['source'],c['monitor']])),'stage_group':e['platform']+'|'+e['source']+'|'+e['stage'],'pressure_file':r['path'],'pressure_sheet':c['sheet'],'pressure_header':c['label'],'pressure_sha256':r['sha256'],'pump_file':e['path'],'pump_sheet':e['sheet'],'alignment':alignment,'clock_day_shift_s':shift,'t0_s':start,'t1_s':end,'duration_min':dur/60,'native_dt_s':dt,'invalid_pressure_points':invalid_pressure,'duplicate_time_points':c['duplicates'],'out_of_order_points':c['backwards']}
 sel=(t>=start)&(t<=end);tt=t[sel];pp=p[sel]
 if len(tt)<10:return rec|{'qc':'exclude','reason':'施工窗内少于10点'},None,None
 coverage=max(0,min(tt[-1],end)-max(tt[0],start))/dur
 gaps=np.diff(tt);maxgap=float(max(np.max(gaps),tt[0]-start,end-tt[-1]))
 pre=(t<start)&(t>=start-600);bt=t[pre];bp=p[pre]
 baseline_good=len(bt)>=5 and bt[-1]-bt[0]>=180 and np.max(np.diff(bt))<=max(30,1.5*dt)
 if baseline_good:base=float(np.median(bp));bs='pre_pump_10min';noise=float(1.4826*np.median(np.abs(bp-base)));slope=float(np.polyfit((bt-bt[0])/60,bp,1)[0])
 else:
  baseline=pp[tt<=tt[0]+min(180,dur*.1)];base=float(np.median(baseline));bs='initial_window_proxy';noise=float(1.4826*np.median(np.abs(baseline-base)));slope=np.nan
 rec.update({'coverage':coverage,'max_gap_s':maxgap,'baseline_mpa':base,'baseline_source':bs,'baseline_points':len(bt),'baseline_span_s':bt[-1]-bt[0] if len(bt)>1 else 0,'baseline_noise_mpa':noise,'baseline_slope_mpa_min':slope})
 reasons=[]
 if coverage<.95:reasons.append('施工窗覆盖不足95%')
 if dt>30.01:reasons.append('原始采样间隔大于30秒')
 if maxgap>30.01:reasons.append('施工窗缺口大于30秒')
 if not baseline_good:reasons.append('起泵前基线不足180秒')
 if alignment=='unknown':reasons.append('施工时间不明')
 qc='primary' if not reasons else 'supplementary'
 if coverage<.8 or maxgap>max(180,3*dt) or dt>120:qc='exclude'
 if np.ptp(pp)<1e-8 and abs(np.median(pp))<.01:qc='exclude';reasons.append('全零占位或无有效压力')
 pre_span=float(np.quantile(bp,.9)-np.quantile(bp,.1)) if len(bp)>4 else np.nan
 pre_last=float(np.median(bp[bt>=start-60])) if np.any(bt>=start-60) else np.nan
 max_step=float(np.max(np.abs(np.diff(pp))))
 operational=[]
 if baseline_good and (pre_span>1 or abs(pre_last-base)>.5 or abs(slope)>.1):operational.append('起泵前基线非平稳_需工况复核')
 if np.min(pp)<1 and max_step>5 and dt<=5:operational.append('近零压力与秒级大跳变共存_需仪表或操作复核')
 if operational:
  reasons.extend(operational)
  if qc!='exclude':qc='operational_review'
 rec.update(qc=qc,reason='；'.join(reasons) if reasons else '合格',baseline_p90_p10_mpa=pre_span,baseline_last60_mpa=pre_last,max_raw_step_mpa=max_step)
 # No filling gaps exceeding QC limits in primary. Supplementary representation is separately flagged.
 width=max(1,int(round(15/max(dt,1))));width+=1-width%2
 smooth=median_filter(pp,size=width,mode='nearest');dp=smooth-base
 u=np.linspace(start,end,201);x=np.interp(u,tt,dp)
 gridstep=max(5,dt);tg=np.arange(tt[0],tt[-1]+.001,gridstep);yg=np.interp(tg,tt,dp)
 w=max(2,int(np.ceil(60/gridstep))+1)
 a60=pd.Series(yg).rolling(w,min_periods=w).median().max() if gridstep<=30 else np.nan
 threshold=max(.1,3*noise);onsets=[]
 for a,b in runs(yg>threshold):
  if tg[b-1]-tg[a]>=60:onsets.append(tg[a]-start)
 lag=onsets[0]/60 if onsets else np.nan
 rate=(np.interp(tg[:-w],tg,yg) if False else np.diff(yg)/gridstep*60)
 k=max(1,int(round(60/gridstep)));rate60=(yg[k:]-yg[:-k])/(k*gridstep/60)
 rec.update({'a60_mpa':float(a60),'peak_dp_mpa':float(np.max(dp)),'min_dp_mpa':float(np.min(dp)),'end_dp_mpa':float(np.median(dp[-max(1,w):])),'area_mean_mpa':float(np.trapezoid(dp,tt)/(tt[-1]-tt[0])),'max_rate60_mpa_min':float(np.max(rate60)) if len(rate60) else np.nan,'onset_min':lag,'onset_censored':not bool(onsets),'onset_fraction':lag/(dur/60) if onsets else 1.,'onset_threshold_mpa':threshold,'smoothing_seconds':width*dt,'flat_fraction':float(np.mean(np.diff(pp)==0))})
 for threshold in [1,5]:
  lens=[tg[b-1]-tg[a] for a,b in runs(yg>=threshold)]
  rec[f'duration_above_{threshold}_min']=max(lens,default=0)/60
 post=(t>end)&(t<=end+600)
 rec['post10min_available']=bool(post.sum()>2 and t[post][-1]>=end+540)
 rec['post10min_dp_mpa']=float(np.median(p[post][-max(1,w):])-base) if rec['post10min_available'] else np.nan
 raw={'t_min':(tt-start)/60,'p_mpa':pp,'dp_mpa':dp,'pre_t_min':(bt-start)/60,'pre_p_mpa':bp,'post_t_min':(t[post]-start)/60,'post_p_mpa':p[post],'pump_t_min':(e['t']-start)/60,'pump_q':e['q']}
 return rec,x,raw

def main():
 pressure=[normalize(r) for r in pd.read_pickle(BASE/'pressure_parsed.pkl')]
 pumps=[normalize(r) for r in pd.read_pickle(BASE/'pump_parsed.pkl')]
 candidates={};pump_log=[]
 for r in pumps:
  for c in r['channels']:
   e=pump_event(r,c)
   if e:
    key=(r['platform'],r['source'],r['stage']);candidates.setdefault(key,[]).append(e)
 events={}
 for key,es in candidates.items():
  es.sort(key=lambda e:(e['dt'],-(e['t1']-e['t0'])))
  events[key]=es[0]
  for e in es:pump_log.append({k:v for k,v in e.items() if k not in ['t','q']}|{'selected':e is es[0]})
 pd.DataFrame(pump_log).to_csv(BASE/'pump_windows.csv',index=False,encoding='utf-8-sig')
 print('PUMP_EVENTS',len(events),flush=True)
 rows=[];series=[];raws=[];unmatched=[]
 for r in pressure:
  named=re.search(r'([泸阳自]\d+H\d+)',Path(r['path']).name)
  if named and named.group(1)!=r['platform']:
   unmatched.append({'file':r['path'],'reason':'文件名平台与目录归属冲突','details':named.group(1)+' vs '+r['platform']});continue
  if not r['channels']:unmatched.append({'file':r['path'],'reason':'未解析出压力通道','details':str(r['errors'])});continue
  for c in r['channels']:
   key=(r['platform'],c.get('sheet_source') or r['source'],c.get('sheet_stage') or r['stage']);es=[]
   if key in events:es=[events[key]]
   elif not r['source'] and c['absolute']:
    es=[e for e in events.values() if e['platform']==r['platform'] and e['absolute'] and e['t0']>=c['t'][0]-30 and e['t1']<=c['t'][-1]+30 and e['source']!=c['monitor']]
    # Overlapping pump windows from different sources are composite operations, not attributable.
    es=[e for e in es if not any(o['source']!=e['source'] and min(e['t1'],o['t1'])-max(e['t0'],o['t0'])>60 for o in es)]
   if not es:unmatched.append({'file':r['path'],'sheet':c['sheet'],'monitor':c['monitor'],'reason':'无可唯一匹配施工事件','details':f'{r["source"]}/{r["stage"]}'});continue
   for e in es:
    if c['monitor']==e['source']:
     unmatched.append({'file':r['path'],'sheet':c['sheet'],'monitor':c['monitor'],'reason':'施工井自身压力通道','details':''});continue
    alignment='absolute' if c['absolute'] and e['absolute'] else 'named_stage_clock_only'
    anchoredoffset=0 if e['absolute'] else (round((c['t'][0]-e['t0'])/86400)*86400 if c['absolute'] else None)
    if anchoredoffset is not None:
     a0,a1=e['t0']+anchoredoffset,e['t1']+anchoredoffset
     concurrent=[o for o in events.values() if o['platform']==e['platform'] and o['absolute'] and o['source']!=e['source'] and min(a1,o['t1'])-max(a0,o['t0'])>60]
     if concurrent:
      reason='监测邻井同期施工' if any(o['source']==c['monitor'] for o in concurrent) else '多源同期施工_归因不唯一'
      unmatched.append({'file':r['path'],'sheet':c['sheet'],'monitor':c['monitor'],'reason':reason,'details':','.join(o['source']+'井'+o['stage']+'段' for o in concurrent)});continue
    if alignment=='absolute' and (c['t'][-1]<e['t0'] or c['t'][0]>e['t1']):
     unmatched.append({'file':r['path'],'sheet':c['sheet'],'monitor':c['monitor'],'reason':'同名井段绝对日期不重叠','details':f'pressure {c["t"][0]} pump {e["t0"]}'});continue
    rec,x,raw=make_sample(r,c,e,alignment)
    if x is not None:rec['matrix_row']=len(series);series.append(x);raws.append(raw)
    else:rec['matrix_row']=-1
    rows.append(rec)
 df=pd.DataFrame(rows)
 # Prefer primary-quality, higher coverage and finer sampling; never count repeat exports as independent samples.
 df['quality_rank']=df.qc.map({'primary':0,'supplementary':1,'operational_review':2,'exclude':3})
 df=df.sort_values(['sample_id','quality_rank','coverage','native_dt_s','baseline_span_s'],ascending=[True,True,False,True,False],na_position='last')
 df['selected']=~df.duplicated('sample_id');df['status']=np.where(df.selected,df.qc,'duplicate_export')
 df.to_csv(OUT/'01_全部候选曲线与质控.csv',index=False,encoding='utf-8-sig')
 df[df.status=='operational_review'].to_csv(OUT/'12_工况或仪表待复核.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(unmatched).to_csv(OUT/'02_未匹配及未解析记录.csv',index=False,encoding='utf-8-sig')
 np.save(BASE/'all_curves_201.npy',np.asarray(series));pd.to_pickle(raws,BASE/'event_raw_curves.pkl')
 df.to_pickle(BASE/'samples.pkl')
 print('STATUS',df.status.value_counts().to_dict(),flush=True)
 print('PRIMARY_PLATFORMS',df[df.status=='primary'].platform_id.value_counts().to_dict(),flush=True)
 print('UNMATCHED',pd.DataFrame(unmatched).reason.value_counts().to_dict(),flush=True)
 print('REASONS',df[df.selected].reason.value_counts().head(12).to_dict(),flush=True)
if __name__=='__main__':main()
