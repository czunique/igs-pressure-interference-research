from pathlib import Path
import json,sys,re,collections
import pandas as pd,numpy as np
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');OLD=B.parent/'天然裂缝特征_20260916';OUT=Path('F:/论文库/IGS/实验/结论/地质工程因素_20260916');OUT.mkdir(exist_ok=True)
d=pd.read_pickle(B/'stage_factors.pkl');cl=pd.read_pickle(B/'classification_linkage.pkl');leg=pd.read_pickle(B/'legacy_candidates.pkl');eng=pd.read_pickle(B/'engineering_candidates.pkl');layers=pd.read_pickle(B/'layers.pkl');lin=pd.read_pickle(B/'factor_lineage.pkl');conf=pd.read_pickle(B/'factor_conflicts.pkl');sourceissues=pd.read_pickle(B/'source_issues.pkl');tr=pd.read_pickle(OLD/'trajectories.pkl');qa=pd.read_csv(OLD/'trajectory_checks.csv');nf=pd.read_pickle(OLD/'pair_features.pkl');st=pd.read_csv(OLD/'stages.csv')
K=['platform','well','stage'];PK=K+['monitor_platform','monitor_well']
# Discard only bogus pressure-derived stage identities. Keep all independently documented stages.
knownkeys=set(map(tuple,st[K].to_numpy()))|set(map(tuple,eng[K].drop_duplicates().to_numpy()))|set(map(tuple,leg[K].drop_duplicates().to_numpy()))|set(map(tuple,cl[cl.join_valid][K].drop_duplicates().to_numpy()))
d=d[[tuple(r) in knownkeys for r in d[K].to_numpy()]].copy();have=set(map(tuple,d[K].to_numpy()))
for k in knownkeys-have:d.loc[len(d)]={**dict(zip(K,k)),'井段键':f'{k[0]}-{int(k[1])} | 第{int(k[2])}段','地质计算状态':'缺分段深度','解释覆盖率':0}
d=d.sort_values(K).reset_index(drop=True);assert not d.duplicated(K).any()
# Attach compact source IDs rather than duplicating long source strings across every pair.
registry={}
def sid(text):
 if not isinstance(text,str) or not text.strip():return ''
 if text not in registry:registry[text]=f'S{len(registry)+1:05d}'
 return registry[text]
lin['来源编号']=lin.source.map(sid);geo_sources={};eng_sources={}
for key,g in lin.groupby('key'):
 eng_sources[key]=';'.join(sorted(set(g[g.scope=='工程']['来源编号'])))
for c in ['地质来源','射孔_地质来源','分段来源']:
 if c in d:d[c+'编号']=d[c].map(sid)
d['工程来源编号']=d['井段键'].map(eng_sources).fillna('')
# Legacy spacing and holes per cluster supplement source-stage data only if consistent.
for k,q in leg.groupby(K):
 idx=d.index[(d.platform==k[0])&(d.well==k[1])&(d.stage==k[2])]
 for f in ['簇间距_m','单簇孔数']:
  if f not in q:continue
  vals=q[f].dropna();vals=vals[(vals>=0)&(vals<=1000)]
  if len(vals) and vals.max()-vals.min()<1e-4:
   d.loc[idx,f]=float(vals.iloc[0])
   for ii in idx:d.loc[ii,'工程来源编号']=';'.join(filter(None,[d.loc[ii,'工程来源编号'],sid(q.iloc[0].source)]))
# Check unit-sensitive actual strengths against their reported numerator and length; retain source values.
d['用液强度复核差']=np.nan;d['加砂强度复核差']=np.nan
for i,r in d.iterrows():
 length=r.get('段长_m',np.nan)
 if pd.notna(length) and length>0:
  if pd.notna(r.get('总砂量_t')) and pd.notna(r.get('加砂强度_t_m')):d.loc[i,'加砂强度复核差']=r['加砂强度_t_m']-r['总砂量_t']/length
  # Net-vs-total fluid conventions differ, so do not force an equality for fluid intensity.
  if pd.notna(r.get('净液量_m3')) and pd.notna(r.get('用液强度_m3_m')):d.loc[i,'用液强度复核差']=r['用液强度_m3_m']-r['净液量_m3']/length
# Main paired population: confirmed wells within each platform plus explicitly observed cross-platform pairs.
wellset=collections.defaultdict(set)
for frame,pc,wc in [(d,'platform','well'),(tr,'platform','well'),(cl[cl.join_valid],'monitor_platform','monitor_well'),(leg,'monitor_platform','monitor_well')]:
 for p,w in frame[[pc,wc]].dropna().drop_duplicates().itertuples(index=False,name=None):wellset[p].add(int(w))
pairs=set()
for p,w,s in d[K].itertuples(index=False,name=None):
 for n in wellset[p]:
  if int(w)!=n:pairs.add((p,int(w),int(s),p,n))
for frame in [cl[cl.join_valid],leg]:
 for p,w,s,mp,mw in frame[PK].dropna().itertuples(index=False,name=None):
  if (p,int(w))!=(mp,int(mw)):pairs.add((p,int(w),int(s),mp,int(mw)))
cm={k:g for k,g in cl[cl.join_valid].groupby(PK)};lm={k:g for k,g in leg.groupby(PK)};dm={tuple(r[c] for c in K):r for _,r in d.iterrows()};fm={(r.platform,int(r.source_well),int(r.stage),r.platform,int(r.neighbor)):r for _,r in nf.iterrows() if str(r.source_well).isdigit() and str(r.stage).isdigit() and str(r.neighbor).isdigit()}
tm={k:g.sort_values('md').drop_duplicates('md') for k,g in tr.groupby(['platform','well'])};bad={k:bool(g.geometry_review_required.iloc[0]) for k,g in qa.groupby(['platform','well'])}
def east(v):return v-18000000 if 18000000<v<19000000 else v
def geometry(k,r):
 a=tm.get(k[:2]);b=tm.get((k[3],k[4]));mid=(r.get('段顶测深_m',np.nan)+r.get('段底测深_m',np.nan))/2
 if a is None or b is None or pd.isna(mid):return np.nan,'缺井轨迹或分段深度'
 if bad.get(k[:2],True) or bad.get((k[3],k[4]),True):return np.nan,'井轨迹一致性待核查'
 a=a[a.east.notna()&a.north.notna()];b=b[b.east.notna()&b.north.notna()]
 if len(a)<2 or len(b)<2 or mid<a.md.min() or mid>a.md.max():return np.nan,'缺坐标或段深超出轨迹范围'
 x=east(float(np.interp(mid,a.md,a.east)));y=float(np.interp(mid,a.md,a.north));bxy=np.column_stack([b.east.map(east),b.north]);u=bxy[:-1];v=bxy[1:]-u;den=np.sum(v*v,axis=1);t=np.divide(np.sum((np.array([x,y])-u)*v,axis=1),den,out=np.zeros(len(den)),where=den>0);closest=u+np.clip(t,0,1)[:,None]*v;distance=float(np.min(np.linalg.norm(closest-[x,y],axis=1)))
 return distance,'段中点至邻井轨迹最短平面距离；坐标基准待确认'
main=[];comparison=[]
core=['段顶测深_m','段底测深_m','测深段长_m','段长_m','簇间距_m','簇数','孔数','单簇孔数','加砂强度_t_m','用液强度_m3_m','排量_m3_min','最低排量_m3_min','最高排量_m3_min','总砂量_t','净液量_m3','总液量_m3','孔隙度_pct','含水饱和度_pct','总有机碳_pct','含气量_压力系数2_m3_t','杨氏模量_GPa','泊松比','最大主应力_MPa','最小主应力_MPa','水平应力差_MPa','垂向应力_MPa','破裂压力_MPa','平面应变模量_GPa','剪切模量_GPa','体积模量_GPa','矿物脆性指数_pct','脆性矿物_含碳酸盐_pct','脆性矿物_不含碳酸盐_pct','泊杨脆性指数_pct','解释覆盖率','杨氏模量_GPa_覆盖率','最小主应力_MPa_覆盖率']
for count,k in enumerate(sorted(pairs)):
 r=dm.get(k[:3],pd.Series(dtype=object));cq=cm.get(k);lq=lm.get(k);key=f'{k[0]}-{k[1]}—第{k[2]}段—{k[3]}-{k[4]}'
 label='暂无可关联曲线';orig='';ids='';classissue='';strength=''
 if cq is not None:
  types=sorted(set(cq.morphology_code));orig=';'.join(sorted(set(cq.original_class)));ids=';'.join(cq.sample_id);strength=';'.join(sorted(set(cq.response_strength)))
  if len(types)==1:label=cq.five_class.iloc[0];classissue='形态边界待复核' if cq.morphology_uncertain.any() else ''
  else:label='多记录分类不一致';classissue='同一井段邻井存在多个曲线分类，未投票强行合并'
 q={'压裂井-压裂段-压窜井':key,'平台':k[0],'压窜分类_目标5类':label,'压裂井':f'{k[0]}-{k[1]}','压裂段':k[2],'压窜井':f'{k[3]}-{k[4]}','记录范围':'已有曲线' if cq is not None else '原统计记录' if lq is not None else '同平台候选邻井','原始曲线分类':orig,'压力响应强度_非验证风险':strength,'关联曲线数':0 if cq is None else len(cq),'分类说明':classissue}
 q.update({f:r.get(f,np.nan) for f in core})
 # Pair-specific engineering from client workbook takes precedence, but conflicting repeated values stay flagged.
 pair_sources=[];pair_conf=[]
 if lq is not None:
  for f in ['井距_m','段长_m','簇间距_m','簇数','孔数','单簇孔数','加砂强度_t_m','用液强度_m3_m','排量_m3_min']:
   if f not in lq:continue
   vals=lq[f].dropna();vals=vals[vals>=0]
   if len(vals) and vals.max()-vals.min()<=max(.0001,abs(vals.mean())*.005):q[f]=float(vals.iloc[0]);pair_sources.append(sid(lq.iloc[0].source))
   elif len(vals):pair_conf.append(f)
  for oldfield,newfield in [('原统计_杨氏模量_GPa','杨氏模量_GPa'),('原统计_最小主应力_MPa','最小主应力_MPa'),('原统计_水平应力差_MPa','水平应力差_MPa')]:
   vals=lq[oldfield].dropna() if oldfield in lq else pd.Series(dtype=float)
   if len(vals):comparison.append({'组合键':key,'指标':newfield,'本次重算':r.get(newfield,np.nan),'原统计值':float(vals.iloc[0]),'差值_重算减原值':r.get(newfield,np.nan)-float(vals.iloc[0]),'重算有效覆盖率':r.get(newfield+'_覆盖率',np.nan),'原统计来源编号':sid(lq.iloc[0].source)})
 dist,diststatus=geometry(k,r);q['重算井距_段中点平面_m']=dist;q['井距说明']=diststatus
 f=fm.get(k);q['天然裂缝逼近角_deg']=np.nan if f is None else f.theta_deg;q['天然裂缝解释线总长_m']=np.nan if f is None else f.line_length_m;q['天然裂缝发育程度_km_km2']=np.nan if f is None else f.development_km_km2;q['天然裂缝状态']='缺可用层位裂缝特征' if f is None else f.status
 q['地质计算状态']=r.get('地质计算状态','缺分段深度');q['分段冲突']=bool(r.get('分段冲突',False)) if pd.notna(r.get('分段冲突')) else False;q['工程冲突字段']=';'.join(filter(None,[str(r.get('工程冲突字段','')),','.join(pair_conf)]));q['井段键']=r.get('井段键',f'{k[0]}-{k[1]} | 第{k[2]}段');q['工程来源编号']=';'.join(filter(None,[r.get('工程来源编号',''),*sorted(set(pair_sources))]));q['地质来源编号']=r.get('地质来源编号','');q['曲线样本ID']=ids
 main.append(q)
 if count%5000==0:print('PAIRS',count,flush=True)
pd.DataFrame(main).to_pickle(B/'pair_factors.pkl');d.to_pickle(B/'stage_factors_final.pkl');pd.DataFrame(comparison).to_pickle(B/'legacy_comparison.pkl')
# Detailed reproducibility stays in process, with compact workbook source registry.
lin[['key','scope','field','value','coverage','method','来源编号']].to_csv(B/'逐项计算与来源.csv',index=False,encoding='utf-8-sig')
sources=pd.DataFrame([{'来源编号':v,'来源定位':k} for k,v in registry.items()]);sources.to_pickle(B/'source_registry.pkl');sources.to_csv(B/'来源定位.csv',index=False,encoding='utf-8-sig')
pair=pd.DataFrame(main);summary=[]
for p,g in d.groupby('platform'):
 pg=pair[pair.平台==p];summary.append({'平台':p,'井数':g.well.nunique(),'井段数':len(g),'地质覆盖至少95%井段':int((g['解释覆盖率']>=.95).sum()),'有杨氏模量井段':int(g['杨氏模量_GPa'].notna().sum()),'有加砂强度井段':int(g['加砂强度_t_m'].notna().sum()),'有用液强度井段':int(g['用液强度_m3_m'].notna().sum()),'组合数':len(pg),'有曲线组合数':int((pg.关联曲线数>0).sum()),'目标5类组合数':int(pg['压窜分类_目标5类'].isin(['涨幅微弱型','快速稳定型','渐进上升型','稳定缓升型','先降后升型']).sum())})
pd.DataFrame(summary).to_pickle(B/'platform_coverage.pkl')
stats={'platforms':d.platform.nunique(),'wells':len(d[['platform','well']].drop_duplicates()),'stages':len(d),'pairs':len(pair),'class_counts':pair['压窜分类_目标5类'].value_counts().to_dict(),'geology_stages_95':int((d['解释覆盖率']>=.95).sum()),'curve_records':len(cl),'linked_curve_records':int(cl.join_valid.sum()),'unlinked_curve_records':int((~cl.join_valid).sum()),'source_registry':len(sources),'engineering_conflicts':len(conf),'source_issues':len(sourceissues)}
(B/'delivery_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats,ensure_ascii=False))
