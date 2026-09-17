from pathlib import Path
import sys,re,json,pickle,collections
import numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');OLD=B.parent/'天然裂缝特征_20260916'
g=pd.read_pickle(B/'layers.pkl');perf=pd.read_pickle(B/'perforations.pkl');eng=pd.read_pickle(B/'engineering_candidates.pkl');leg=pd.read_pickle(B/'legacy_candidates.pkl');st=pd.read_csv(OLD/'stages.csv');cl=pd.read_pickle(B.parent/'时序聚类_V3_形态与强度/results.pkl')
keys=['platform','well','stage'];ef=[c for c in eng if c not in keys+['source','file','sheet','施工排量原文']];gf=[c for c in g if c not in ['platform','well','top','bottom','layer','source']]
issues=[];lineage=[];contrasts=[]
def finite(v):return isinstance(v,(int,float,np.number)) and np.isfinite(v)
def keystr(k):return f'{k[0]}-{int(k[1])} | 第{int(k[2])}段'
def numeric_identity(x):
 try:
  v=float(x);return int(v) if np.isfinite(v) and v==int(v) and v>0 else None
 except:return None
# Actual engineering candidates: most complete workbook first, then explicit filename date, retaining disagreements.
eng['completeness']=eng.groupby('file')['source'].transform('count');eng['dated']=eng.file.str.extract(r'(20\d{2}[.\-年]\d{1,2}[.\-月]\d{1,2})',expand=False).fillna('')
eng=eng.sort_values(['completeness','dated','file','sheet','source'],ascending=[False,False,True,True,True])
em={k:q for k,q in eng.groupby(keys)};lm={k:q for k,q in leg.groupby(keys)};gm={k:q for k,q in g.groupby(['platform','well'])};pm={k:q.drop_duplicates(['top','bottom']) for k,q in perf.groupby(keys)}
# All known stage identities, even when measured-depth boundaries are unavailable.
allkeys=set(map(tuple,st[keys].to_numpy()))|set(em)|set(lm)
for _,r in cl.iterrows():
 w,s=numeric_identity(r.source_well),numeric_identity(r.stage_id)
 if w and s:allkeys.add((r.platform_id,w,s))
smap={tuple(r[k] for k in keys):r.to_dict() for _,r in st.iterrows()}
def choose(q,field,key,scope):
 if q is None or field not in q:return np.nan,'',False
 v=q[q[field].apply(finite)]
 if v.empty:return np.nan,'',False
 # implausible engineering input remains in candidate audit but is not imputed.
 if scope=='工程':
  lo,hi=(0,30) if '强度_t' in field else (0,200) if '强度_m3' in field else (0,80) if '排量' in field else (0,500) if '压力' in field else (0,3000) if field=='段长_m' else (0,12000) if '段深' in field or '垂深' in field else (0,1e6)
  v=v[(v[field]>=lo)&(v[field]<=hi)]
  if v.empty:return np.nan,'',False
 conflict=float(v[field].max()-v[field].min())>max(1e-4,.005*abs(float(v[field].median())))
 if conflict:issues.append(dict(key=keystr(key),field=field,issue=scope+'多来源数值不一致',values=';'.join(f"{x[field]:g}@{x.source}" for _,x in v.head(8).iterrows())))
 r=v.iloc[0];return float(r[field]),r.source,conflict

def weighted(layers,ranges,key,scope):
 result={};total=sum(b-a for a,b in ranges);weights=collections.defaultdict(list);source_set=set();layer_w=collections.defaultdict(float);conflict_len=0
 if layers is None or total<=0:return {'解释覆盖率':0.,'层位组成':'','地质来源':''}
 # Excel first; document only fills absent values, equal-priority conflicts stay missing.
 q=layers.copy();q['priority']=q.source.str.contains(r'\.xlsx').map({True:0,False:1})
 cover=0
 for a,b in ranges:
  hits=q[(q.bottom>a)&(q.top<b)]
  points=sorted(set([a,b]+[max(a,float(v)) for v in hits.top]+[min(b,float(v)) for v in hits.bottom]))
  for x,y in zip(points[:-1],points[1:]):
   if y<=x:continue
   at=hits[(hits.top<y-1e-7)&(hits.bottom>x+1e-7)]
   if at.empty:continue
   length=y-x;cover+=length;layer_w['/'.join(sorted(set(at.layer)))]+=length
   for f in gf:
    valid=at[at[f].notna()] if f in at else at.iloc[:0]
    if valid.empty:continue
    valid=valid[valid.priority==valid.priority.min()]
    if valid[f].max()-valid[f].min()>max(1e-6,abs(valid[f].mean())*1e-4):
     conflict_len+=length;continue
    v=float(valid[f].iloc[0]);weights[f].append((length,v));source_set.update(valid.source)
 result['解释覆盖率']=min(cover/total,1);result['层位组成']=';'.join(f'{l}:{v/total:.1%}' for l,v in layer_w.items());result['地质来源']=';'.join(sorted(source_set));result['多源冲突标志']=conflict_len>0
 for f,arr in weights.items():
  a=np.array(arr);w=a[:,0];v=a[:,1];cov=min(w.sum()/total,1);mean=float(np.average(v,weights=w));result[f]=mean;result[f+'_覆盖率']=cov
  if f in ['杨氏模量_GPa','最小主应力_MPa','水平应力差_MPa','总有机碳_pct']:
   result[f+'_段内标准差']=float(np.sqrt(np.average((v-mean)**2,weights=w)));result[f+'_最小值']=float(v.min());result[f+'_最大值']=float(v.max())
  lineage.append(dict(key=keystr(key),scope=scope,field=f,value=mean,coverage=cov,method='按有效重叠测深长度加权；分母为该字段有效覆盖长度',source=result['地质来源']))
 return result

out=[]
for count,k in enumerate(sorted(allkeys)):
 q={'platform':k[0],'well':int(k[1]),'stage':int(k[2]),'井段键':keystr(k)};base=smap.get(k,{});a,b=base.get('md_top',np.nan),base.get('md_bottom',np.nan);ss=base.get('file','');q['分段冲突']=bool(base.get('stage_conflict',False));eq=em.get(k)
 if not finite(a) or not finite(b):
  if eq is not None:
   valid=eq[eq.get('段深A_m',pd.Series(np.nan,index=eq.index)).between(1000,12000)&eq.get('段深B_m',pd.Series(np.nan,index=eq.index)).between(1000,12000)]
   if len(valid):r=valid.iloc[0];a,b=sorted([r['段深A_m'],r['段深B_m']]);ss=r.source
 if eq is not None and finite(a) and finite(b):
  for _,r in eq.iterrows():
   v1,v2=r.get('段深A_m',np.nan),r.get('段深B_m',np.nan)
   if finite(v1) and finite(v2) and min(v1,v2)>1000 and (abs(min(v1,v2)-a)>1 or abs(max(v1,v2)-b)>1):q['分段冲突']=True
 q.update({'段顶测深_m':a,'段底测深_m':b,'分段来源':ss,'测深段长_m':b-a if finite(a) and finite(b) else np.nan})
 for f in ef:
  if f.startswith('段深'):continue
  v,src,conf=choose(lm.get(k),f,k,'原统计')
  # Sharing a source-stage value across neighbors is allowed only when legacy rows agree.
  if conf:v,src=np.nan,''
  if not finite(v):v,src,conf=choose(eq,f,k,'工程')
  if finite(v):q[f]=v;lineage.append(dict(key=keystr(k),scope='工程',field=f,value=v,coverage=None,method='原统计一致值优先，平台实际施工表补缺',source=src))
 q['工程冲突字段']=';'.join(sorted(set(i['field'] for i in issues if i['key']==keystr(k))))
 layers=gm.get(k[:2]);valid_depth=finite(a) and finite(b) and b>a
 if valid_depth:q.update(weighted(layers,[(a,b)],k,'全井段'))
 else:q.update({'解释覆盖率':0.,'层位组成':'','地质来源':''})
 clusters=pm.get(k)
 if clusters is not None:
  ranges=list(zip(clusters.top,clusters.bottom));q['设计射孔簇数_重算']=len(clusters);q['设计射孔总长度_m']=sum(y-x for x,y in ranges)
  center=np.sort((clusters.top.to_numpy()+clusters.bottom.to_numpy())/2)
  if len(center)>1:q['簇中心平均间距_m']=float(np.mean(np.diff(center)))
  q['射孔超出分段']=bool(valid_depth and ((clusters.top<a-.01)|(clusters.bottom>b+.01)).any())
  density=clusters.density.dropna()
  if len(density)==len(clusters):q['设计孔数_重算']=float(np.sum((clusters.bottom-clusters.top)*clusters.density))
  # Merge overlapping perforation intervals to prevent double weighting.
  united=[]
  for x,y in sorted(ranges):
   if united and x<=united[-1][1]:united[-1][1]=max(y,united[-1][1])
   else:united.append([x,y])
  pw=weighted(layers,united,k,'射孔簇');q.update({'射孔_'+f:v for f,v in pw.items()})
 q['深度基准说明']='小层顶底深按测深匹配，原表未明确MD者待确认；未与垂深混用'
 q['地质计算状态']='缺分段深度' if not valid_depth else '缺小层解释' if layers is None else '无深度交集' if q['解释覆盖率']==0 else '覆盖不足95%' if q['解释覆盖率']<.95 else '覆盖至少95%'
 out.append(q)
 if count%500==0:print('STAGES',count,flush=True)
d=pd.DataFrame(out);d.to_pickle(B/'stage_factors.pkl');pd.DataFrame(lineage).to_pickle(B/'factor_lineage.pkl');pd.DataFrame(issues).to_pickle(B/'factor_conflicts.pkl')
print('DONE',d.shape,'GEO',d['地质计算状态'].value_counts().to_dict(),'PLAT',d.platform.nunique())
