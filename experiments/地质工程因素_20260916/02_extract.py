from pathlib import Path
import sys,re,pickle,json,collections
import numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');OLD=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916')
rs=pickle.load(open(B/'source_tables.pkl','rb'))
def clean(x):return re.sub(r'\s+','',str(x or '')).replace('（','(').replace('）',')')
def number(x):
 try:
  v=float(str(x).strip());return v if np.isfinite(v) else np.nan
 except:return np.nan
def integer(x):
 v=number(x)
 if np.isfinite(v) and v==int(v) and 0<v<=200:return int(v)
 m=re.fullmatch(r'第?(\d+)段?',clean(x));return int(m[1]) if m and 0<int(m[1])<=200 else None
def wid(x):
 m=re.search(r'([泸阳自]\d+H\d+)[-－—](\d+)',str(x),re.I);return (m[1].upper(),int(m[2])) if m else None
def plat(x):
 m=re.search(r'[泸阳自]\d+H\d+',str(x),re.I);return m[0].upper() if m else None
def provenance(r,i):return f"{r['file']} | {r['sheet']} | 行{i}"
def col(h,term):return next((j for j,v in enumerate(h) if term in v),None)
GEO={'孔隙度_pct':'孔隙度','含水饱和度_pct':'含水饱和度','总有机碳_pct':'有机碳','最大主应力_MPa':'最大主应力','最小主应力_MPa':'最小主应力','垂向应力_MPa':'垂向应力','破裂压力_MPa':'破裂压力','泊松比':'泊松比','杨氏模量_GPa':'杨氏模量','自然伽马_API':'自然伽马'}
geo=[];issues=[];headers=[];perfs=[];eng=[];legacy=[]
for r in rs:
 rows=r['rows'];w=wid(r['file'])
 if r['kind']=='geology' and w:
  start=next((i for i,row in enumerate(rows) if 1000<number(row[2])<12000 and 1000<number(row[3])<12000),None)
  if start is None:issues.append({'source':r['file'],'issue':'未识别小层深度'});continue
  h=[''.join(clean(row[j]) for row in rows[:start]) for j in range(40)]
  mp={k:col(h,t) for k,t in GEO.items()}
  for j,v in enumerate(h):
   if '含气量' in v:mp['含气量_压力系数2_m3_t' if '2.0' in v else '含气量_压力系数1_m3_t' if '1.0' in v else '含气量_口径未注明_m3_t']=j
   if '脆性' in v:
    k='脆性矿物_不含碳酸盐_pct' if '不含碳酸盐' in v else '脆性矿物_含碳酸盐_pct' if '含碳酸盐' in v else '泊杨脆性指数_pct' if '泊杨' in v else '纵横波脆性指数_pct' if '纵横波' in v else '矿物脆性指数_pct'
    mp[k]=j
  headers.append(dict(file=r['file'],kind='geology',mapping={k:f'{j+1}:{h[j]}' for k,j in mp.items() if j is not None},depth_basis='顶底深按测深匹配；源表未明确标注MD，基准待确认'))
  for i,row in enumerate(rows[start:],start+1):
   a,b=number(row[2]),number(row[3])
   if not (1000<a<b<12000):continue
   q=dict(platform=w[0],well=w[1],top=a,bottom=b,layer=str(row[1]),source=provenance(r,i))
   q.update({k:number(row[j]) for k,j in mp.items() if j is not None})
   nu=q.get('泊松比',np.nan);stressbad=q.get('最大主应力_MPa',np.nan)<q.get('最小主应力_MPa',np.nan)
   shifted=np.isfinite(nu) and not 0<nu<.5
   if shifted:
    issues.append(dict(source=q['source'],issue='泊松比超出物理范围，疑似表头/数值错列；力学及脆性字段不采用',raw_nu=nu))
    for k in list(q):
     if any(t in k for t in ['应力','压力','泊松','模量','脆性','伽马']):q[k]=np.nan
   for k,v in list(q.items()):
    if not isinstance(v,(int,float)) or k in ['well','top','bottom'] or not np.isfinite(v):continue
    ok=(0<=v<=100) if k.endswith('_pct') else (.0<v<.5) if k=='泊松比' else (1<v<150) if k=='杨氏模量_GPa' else (0<=v<500) if k.endswith('_MPa') else v>=0
    if not ok:issues.append(dict(source=q['source'],issue=f'{k}值异常，留空',raw_value=v));q[k]=np.nan
   if stressbad and not shifted:
    issues.append(dict(source=q['source'],issue='最大主应力小于最小主应力，应力字段待核查'))
    q['最大主应力_MPa']=q['最小主应力_MPa']=np.nan
   q['水平应力差_MPa']=q.get('最大主应力_MPa',np.nan)-q.get('最小主应力_MPa',np.nan)
   if q.get('最小主应力_MPa',0)>0:q['水平应力差系数']=q['水平应力差_MPa']/q['最小主应力_MPa']
   if np.isfinite(q.get('杨氏模量_GPa',np.nan)) and np.isfinite(q.get('泊松比',np.nan)):
    e,nu=q['杨氏模量_GPa'],q['泊松比'];q['平面应变模量_GPa']=e/(1-nu*nu);q['剪切模量_GPa']=e/(2*(1+nu));q['体积模量_GPa']=e/(3*(1-2*nu))
   geo.append(q)
 elif r['kind']=='perforation' and w:
  h=[clean(c) for c in rows[0]];stage=None;density=col(h,'孔密')
  standard=any('分段' in x for x in h) and any('射孔' in x for x in h) and len(h)>8 and ('射孔' in h[6])
  if not standard:issues.append(dict(source=r['file'],issue='非逐簇标准格式；保留井段统计，不从此表推算逐簇深度'));continue
  for i,row in enumerate(rows,1):
   st=integer(row[0])
   if st is not None:stage=st
   a,b=number(row[6]),number(row[7])
   if stage and 1000<a<12000 and 1000<b<12000 and 0<abs(a-b)<=30:perfs.append(dict(platform=w[0],well=w[1],stage=stage,top=min(a,b),bottom=max(a,b),density=number(row[density]) if density is not None else np.nan,source=provenance(r,i)))
 elif r['kind']=='legacy':
  p=plat(r['sheet']);h=[clean(c) for c in rows[1]]
  for i,row in enumerate(rows[2:],3):
   m=re.fullmatch(r'(\d+)[-－—](\d+)[-－—](\d+)',clean(row[0]))
   if not m or not p:continue
   q=dict(platform=p,well=int(m[1]),stage=int(m[2]),monitor_platform=p,monitor_well=int(m[3]),source=provenance(r,i))
   for k,t in {'井距_m':'井距','段长_m':'段长','簇间距_m':'簇间距','簇数':'簇数','孔数':'孔数','单簇孔数':'单簇孔数','加砂强度_t_m':'加砂强度','用液强度_m3_m':'用液强度','排量_m3_min':'排量','原统计_杨氏模量_GPa':'杨氏模量','原统计_最小主应力_MPa':'最小水平主应力','原统计_水平应力差_MPa':'水平应力差'}.items():
    j=next((j for j,x in enumerate(h) if x==t),None)
    if j is not None:q[k]=number(row[j])
   legacy.append(q)
 elif r['kind']=='engineering':
  hi=next((i for i,row in enumerate(rows[:8]) if any(clean(c)=='井号' for c in row) and any(clean(c) in ['段号','段次','施工段'] for c in row) and '加砂' in str(row) and '用液' in str(row)),None)
  if hi is None:continue
  h=[clean(c) for c in rows[hi]];wi=h.index('井号');si=next(j for j,x in enumerate(h) if x in ['段号','段次','施工段']);last=None
  mapping={'段长_m':['实际分段段长','段长(m)'],'簇数':['实际簇数','簇数'],'孔数':['实际孔数','孔数'],'加砂强度_t_m':['加砂强度','加砂强度(t/m)'],'用液强度_m3_m':['用液强度','用液强度(m3/m)'],'排量_m3_min':['平均排量','一般排量(m3/min)'],'最高排量_m3_min':['最高施工排量'],'最低排量_m3_min':['最低施工排量'],'平均施工压力_MPa':['平均施工压力','一般压力(MPa)'],'总液量_m3':['总液量','用液量(m3)'],'净液量_m3':['净液量'],'总砂量_t':['总砂量','加砂量(t)'],'前置液量_m3':['前置液'],'段深A_m':['A点试油深度'],'段深B_m':['B点试油深度'],'施工表平均垂深_m':['平均垂深']}
  mp={k:next((j for term in terms for j,x in enumerate(h) if x==term),None) for k,terms in mapping.items()}
  headers.append(dict(file=r['file'],sheet=r['sheet'],kind='engineering',mapping={k:f'{j+1}:{h[j]}' for k,j in mp.items() if j is not None}))
  for i,row in enumerate(rows[hi+1:],hi+2):
   cur=wid(row[wi])
   if cur:last=cur
   st=integer(row[si])
   if not last or st is None:continue
   if row[wi] is not None and not cur:continue
   q=dict(platform=last[0],well=last[1],stage=st,source=provenance(r,i),file=r['file'],sheet=r['sheet'])
   q.update({k:number(row[j]) for k,j in mp.items() if j is not None})
   if not any(np.isfinite(q.get(k,np.nan)) for k in ['加砂强度_t_m','用液强度_m3_m','排量_m3_min']):continue
   eng.append(q)
for name,arr in [('layers',geo),('perforations',perfs),('engineering_candidates',eng),('legacy_candidates',legacy),('source_issues',issues)]:
 d=pd.DataFrame(arr);d.to_pickle(B/(name+'.pkl'));d.to_csv(B/(name+'.csv'),index=False,encoding='utf-8-sig')
(B/'field_mappings.json').write_text(json.dumps(headers,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'layers':len(geo),'geology_wells':len(set((q['platform'],q['well']) for q in geo)),'perforations':len(perfs),'engineering_rows':len(eng),'engineering_stages':len(set((q['platform'],q['well'],q['stage']) for q in eng)),'legacy':len(legacy),'issues':len(issues)},ensure_ascii=False))
