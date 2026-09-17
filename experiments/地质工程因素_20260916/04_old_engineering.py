from pathlib import Path
import pickle,re,sys,json
import pandas as pd,numpy as np
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');rs=pickle.load(open(B/'source_tables.pkl','rb'));e=pd.read_pickle(B/'engineering_candidates.pkl');out=[]
def clean(x):return re.sub(r'\s+','',str(x or '')).replace('（','(').replace('）',')')
def num(x):
 try:return float(str(x).strip())
 except:return np.nan
def wid(s):
 m=re.search(r'([泸阳自]\d+H\d+)[-－—](\d+)',str(s),re.I);return (m[1].upper(),int(m[2])) if m else None
for r in rs:
 if r['kind']!='engineering':continue
 rows=r['rows'];h=[clean(c) for c in rows[0]];si=next((j for j,c in enumerate(h) if c in ['段序','第几段']),None)
 if si is None:continue
 w=wid(r['sheet']) or wid(Path(r['file']).name)
 if w is None:
  p=re.search(r'[泸阳自]\d+H\d+',r['file']);v=re.fullmatch(r'(\d+)井',r['sheet']);w=(p[0],int(v[1])) if p and v else None
 wi=next((j for j,c in enumerate(h) if c=='井号'),None)
 mp={}
 patterns={'段长_m':r'^段长','加砂强度_t_m':r'^加砂强度\(','用液强度_m3_m':r'^(实际)?用液强度\(','排量_m3_min':r'^主体施工排量','总液量_m3':r'^总液量\(','净液量_m3':r'^\(净液\)总液量','总砂量_t':r'^总砂量\(','前置液量_m3':r'^前置液$','段深A_m':r'^试油位置'}
 for k,pat in patterns.items():mp[k]=next((j for j,c in enumerate(h) if re.search(pat,c)),None)
 if mp['段长_m'] is None and len(rows)>1:mp['段长_m']=next((j for j,c in enumerate(rows[1]) if '段长' in str(c)),None)
 ratecol=next((j for j,c in enumerate(h) if c.startswith('施工排量')),None)
 for i,row in enumerate(rows[2:],3):
  st=num(row[si]);cur=wid(row[wi]) if wi is not None else None
  if cur:w=cur
  if w is None or not np.isfinite(st) or st!=int(st) or not 1<=st<=150:continue
  q=dict(platform=w[0],well=w[1],stage=int(st),source=f"{r['file']} | {r['sheet']} | 行{i}",file=r['file'],sheet=r['sheet'])
  q.update({k:num(row[j]) for k,j in mp.items() if j is not None})
  if mp['段深A_m'] is not None:q['段深B_m']=num(row[mp['段深A_m']+1])
  if ratecol is not None:
   v=num(row[ratecol]);raw=clean(row[ratecol]);q['施工排量原文']=raw
   if np.isfinite(v) and not np.isfinite(q.get('排量_m3_min',np.nan)):q['排量_m3_min']=v
   m=re.fullmatch(r'(\d+\.?\d*)[-～~—](\d+\.?\d*)',raw)
   if m:q['最低排量_m3_min'],q['最高排量_m3_min']=sorted(map(float,m.groups()))
  if any(np.isfinite(q.get(k,np.nan)) and q[k]>0 for k in ['加砂强度_t_m','用液强度_m3_m','总液量_m3','净液量_m3']):out.append(q)
e=pd.concat([e,pd.DataFrame(out)],ignore_index=True);e.to_pickle(B/'engineering_candidates.pkl');e.to_csv(B/'engineering_candidates.csv',index=False,encoding='utf-8-sig');print('ADDED',len(out),'STAGES',len(e[['platform','well','stage']].drop_duplicates()))
r=pd.read_pickle(B.parent/'时序聚类_V3_形态与强度/results.pkl');print(r.groupby(['morphology_code_v3','morphology_v3']).size().to_string())
