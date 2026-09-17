from pathlib import Path
import json,re,sys,zipfile
import numpy as np,pandas as pd,openpyxl
from lxml import etree as ET
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');R=Path('E:/压裂窜扰项目/02 规范化')
def ident(s):
 m=re.search(r'([泸阳]\d+H\d+)[-－—](\d+)',s,re.I)
 return (m[1].upper(),int(m[2])) if m else None
def num(v):
 try:return float(str(v).strip())
 except:return np.nan
def clean(v):return re.sub(r'\s','',str(v or ''))
heads=[];stage=[];survey=[];audit=[]
for p in R.rglob('*.xlsx'):
 if '地质工程因素' not in str(p):continue
 wid=ident(p.name)
 if not wid and '井位' not in p.name:continue
 if not any(x in p.name for x in ['井斜','射孔','分段','井位']):continue
 try:
  w=openpyxl.load_workbook(p,read_only=True,data_only=True);rows=list(w.worksheets[0].iter_rows(values_only=True));w.close()
 except Exception as e:audit.append(dict(file=str(p),kind='read_error',note=str(e)));continue
 if '井位' in p.name:
  for i,r in enumerate(rows[1:],2):
   wi=ident(str(r[0]));
   if wi and len(r)>=5:heads.append(dict(platform=wi[0],well=wi[1],east=num(r[2]),north=num(r[1]),kb=num(r[4]),file=str(p),row=i,note='井位表：Y为东坐标，X为北坐标'))
  continue
 if '井斜' in p.name:
  h=next((i for i,r in enumerate(rows[:8]) if any('垂深' in clean(x) for x in r)),None)
  if h is None:continue
  hdr=[clean(x) for x in rows[h]]
  def col(terms,exclude='xxx'):
   return next((i for i,t in enumerate(hdr) if any(k in t for k in terms) and exclude not in t),None)
  cc={k:col(t,e) for k,t,e in [('md',['测深','井深'],'xxx'),('inc',['井斜'],'xxx'),('az',['方位'],'闭合'),('tvd',['垂深'],'xxx'),('closure',['闭合距'],'xxx'),('closure_az',['闭合方位'],'xxx'),('dn',['北坐标'],'xxx'),('de',['东坐标'],'xxx')]}
  if any(cc[k] is None for k in ['md','inc','az','tvd']):continue
  for i,r in enumerate(rows[h+1:],h+2):
   q={k:num(r[j]) if j is not None and j<len(r) else np.nan for k,j in cc.items()}
   if np.isfinite(q['md']) and 0<=q['md']<=12000 and 0<=q['inc']<=180:survey.append(dict(platform=wid[0],well=wid[1],file=str(p),row=i,priority=0,**q))
 else:
  for i,r in enumerate(rows,1):
   if len(r)<4:continue
   a,b,c,l=map(num,r[:4])
   if not (np.isfinite(a) and a.is_integer() and 1<=a<=150 and 1000<b<12000 and 1000<c<12000 and 1<=abs(b-c)<=2000):continue
   stage.append(dict(platform=wid[0],well=wid[1],stage=int(a),md_top=min(b,c),md_bottom=max(b,c),declared_length=l,file=str(p),row=i,priority=0 if '分段' in p.name else 1))
# Documents: literal coordinates and structured survey/stage tables, including Word binary table delimiters.
index=json.loads((B/'document_index.json').read_text(encoding='utf-8'))
for rec in index:
 if 'cache' not in rec:continue
 p=Path(rec['path']);wid=ident(p.name)
 if not wid:continue
 tx=(B/'document_text'/rec['cache']).read_text(encoding='utf-8')
 for m in re.finditer(r'井口.{0,8}坐标',tx):
  snippet=tx[m.end():m.end()+180];xy=re.findall(r'(?<!\d)(\d{6,8}\.\d+)',snippet)
  if len(xy)<2:continue
  coords=list(map(float,xy[:2]));north=next((v for v in coords if 3000000<v<4000000),None);east=next((v for v in coords if v>10000000 or 100000<v<1000000),None)
  if north is None or east is None:continue
  kb=re.search(r'补心海拔[^\d]{0,15}(\d+\.?\d*)',tx[m.end():m.end()+400]);heads.append(dict(platform=wid[0],well=wid[1],east=east,north=north,kb=float(kb[1]) if kb else np.nan,file=str(p),row=None,note=snippet[:130]));break
 tables=[]
 if p.suffix.lower()=='.docx':
  try:
   with zipfile.ZipFile(p) as z:root=ET.fromstring(z.read('word/document.xml'))
   ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
   tables=[[[''.join(c.itertext()) for c in r.findall('w:tc',ns)] for r in t.findall('w:tr',ns)] for t in root.findall('.//w:tbl',ns)]
  except:pass
 else:
  # Consecutive table rows terminate with a pair of cell marks in extracted Word text.
  rr=[r.lstrip('\x07').split('\x07') for r in tx.split('\x07\x07')];tables=[rr]
 for ti,rows in enumerate(tables):
  mode=None;cc={};sstart=None
  for ri,r in enumerate(rows):
   hdr=[clean(x) for x in r]
   if any('垂深' in x for x in hdr) and any('井斜' in x for x in hdr) and any('方位' in x for x in hdr):
    def col(ts,ex='xxx'):return next((j for j,x in enumerate(hdr) if any(k in x for k in ts) and ex not in x and '序号' not in x and len(x)<35),None)
    cc={k:col(ts,ex) for k,ts,ex in [('md',['测深','井深'],'最大'),('inc',['井斜'],'最大'),('az',['方位'],'闭合'),('tvd',['垂深'],'xxx'),('closure',['闭合距'],'xxx'),('closure_az',['闭合方位'],'xxx'),('dn',['北坐标'],'xxx'),('de',['东坐标'],'xxx')]}
    mode='survey' if all(cc[k] is not None for k in ['md','inc','az','tvd']) else None;continue
   if any('分段' in x for x in hdr) and (any('段序' in x or '段数' in x or x=='段次' for x in hdr)):
    mode='stage';continue
   if mode=='survey':
    q={k:num(r[j]) if j is not None and j<len(r) else np.nan for k,j in cc.items()}
    if np.isfinite(q['md']) and 0<=q['md']<=12000 and 0<=q['inc']<=180 and 0<=q['az']<=360 and 0<=q['tvd']<=12000:survey.append(dict(platform=wid[0],well=wid[1],file=str(p),row=f'T{ti+1}R{ri+1}',priority=2,**q))
   elif mode=='stage' and len(r)>=4:
    a,b,c,l=map(num,r[:4])
    if np.isfinite(a) and a.is_integer() and 1<=a<=150 and 1000<b<12000 and 1000<c<12000 and 1<=abs(b-c)<=2000 and abs(abs(b-c)-l)<2:stage.append(dict(platform=wid[0],well=wid[1],stage=int(a),md_top=min(b,c),md_bottom=max(b,c),declared_length=l,file=str(p),row=f'T{ti+1}R{ri+1}',priority=2))
for name,rs in [('wellhead_candidates',heads),('stage_candidates',stage),('survey_candidates',survey)]:pd.DataFrame(rs).to_pickle(B/(name+'.pkl'))
# Select authoritative row source by explicit structured-table priority, retain conflicts.
h=[]
for key,g in pd.DataFrame(heads).groupby(['platform','well']):
 r=g.iloc[0].to_dict();r['coordinate_conflict']=bool(g.east.max()-g.east.min()>1 or g.north.max()-g.north.min()>1);h.append(r)
h=pd.DataFrame(h);h.to_csv(B/'wellheads.csv',index=False,encoding='utf-8-sig')
st=[]
for key,g in pd.DataFrame(stage).groupby(['platform','well','stage']):
 g=g.sort_values(['priority','file']);r=g.iloc[0].to_dict();r['stage_conflict']=bool(g.md_top.max()-g.md_top.min()>1 or g.md_bottom.max()-g.md_bottom.min()>1);r['md_mid']=(r['md_top']+r['md_bottom'])/2;st.append(r)
st=pd.DataFrame(st);st.to_csv(B/'stages.csv',index=False,encoding='utf-8-sig')
tr=[]
for key,g in pd.DataFrame(survey).groupby(['platform','well']):
 sources=g.groupby(['priority','file']).size().reset_index(name='n').sort_values(['priority','n'],ascending=[True,False]);f=sources.iloc[0]['file'];q=g[g.file==f].sort_values('md').drop_duplicates('md').copy();q=q[q.tvd.between(0,12000)&q.az.between(0,360)];q['trajectory_source_conflict']=False
 if len(q)<3:continue
 if q.de.notna().sum()>len(q)*.9 and q.dn.notna().sum()>len(q)*.9:q['dx']=q.de;q['dy']=q.dn;q['offset_basis']='源表东/北偏移'
 else:q['dx']=q.closure*np.sin(np.deg2rad(q.closure_az));q['dy']=q.closure*np.cos(np.deg2rad(q.closure_az));q['offset_basis']='闭合距/闭合方位'
 valid_offsets=q.dx.notna()&q.dy.notna()
 if valid_offsets.sum()>=3:q=q[valid_offsets].copy()
 hh=h[(h.platform==key[0])&(h.well==key[1])]
 q['east']=np.nan;q['north']=np.nan;q['kb']=np.nan;q['coordinate_conflict']=False
 if len(hh):
  q['east']=q.dx+hh.iloc[0].east;q['north']=q.dy+hh.iloc[0].north;q['kb']=hh.iloc[0].kb;q['coordinate_conflict']=hh.iloc[0].coordinate_conflict
 tr.extend(q.to_dict('records'))
tr=pd.DataFrame(tr);tr.to_pickle(B/'trajectories.pkl');pd.DataFrame(audit).to_csv(B/'read_errors.csv',index=False,encoding='utf-8-sig')
print(json.dumps(dict(wellheads=len(h),head_conflicts=int(h.coordinate_conflict.sum()),stages=len(st),stage_conflicts=int(st.stage_conflict.sum()),survey_wells=len(tr[['platform','well']].drop_duplicates()),located_survey_wells=len(tr[tr.east.notna()][['platform','well']].drop_duplicates()),platforms=sorted(st.platform.unique())),ensure_ascii=False))
