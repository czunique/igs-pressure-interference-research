"""Read-only source ingestion. All generated files stay in the requested process directory."""
import os,sys,re,json,hashlib,warnings,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
ROOT=Path('E:/压裂窜扰项目/02 规范化')
BASE=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915')
OUT=Path('F:/论文库/IGS/实验/结论/时序聚类_20260915')
CACHE=BASE/'cache'; CACHE.mkdir(exist_ok=True)

def identity(path):
 rel=path.relative_to(ROOT); parts=rel.parts
 platform=parts[1] if parts[0]=='不重要的' else parts[0]
 s=path.stem.replace('平台','').replace(' ','')
 s=re.sub(r'^'+re.escape(platform),'',s)
 m=re.search(r'^-?(\d+)井?[-_](\d+)(?:段|压裂|$)',s)
 if not m:m=re.search(r'(\d+)井(?:第)?\s*(\d+)\s*段',s)
 if not m:m=re.search(r'(\d+)井_(\d+)_',s)
 if not m:m=re.search(r'(\d+)井第?(首|一|二|三|四|五|六|七|八|九|十)段',s)
 if m:
  a,b=m.groups(); b=str({'首':1,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}.get(b,b))
  return platform,a,b
 return platform,'',''

def read_sheets(path):
 try:
  x=pd.ExcelFile(path,engine='calamine')
  return [(s,pd.read_excel(x,s,header=None)) for s in x.sheet_names]
 except Exception as orig:
  data=path.read_bytes()
  if data[:2]==b'PK' or data[:4]==b'\xd0\xcf\x11\xe0':raise orig
  text=None
  for enc in ['utf-16' if data[:2] in [b'\xff\xfe',b'\xfe\xff'] else 'utf-8-sig','gb18030','utf-16']:
   try:text=data.decode(enc);break
   except:pass
  if text is None:raise orig
  if '<Workbook' in text or ':Workbook' in text:
   import xml.etree.ElementTree as ET
   rt=ET.fromstring(text); ss='{urn:schemas-microsoft-com:office:spreadsheet}'
   result=[]
   for sh in rt.iter(ss+'Worksheet'):
    rows=[]
    for row in sh.iter(ss+'Row'):
     vals=[]
     for cell in row.findall(ss+'Cell'):
      ind=int(cell.attrib.get(ss+'Index',len(vals)+1))-1
      vals += [None]*max(0,ind-len(vals))
      d=cell.find(ss+'Data');vals.append(d.text if d is not None else None)
     rows.append(vals)
    result.append((sh.attrib.get(ss+'Name','Sheet'),pd.DataFrame(rows)))
   if result:return result
  if '<table' in text.lower():
   from io import StringIO
   return [(str(i),d) for i,d in enumerate(pd.read_html(StringIO(text),header=None))]
  from io import StringIO
  for delim in ['\t',',',';']:
   if delim in text:
    import csv
    rows=list(csv.reader(StringIO(text),delimiter=delim))
    if max(map(len,rows),default=0)>1:return [('text',pd.DataFrame(rows))]
  for sep in ['\t',',',';']:
   try:
    df=pd.read_csv(StringIO(text),sep=sep,header=None,on_bad_lines='skip',engine='python')
    if df.shape[1]>1:return [('text',df)]
   except:pass
  raise orig

def time_values(df):
 head=df.iloc[:80,:].fillna('').astype(str)
 found=[]
 for i in range(len(head)):
  for j in range(df.shape[1]):
   v=head.iat[i,j].strip().rstrip(':：')
   if v.startswith(('时间(', '时间（')) or v in ['时间','日期时间','采集时间','施工时间','Time','Time Stamp','日期/时间'] or (len(v)<16 and '时间' in v and ':' not in v):found.append((i,j))
 if found: h,tc=found[0]; start=h+1
 else:
  h=-1;start=0;tc=None
  for j in range(min(4,df.shape[1])):
   vals=head.iloc[:15,j]
   if vals.str.contains(r'\d{1,2}:\d{2}:\d{2}',regex=True).sum()>=3:tc=j;break
  if tc is None:raise ValueError('no_time_column')
 raw=df.iloc[start:,tc].astype(str).str.strip()
 datecol=None
 if h>=0:
  for j in range(df.shape[1]):
   if j!=tc and str(df.iat[h,j]).strip() in ['日期','Date']:datecol=j
 if datecol is not None:
  ds=df.iloc[start:,datecol].ffill().astype(str).str.strip()
  ds=ds.str.replace(r'^(\d{2})/',r'20\1/',regex=True)
  raw=ds+' '+raw
 absflag=raw.str.contains(r'(?:19|20)\d{2}[-/]\d',regex=True).sum()>max(2,len(raw)*.2)
 if absflag:
  t=pd.to_datetime(raw,errors='coerce',format='mixed')
  arr=t.dt.as_unit('ns').astype('int64').to_numpy(dtype=float,copy=True)/1e9;arr[t.isna()]=np.nan
 else:
  mm=raw.str.extract(r'(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)')
  arr=(pd.to_numeric(mm[0],errors='coerce')*3600+pd.to_numeric(mm[1],errors='coerce')*60+pd.to_numeric(mm[2],errors='coerce')).to_numpy(float,copy=True)
  good=np.flatnonzero(np.isfinite(arr))
  if len(good)>1:
   rolls=np.cumsum(np.r_[0,np.diff(arr[good]) < -43200])*86400;arr[good]+=rolls
  # Only construction-date header, never export timestamp.
  ht=' '.join(head.to_numpy().flatten())
  match=re.search(r'(?:施工时间|日期)\s*[:：]?\s*(20\d{2})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})',ht)
  if match:
   y,m,d=map(int,match.groups());arr+=pd.Timestamp(y,m,d).timestamp();absflag=True
 return arr,start,h,tc,datecol,absflag

def parse_file(path,kind):
 fid=hashlib.sha1(str(path).encode()).hexdigest()[:16]
 dest=CACHE/(kind+'_'+fid+'.pkl')
 if dest.exists():
  cached=pd.read_pickle(dest)
  if cached['channels'] and not any('read-only' in x for x in cached['errors']):
   for c in cached['channels']:
    if c['absolute'] and np.nanmedian(c['t'])<1e8:
     c['t']=np.round(c['t']*1000,4);c['dt']=float(np.median(np.diff(c['t'])))
   return cached
 platform,source,stage=identity(path)
 result={'path':str(path),'platform':platform,'source':source,'stage':stage,'kind':kind,'size':path.stat().st_size,'mtime':path.stat().st_mtime_ns,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'channels':[],'errors':[]}
 try:
  sheets=read_sheets(path)
  for sheet,df in sheets:
   try:
    df=df.dropna(axis=1,how='all').reset_index(drop=True)
    df.columns=range(df.shape[1])
    if df.shape[0]<10:continue
    t,start,h,tc,dc,ab=time_values(df)
    cols=[]
    if kind=='pump':
     for j in range(df.shape[1]):
      s=' '.join(df.iloc[max(0,h):max(0,h)+2,j].fillna('').astype(str))
      if '排量' in s and not any(k in s for k in ['累计','砂','设计']):cols.append((j,'rate',s))
    else:
     if h<0:
      mons=re.search(r'压裂施工[-－](.+?)井',path.stem)
      ids=re.findall(r'\d+',mons.group(1)) if mons else []
      if not ids:
       sm=re.fullmatch(r'(\d+)-(\d+)-(\d+)',sheet)
       if sm:ids=[sm.group(3)]
      if len(ids)==df.shape[1]-1:
       cols=[(j,ids[j-1],'无表头_文件名井号') for j in range(1,df.shape[1])]
     else:
      for j in range(df.shape[1]):
       if j in [tc,dc]:continue
       label=str(df.iat[h,j]);m=re.search(r'(\d+)\s*(?:#|井|号)',label)
       if m and ('压' in label or '井' in label):cols.append((j,m.group(1),label))
      if not cols and df.shape[1]<=4:
       for j in range(df.shape[1]):
        if j in [tc,dc]:continue
        label=str(df.iat[h,j]);mons=re.search(r'压裂施工[-－](\d+)井',path.stem)
        if mons and ('压' in label or df.shape[1]==2):cols.append((j,mons.group(1),label+'_文件名井号'))
    for col,well,label in cols:
     p=pd.to_numeric(df.iloc[start:,col],errors='coerce').to_numpy(float)
     good=np.isfinite(t)&np.isfinite(p)
     if good.sum()<10:continue
     tt=t[good];pp=p[good]
     unsorted=int(np.sum(np.diff(tt)<0))
     z=pd.DataFrame({'t':tt,'p':pp}).groupby('t',as_index=False).median().sort_values('t')
     tt=z.t.to_numpy();pp=z.p.to_numpy()
     result['channels'].append({'sheet':sheet,'monitor':well,'sheet_source':(re.fullmatch(r'(\d+)-(\d+)-(\d+)',sheet).group(1) if re.fullmatch(r'(\d+)-(\d+)-(\d+)',sheet) else ''),'sheet_stage':(re.fullmatch(r'(\d+)-(\d+)-(\d+)',sheet).group(2) if re.fullmatch(r'(\d+)-(\d+)-(\d+)',sheet) else ''),'label':label,'t':tt,'p':pp,'absolute':ab,'invalid':int((~good).sum()),'duplicates':int(good.sum()-len(tt)),'backwards':unsorted,'dt':float(np.median(np.diff(tt)))})
   except Exception as e:result['errors'].append(sheet+':'+str(e))
 except Exception as e:result['errors'].append(type(e).__name__+':'+str(e))
 pd.to_pickle(result,dest)
 return result

def main():
 files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in ['.xlsx','.xls','.csv','.txt'] and 'xxx' not in str(p) and not p.name.startswith('~$')]
 pressure=[p for p in files if '邻井压力' in str(p)]
 pump=[p for p in files if '压裂施工曲线' in str(p)]
 plats={identity(p)[0] for p in pressure}
 pump=[p for p in pump if identity(p)[0] in plats and identity(p)[1]]
 for kind,fs in [('pressure',pressure),('pump',pump)]:
  print(kind,'files',len(fs),flush=True);results=[]
  with ThreadPoolExecutor(max_workers=4) as ex:
   futures={ex.submit(parse_file,p,kind):p for p in fs}
   for k,fut in enumerate(as_completed(futures)):
    try:results.append(fut.result())
    except Exception as e:print('FATAL_FILE',futures[fut],str(e),flush=True)
    if (k+1)%100==0:print(kind,k+1,'/',len(fs),flush=True)
  results.sort(key=lambda x:x['path']);pd.to_pickle(results,BASE/(kind+'_parsed.pkl'))
  pd.DataFrame([{k:v for k,v in r.items() if k!='channels'}|{'channel_count':len(r['channels'])} for r in results]).to_csv(BASE/(kind+'_inventory.csv'),index=False,encoding='utf-8-sig')
  print(kind,'done',len(results),'channels',sum(len(r['channels']) for r in results),'unread',sum(not r['channels'] for r in results),flush=True)
if __name__=='__main__':main()

