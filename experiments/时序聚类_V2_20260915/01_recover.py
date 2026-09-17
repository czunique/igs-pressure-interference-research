"""V2: recover sheet-level identities and native-resolution pressure records. Sources read only."""
import sys,re,json,hashlib,importlib.util,warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore')
OLD=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915')
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915');C=B/'cache';C.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('v1',OLD/'01_ingest.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
ROOT=v1.ROOT

def chinese_number(s):
 try:return str(int(s))
 except:pass
 d={'首':1,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'零':0}
 if s=='十':return '10'
 if '十' in s:
  a,b=s.split('十');return str(d.get(a,1)*10+d.get(b,0))
 return str(d.get(s,s))

def identity(text,platform):
 s=str(text).replace('h','H').replace(' ','').replace('平台','-').replace('－','-')
 p=re.search(r'([泸阳自]\d+H\d+)',s);plat=p.group(1) if p else platform
 s=re.sub(r'[泸阳自]\d+H\d+','',s)
 m=re.search(r'(\d+)井?[-_](\d+)段',s)
 if not m:m=re.search(r'(\d+)井?第?([\d一二三四五六七八九十首]+)段',s)
 if not m:m=re.search(r'(\d+)井_(\d+)_',s)
 if m:return plat,m.group(1),chinese_number(m.group(2))
 return plat,'',''

def parse_times(df):
 hd=df.iloc[:60].fillna('').astype(str);opts=[]
 for i in range(min(35,len(df))):
  for j in range(min(8,df.shape[1])):
   label=str(df.iat[i,j]).strip().rstrip(':：')
   if ('时间' in label and len(label)<28 and not any(x in label for x in ['开始','结束','阶段','导出','施工日期'])) or label.lower() in ['time','timestamp','datetime']:opts.append((i,j))
 for h,tc in list(opts):
  if str(df.iat[h,tc]).strip()=='时间列' and tc+1<df.shape[1] and df.iloc[h+1:h+12,tc+1].astype(str).str.contains(':').sum()>=3:opts.append((h,tc+1))
 for j in range(min(4,df.shape[1])):
  if hd.iloc[:12,j].str.contains(r'\d{1,2}[:：]\d{2}',regex=True).sum()>=3:opts.append((-1,j))
 best=None
 for h,tc in opts:
  start=h+1;raw=df.iloc[start:,tc].astype(str).str.strip().str.replace('：',':');dc=None
  if h>=0:
   for j in range(df.shape[1]):
    if j!=tc and str(df.iat[h,j]).strip() in ['日期','Date','时间列']:dc=j
  if dc is None and tc>0:
   prev=df.iloc[start:start+15,tc-1].astype(str)
   if prev.str.contains(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}',regex=True).sum()>=3 and not prev.str.contains(':').any():dc=tc-1
  if dc is not None:
   ds=df.iloc[start:,dc].ffill().astype(str).str.strip().str.replace(r'^(\d{2})/',r'20\1/',regex=True);raw=ds+' '+raw
  absolute=raw.str.contains(r'(?:19|20)\d{2}[-/]\d',regex=True).sum()>2
  if absolute:
   tm=pd.to_datetime(raw,errors='coerce',format='mixed');t=tm.dt.as_unit('ns').astype('int64').to_numpy(float,copy=True)/1e9;t[tm.isna()]=np.nan
  else:
   mm=raw.str.extract(r'(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?');t=(pd.to_numeric(mm[0],errors='coerce')*3600+pd.to_numeric(mm[1],errors='coerce')*60+pd.to_numeric(mm[2].fillna(0),errors='coerce')).to_numpy(float,copy=True)
   ix=np.flatnonzero(np.isfinite(t))
   if len(ix)>1:t[ix]+=np.cumsum(np.r_[0,np.diff(t[ix]) < -43200])*86400
   header=' '.join(hd.iloc[:max(h,0)].to_numpy().ravel());match=re.search(r'(?:施工日期|施工时间|日期)\s*[:：]?\s*(20\d{2})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})',header)
   if match:
    y,m,d=map(int,match.groups());t+=pd.Timestamp(y,m,d).timestamp();absolute=True
  score=np.isfinite(t).sum()
  if score>=5 and (best is None or score>best[0]):best=(score,t,start,h,tc,dc,absolute)
 if best is None:raise ValueError('没有至少5个可解释时间点')
 return best[1:]

def parse(path):
 dest=C/(hashlib.sha1(str(path).encode()).hexdigest()[:16]+'.pkl')
 if dest.exists():return pd.read_pickle(dest)
 directory,_,_=v1.identity(path);fileplat,filesource,filestage=identity(path.stem,directory)
 r={'path':str(path),'directory_platform':directory,'platform':fileplat,'size':path.stat().st_size,'mtime':path.stat().st_mtime_ns,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'channels':[],'sheet_audit':[]}
 try:sheets=v1.read_sheets(path)
 except Exception as e:r['sheet_audit'].append({'sheet':'','status':'读取失败','reason':str(e)});pd.to_pickle(r,dest);return r
 for sheet,df in sheets:
  audit={'sheet':sheet,'nrows':len(df),'ncols':df.shape[1]}
  try:
   if any(z in sheet for z in ['阀门','砂量','核实表','照片','公报封面','施工设计','基础参数','阶段汇总','阶段数据']):audit.update(status='非时序压力表',reason='工况、数量或阶段汇总表');r['sheet_audit'].append(audit);continue
   df=df.replace(r'^\s*$',np.nan,regex=True).dropna(axis=1,how='all');df.columns=range(df.shape[1]);df=df.reset_index(drop=True)
   if len(df)<5 or df.shape[1]<2:audit.update(status='无可用序列',reason='空表或有效行列过少');r['sheet_audit'].append(audit);continue
   t,start,h,tc,dc,absolute=parse_times(df)
   header=' '.join(df.iloc[:max(h,0)].fillna('').astype(str).to_numpy().ravel())
   plat,source,stage=identity(header,fileplat)
   if not source:plat,source,stage=fileplat,filesource,filestage
   shmatch=re.fullmatch(r'(\d+)-(\d+)-(\d+)(?:\((H\d+)\))?',sheet.strip())
   if shmatch:source,stage=shmatch.group(1),shmatch.group(2)
   context=header+' '+path.stem
   # Pump-rate workbooks can be misplaced in monitoring directories: only separately named neighbor channels qualify.
   labels=[str(x) for x in df.iloc[h]] if h>=0 else ['']*df.shape[1]
   hasrate=any('排量' in a for a in labels)
   filename_monitor=None
   mm=re.search(r'压裂施工[-－](\d+)井',path.stem)
   if not mm:mm=re.search(r'(?:期间|施工|压裂|压窜)(?:邻井)?[-_ ]?(?:\d+-)?(\d+)井(?:套压|压力|秒点|分点)',path.stem)
   if mm:filename_monitor=mm.group(1)
   cols=[]
   for j in range(df.shape[1]):
    if j in [tc,dc]:continue
    label=labels[j] if h>=0 else ''
    if any(x in label for x in ['排量','液量','砂','油压','B环空','C环空','温度','序号','阶段']):continue
    if h>=0 and j>0 and label in ['nan',''] and any('环空' in a for a in labels):continue
    values=pd.to_numeric(df.iloc[start:,j],errors='coerce').to_numpy(float);good=np.isfinite(t)&np.isfinite(values)
    if good.sum()<5:continue
    mon='';monplat=plat;mapping='header'
    if re.fullmatch(r'ch\d+',label,re.I):
     hm=re.search(re.escape(label)+r'=\s*(\d+)井井口(?!温度)',header,re.I)
     if hm:mon=hm.group(1);mapping='channel_header_dictionary'
    if label in ['压力','套压']:
     above=' '.join(str(v) for v in df.iloc[:max(h,0),max(0,j-1):j+1].fillna('').to_numpy().ravel())
     hm=re.search(r'期间(\d+)井(?:压力|套压)',above)
     if hm:mon=hm.group(1);mapping='local_group_header'

    pm=re.search(r'([泸阳自]\d+H\d+)[- ]*(\d+)井?',label)
    wm=re.search(r'(\d+)\s*(?:#|井|号)',label)
    if pm:monplat,mon=pm.groups()
    elif wm:mon=wm.group(1)
    elif re.search(r'([一二三四五六七八九十]+)(?:井|号)',label):mon=chinese_number(re.search(r'([一二三四五六七八九十]+)(?:井|号)',label).group(1))
    elif re.search(r'No\d+|通道\d+|\d+通道',label,re.I):
     mon=filename_monitor or ('sensor:'+label);mapping='filename_sensor_map' if filename_monitor else 'sensor_unknown_well'
    elif '压' in label and not hasrate and not mon:
     if filename_monitor:mon=filename_monitor;mapping='filename'
     else:
      sm=re.match(r'(\d+)井',sheet)
      if sm:mon=sm.group(1);mapping='sheet'
      else:
       for nm in range(df.shape[1]):
        if labels[nm]=='井名':
         vm=re.search(r'([泸阳自]\d+H\d+)-(\d+)',str(df.iloc[start:,nm].dropna().iloc[0]))
         if vm:monplat,mon=vm.groups();mapping='well_name_column';break
    elif h<0 and shmatch:
     mon=shmatch.group(3);mapping='sheet_triplet'
     if shmatch.group(4):monplat=re.sub(r'H\d+$',shmatch.group(4),plat)
    elif h<0 and filename_monitor and df.shape[1]<=3:mon=filename_monitor;mapping='filename_headerless'
    if not mon and '压' in label and not hasrate:mon='sensor:'+label;mapping='pressure_unknown_well'
    if not mon:continue
    if hasrate and (not wm or mon==source):continue
    tt=t[good];pp=values[good]
    unit='MPa_explicit' if re.search(r'mpa',label,re.I) else 'MPa_context_assumed'
    if re.search(r'kpa',label,re.I):pp=pp/1000;unit='kPa_to_MPa'
    if re.search(r'psi',label,re.I):pp=pp*.006894757;unit='psi_to_MPa'
    zz=pd.DataFrame({'t':tt,'p':pp}).groupby('t',as_index=False).median().sort_values('t');tt=zz.t.to_numpy();pp=zz.p.to_numpy()
    if len(tt)<5:continue
    cf={'sheet':sheet,'column':j,'platform':plat,'source':source,'stage':stage,'monitor':mon,'monitor_platform':monplat,'mapping':mapping,'header':label,'unit':unit,'absolute':bool(absolute),'t':tt,'p':pp,'dt':float(np.median(np.diff(tt))),'n_raw':int(good.sum()),'self_pressure':bool(source and source==mon and plat==monplat)}
    r['channels'].append(cf);cols.append(j)
   audit.update(status='已提取' if cols else '未提取压力通道',reason=f'{len(cols)}个通道')
  except Exception as e:audit.update(status='解析失败',reason=str(e))
  r['sheet_audit'].append(audit)
 pd.to_pickle(r,dest);return r

def main():
 fs=[Path(x) for x in pd.read_csv(OLD/'pressure_inventory.csv').path];result=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(parse,fs)):
   result.append(r)
   if i%100==0:print('READ',i,'/',len(fs),flush=True)
 pd.to_pickle(result,B/'pressure_all.pkl')
 pd.DataFrame([{k:v for k,v in r.items() if k not in ['channels','sheet_audit']}|{'channel_count':len(r['channels'])} for r in result]).to_csv(O/'01_全量文件覆盖.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame([{'file':r['path'],**s} for r in result for s in r['sheet_audit']]).to_csv(O/'02_逐工作表读取审计.csv',index=False,encoding='utf-8-sig')
 print('DONE_FILES',len(result),'CHANNELS',sum(len(r['channels']) for r in result),'EMPTY',sum(not r['channels'] for r in result),flush=True)
if __name__=='__main__':main()
