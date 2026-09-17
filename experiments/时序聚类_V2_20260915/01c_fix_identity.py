import sys,re,importlib.util
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');s=importlib.util.spec_from_file_location('r',B/'01_recover.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=pd.read_pickle(B/'pressure_all.pkl');audit=[]
for r in a:
 p,src,st=m.identity(Path(r['path']).stem,r['directory_platform'])
 mons=re.search(r'(?:期间|监测)[- ]*([泸阳自]\d+H\d+)',Path(r['path']).stem)
 for c in r['channels']:
  old=(c['platform'],c['source'],c['stage'],c['monitor_platform'])
  if 'h26' in Path(r['path']).stem.lower():c['platform']=p;c['monitor_platform']=p
  if c['platform'] in ['阳101H312','阳101H351','阳101H352','阳101H353']:
   c['platform']=p;c['monitor_platform']=p;c['source']=src;c['stage']=st
  if src and st:c['platform']=p;c['source']=src;c['stage']=st
  if mons:c['monitor_platform']=mons.group(1)
  c['self_pressure']=bool(c['source'] and c['source']==c['monitor'] and c['platform']==c['monitor_platform'])
  if '施工压力' in c['header']:c['self_pressure']=True
  new=(c['platform'],c['source'],c['stage'],c['monitor_platform'])
  if old!=new:audit.append(dict(file=r['path'],sheet=c['sheet'],old=str(old),new=str(new)))
pd.to_pickle(a,B/'pressure_all.pkl');pd.DataFrame(audit).to_csv(B/'identity_corrections.csv',index=False,encoding='utf-8-sig');print('CORRECTED',len(audit))
