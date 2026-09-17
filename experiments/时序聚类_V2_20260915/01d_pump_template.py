import pandas as pd,re
from pathlib import Path
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');a=pd.read_pickle(B/'pressure_all.pkl')
for r in a:
 if '压窜模板1井' in r['path']:
  for c in r['channels']:
   sm=re.match(r'(\d+)-(\d+)',c['sheet'])
   if sm:
    c['source']=sm.group(1);c['stage']=sm.group(2)
    if c['monitor'] in ['1','5','8']:c['self_pressure']=True
pd.to_pickle(a,B/'pressure_all.pkl')
