import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
p=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915')
spec=importlib.util.spec_from_file_location('ingest',p/'01_ingest.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fs=[Path(x) for x in pd.read_csv(p/'pressure_inventory.csv').path]
r=[]
with ThreadPoolExecutor(max_workers=4) as ex:
 for i,x in enumerate(ex.map(lambda f:m.parse_file(f,'pressure'),fs)):
  r.append(x)
  if i%200==0:print(i,flush=True)
pd.to_pickle(r,p/'pressure_parsed.pkl')
pd.DataFrame([{k:v for k,v in x.items() if k!='channels'}|{'channel_count':len(x['channels'])} for x in r]).to_csv(p/'pressure_inventory.csv',index=False,encoding='utf-8-sig')
print('DONE',len(r),sum(len(x['channels']) for x in r),sum(not x['channels'] for x in r),flush=True)
