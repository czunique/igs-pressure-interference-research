import pandas as pd,sys,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915');s=importlib.util.spec_from_file_location('r',B/'01_recover.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.C=B/'cache_final_retry';m.C.mkdir(exist_ok=True)
a=pd.read_pickle(B/'pressure_all.pkl');lookup={r['path']:r for r in a};target=[Path(r['path']) for r in a if not r['channels'] or any(c['monitor'] in ['sensor:压力','sensor:套压'] for c in r['channels'])];added=0;aud=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(m.parse,target):
  old=lookup[r['path']];old['channels']=[c for c in old['channels'] if c['monitor'] not in ['sensor:压力','sensor:套压']]
  for c in r['channels']:
   if not any(c['sheet']==x['sheet'] and c['monitor']==x['monitor'] and c['header']==x['header'] for x in old['channels']):old['channels'].append(c);added+=1
  aud.extend([dict(file=r['path'],**z) for z in r['sheet_audit']])
for r in a:
 if '自205H1平台5-8井监测数据' in r['path']:
  for c in r['channels']:c['not_timeseries']=True
pd.to_pickle(a,B/'pressure_all.pkl');pd.DataFrame(aud).to_csv(O/'02d_通道字典与拆分时间补读.csv',index=False,encoding='utf-8-sig');pd.DataFrame([{k:v for k,v in r.items() if k not in ['channels','sheet_audit']}|{'channel_count':sum(not c.get('not_timeseries',False) for c in r['channels'])} for r in a]).to_csv(O/'01_全量文件覆盖.csv',index=False,encoding='utf-8-sig');print('TARGET',len(target),'ADDED',added,'VALID_EMPTY',sum(not any(not c.get('not_timeseries',False) for c in r['channels']) for r in a))
