import pandas as pd,sys,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915');s=importlib.util.spec_from_file_location('r',B/'01_recover.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.C=B/'cache_header_retry';m.C.mkdir(exist_ok=True)
a=pd.read_pickle(B/'pressure_all.pkl');lookup={r['path']:r for r in a};targets=[Path(r['path']) for r in a if any(z['status'] in ['未提取压力通道','解析失败'] for z in r['sheet_audit'])];n=0;aud=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(m.parse,targets):
  old=lookup[r['path']]
  for c in r['channels']:
   if not any(c['sheet']==x['sheet'] and (c['monitor']==x['monitor'] or c['header']==x['header']) for x in old['channels']):old['channels'].append(c);n+=1
  aud.extend([dict(file=r['path'],**z) for z in r['sheet_audit']])
pd.to_pickle(a,B/'pressure_all.pkl');pd.DataFrame(aud).to_csv(O/'02c_剩余表头重读.csv',index=False,encoding='utf-8-sig');pd.DataFrame([{k:v for k,v in r.items() if k not in ['channels','sheet_audit']}|{'channel_count':len(r['channels'])} for r in a]).to_csv(O/'01_全量文件覆盖.csv',index=False,encoding='utf-8-sig');print('TARGETS',len(targets),'NEW_CHANNELS',n,'EMPTY',sum(not r['channels'] for r in a))
