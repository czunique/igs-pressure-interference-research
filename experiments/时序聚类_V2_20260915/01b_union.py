import sys,re,importlib.util
from pathlib import Path
import pandas as pd,numpy as np
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915');OLD=B.parent/'时序聚类_20260915'
s=importlib.util.spec_from_file_location('recover',B/'01_recover.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
new=pd.read_pickle(B/'pressure_all.pkl');old=pd.read_pickle(OLD/'pressure_parsed.pkl');lookup={r['path']:r for r in new};audit=[]
for r in old:
 nr=lookup[r['path']]; plat,src,stage=m.identity(Path(r['path']).stem,r['platform'])
 for i,c in enumerate(r['channels']):
  label=c['label'];mon=c['monitor'];mp=plat
  if any(x in label for x in ['油压','B环空','C环空','温度','排量','液量','砂']):continue
  if re.search('No|通道',label,re.I):
   mm=re.search(r'压裂施工[-－](\d+)井',Path(r['path']).stem)
   mon=mm.group(1) if mm else 'sensor:'+label
  if any(x['sheet']==c['sheet'] and (x['monitor']==mon or x['header']==label) for x in nr['channels']):continue
  cs=c.get('sheet_source') or src or str(r.get('source') or '');ct=c.get('sheet_stage') or stage or str(r.get('stage') or '')
  t=c['t'].copy();p=c['p'].copy()
  if c['absolute'] and np.nanmedian(t)<1e8:t*=1000
  nc=dict(sheet=c['sheet'],column='legacy_'+str(i),platform=plat,source=cs,stage=ct,monitor=mon,monitor_platform=mp,mapping='V1_validated_parser_fallback',header=label,unit='MPa_context_assumed',absolute=bool(c['absolute']),t=t,p=p,dt=float(np.median(np.diff(t))),n_raw=len(t),self_pressure=bool(cs and cs==mon))
  nr['channels'].append(nc);audit.append({'file':r['path'],'sheet':c['sheet'],'monitor':mon,'points':len(t),'reason':'新解析器未识别，保留旧解析器有效通道'})
pd.to_pickle(new,B/'pressure_all.pkl');pd.DataFrame(audit).to_csv(O/'02b_双解析器补充.csv',index=False,encoding='utf-8-sig')
pd.DataFrame([{k:v for k,v in r.items() if k not in ['channels','sheet_audit']}|{'channel_count':len(r['channels'])} for r in new]).to_csv(O/'01_全量文件覆盖.csv',index=False,encoding='utf-8-sig')
print('FALLBACK',len(audit),'TOTAL',sum(len(r['channels']) for r in new),'EMPTY',sum(not r['channels'] for r in new))
