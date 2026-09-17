from pathlib import Path
import sys,re
import pandas as pd,numpy as np
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');O=B.parent/'天然裂缝特征_20260916';cl=pd.read_pickle(B.parent/'时序聚类_V3_形态与强度/results.pkl')
s=pd.read_csv(O/'stages.csv');e=pd.read_pickle(B/'engineering_candidates.pkl');l=pd.read_pickle(B/'legacy_candidates.pkl');known=set(map(tuple,pd.concat([s[['platform','well']],e[['platform','well']],l[['platform','well']]]).drop_duplicates().to_numpy()))
def num(v):
 try:
  x=float(v);return int(x) if np.isfinite(x) and x==int(x) and x>0 else None
 except:return None
out=[]
for _,r in cl.iterrows():
 p,w,t,n,m=r.platform_id,num(r.source_well),num(r.stage_id),r.monitor_platform,num(r.monitor_well);why=[];correction=''
 if (p,w) not in known:
  hit=re.search(r'(阳101H32|H32)\s*[- ]?(\d+)井\s*(\d+)段',Path(r.file).name,re.I)
  if hit and ('阳101H32',int(hit[2])) in known:p,w,t='阳101H32',int(hit[2]),int(hit[3]);correction='文件名明确标注H32源井及段号，修正原解析拼接井号'
 if w is None or w>20:why.append('源井编号异常或缺失')
 if t is None:why.append('缺可靠段号')
 if m is None or not r.well_identity_known:why.append('邻井/传感器身份未确认')
 if not r.event_identity_known:why.append('施工事件身份待确认')
 out.append(dict(sample_id=r.sample_id,platform=p,well=w,stage=t,monitor_platform=n,monitor_well=m,join_valid=not why,join_issue=';'.join(why),identity_correction=correction,original_platform=r.platform_id,original_source_well=r.source_well,morphology_code=int(r.morphology_code_v3),original_class=r.morphology_v3,five_class=r.morphology_v3 if r.morphology_code_v3<=5 else '待归入5类',response_strength=r.response_strength_grade,morphology_uncertain=bool(r.morphology_uncertain),source=r.file,sheet=r.sheet))
d=pd.DataFrame(out);d.to_pickle(B/'classification_linkage.pkl');d.to_csv(B/'classification_linkage.csv',index=False,encoding='utf-8-sig');print('LINK',d.join_valid.value_counts().to_dict(),'CORRECTED',sum(d.identity_correction!=''),'FIVE',sum(d.join_valid&(d.morphology_code<=5)));print(d[~d.join_valid].join_issue.value_counts().to_dict())
