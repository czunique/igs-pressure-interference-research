from pathlib import Path
import pandas as pd,numpy as np,json,sys
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');OLD=B.parent/'天然裂缝特征_20260916';p=pd.read_pickle(B/'pair_factors.pkl');nf=pd.read_pickle(OLD/'pair_features.pkl');print(nf[['platform','source_well','stage','neighbor','status']].head(2).to_dict('records'))
def num(v):
 try:return int(float(str(v).rsplit('-',1)[-1]))
 except:return None
fm={(r.platform,num(r.source_well),num(r.stage),r.platform,num(r.neighbor)):r for _,r in nf.iterrows()}
for i,r in p.iterrows():
 sw=str(r['压裂井']).rsplit('-',1);nw=str(r['压窜井']).rsplit('-',1);f=fm.get((sw[0],int(sw[1]),int(r['压裂段']),nw[0],int(nw[1])))
 if f is not None:
  p.loc[i,'天然裂缝逼近角_deg']=f.theta_deg;p.loc[i,'天然裂缝解释线总长_m']=f.line_length_m;p.loc[i,'天然裂缝发育程度_km_km2']=f.development_km_km2;p.loc[i,'天然裂缝状态']=f.status
p.to_pickle(B/'pair_factors.pkl');print('FRACTURES',p['天然裂缝解释线总长_m'].notna().sum())
