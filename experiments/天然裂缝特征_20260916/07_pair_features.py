from pathlib import Path
import json,sys,ast,struct,hashlib
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');O=Path('F:/论文库/IGS/实验/结论/天然裂缝特征_20260916');REF=Path('E:/code/天然裂缝和压窜积液量的关系')
def clip_edges(edges,box):
 e=edges.copy();d=e[:,2:]-e[:,:2];lo=np.zeros(len(e));hi=np.ones(len(e));valid=np.ones(len(e),bool)
 for axis,lower,upper in [(0,box[0],box[1]),(1,box[2],box[3])]:
  for p,q in [(-d[:,axis],e[:,axis]-lower),(d[:,axis],upper-e[:,axis])]:
   zero=abs(p)<1e-12;valid&=~(zero&(q<0));ratio=np.divide(q,p,out=np.zeros_like(q),where=~zero);lo=np.where(p<0,np.maximum(lo,ratio),lo);hi=np.where(p>0,np.minimum(hi,ratio),hi)
 valid&=hi-lo>1e-9
 return np.column_stack((e[valid,:2]+lo[valid,None]*d[valid],e[valid,:2]+hi[valid,None]*d[valid]))
def angle(a,b):return np.degrees(np.arccos(np.clip(abs(np.cos(np.radians(a-b))),0,1)))
def covered_area(poly,xmin,xmax,ymin,ymax):
 pts=[np.asarray(v) for v in poly]
 for axis,bound,sign in [(0,xmin,1),(0,xmax,-1),(1,ymin,1),(1,ymax,-1)]:
  result=[]
  if not pts:return 0.
  prev=pts[-1];inside_prev=sign*(prev[axis]-bound)>=-1e-8
  for curr in pts:
   inside=sign*(curr[axis]-bound)>=-1e-8
   if inside!=inside_prev:
    frac=(bound-prev[axis])/(curr[axis]-prev[axis]);result.append(prev+frac*(curr-prev))
   if inside:result.append(curr)
   prev=curr;inside_prev=inside
  pts=result
 if len(pts)<3:return 0.
 a=np.array(pts);a-=a[0];return float(abs(np.dot(a[:,0],np.roll(a[:,1],-1))-np.dot(a[:,1],np.roll(a[:,0],-1)))/2)
def metrics(e,box,center,u,v,stress,bearing):
 loc=np.column_stack(((e[:,:2]-center)@u,(e[:,:2]-center)@v,(e[:,2:]-center)@u,(e[:,2:]-center)@v));cl=clip_edges(loc,box)
 if not len(cl):return dict(length=0.,theta=None,theta_pair=None,edges=0,theta_sd=None),np.empty((0,4))
 a=center+cl[:,:1]*u+cl[:,1:2]*v;b=center+cl[:,2:3]*u+cl[:,3:4]*v;world=np.column_stack((a,b));lens=np.linalg.norm(b-a,axis=1);mid=(a+b)/2;directions=b-a;tree=cKDTree(mid)
 for i,neighbors in enumerate(tree.query_ball_point(mid,60)):
  if len(neighbors)>=3:val,vec=np.linalg.eigh(np.cov(mid[neighbors].T));directions[i]=vec[:,-1]
 az=np.degrees(np.arctan2(directions[:,0],directions[:,1]))%180;angles=angle(az,stress);theta=np.average(angles,weights=lens)
 return dict(length=float(lens.sum()),theta=float(theta),theta_pair=float(np.average(angle(az,bearing),weights=lens)),edges=len(lens),theta_sd=float(np.sqrt(np.average((angles-theta)**2,weights=lens)))),world
# Independent segment clipping boundary checks.
assert np.allclose(clip_edges(np.array([[-2.,0,2,0],[0,-2,0,2],[2,2,3,3]]),[-1,1,-1,1]),[[-1,0,1,0],[0,-1,0,1]])
assert np.allclose(angle(np.array([0,90,180]),0),[0,90,0])
st=pd.read_csv(B/'stages.csv');tr=pd.read_pickle(B/'trajectories.pkl');wh=pd.read_csv(B/'wellheads.csv');P=pd.read_pickle(B.parent/'时序聚类_V3_形态与强度/results.pkl')
# Coordinate cross-check against independent minimum-curvature integration.
qa=[];trajectories={}
for key,g in tr.groupby(['platform','well']):
 g=g.sort_values('md');md=g.md.to_numpy();inc=np.radians(g.inc.to_numpy());az=np.radians(g.az.to_numpy());dmd=np.diff(md);beta=np.arccos(np.clip(np.cos(inc[:-1])*np.cos(inc[1:])+np.sin(inc[:-1])*np.sin(inc[1:])*np.cos(np.diff(az)),-1,1));rf=np.divide(2*np.tan(beta/2),beta,out=np.ones_like(beta),where=abs(beta)>1e-10)
 de=.5*dmd*(np.sin(inc[:-1])*np.sin(az[:-1])+np.sin(inc[1:])*np.sin(az[1:]))*rf;dn=.5*dmd*(np.sin(inc[:-1])*np.cos(az[:-1])+np.sin(inc[1:])*np.cos(az[1:]))*rf
 diff=np.hypot(g.dx.to_numpy()-g.dx.iloc[0]-np.r_[0,np.cumsum(de)],g.dy.to_numpy()-g.dy.iloc[0]-np.r_[0,np.cumsum(dn)])
 p95=float(np.nanquantile(diff,.95));bad=bool(p95>30 or (g.tvd>g.md+5).any() or g.coordinate_conflict.any());qa.append(dict(platform=key[0],well=int(key[1]),n=len(g),md_min=float(md.min()),md_max=float(md.max()),coordinate_available=bool(g.east.notna().all()),offset_check_p95_m=p95,geometry_review_required=bad,file=g.file.iloc[0]))
 trajectories[key]=(g,bad)
qa=pd.DataFrame(qa);qa.to_csv(B/'trajectory_checks.csv',index=False,encoding='utf-8-sig')
st['east']=np.nan;st['north']=np.nan;st['tvd']=np.nan;st['azimuth']=np.nan;st['geometry_status']='无可解析井轨迹';st['trajectory_file']=''
for i,r in st.iterrows():
 key=(r.platform,r.well)
 if key not in trajectories:continue
 g,bad=trajectories[key];st.loc[i,'trajectory_file']=g.file.iloc[0]
 if r.md_top<g.md.min() or r.md_bottom>g.md.max():st.loc[i,'geometry_status']='分段超出轨迹测深范围';continue
 if not g.east.notna().all():st.loc[i,'geometry_status']='缺井口绝对坐标';continue
 for c in ['east','north','tvd']:st.loc[i,c]=np.interp(r.md_mid,g.md,g[c])
 st.loc[i,'azimuth']=np.degrees(np.interp(r.md_mid,g.md,np.unwrap(np.radians(g.az))))%360
 st.loc[i,'geometry_status']='轨迹一致性待核查' if bad else '已定位'
 if r.stage_conflict:st.loc[i,'geometry_status']='分段版本冲突'
st.to_csv(B/'stage_geometry.csv',index=False,encoding='utf-8-sig')
# Geographic groups are derived from supplied directory names, not from proximity alone.
groups={'泸201H5':['泸201H5'],'泸203H153':['泸203H153'],'泸203H64_66_76':['泸203H64','泸203H66','泸203H76'],'泸203H6_8_9_10_11_泸208H1_2':['泸203H6','泸203H8','泸203H9','泸203H10','泸203H11','泸208H1','泸208H2'],'泸203H91_泸206H8':['泸203H91','泸206H8'],'阳101H23_31_32_33_34_35_37_43':['阳101H23','阳101H31','阳101H32','阳101H33','阳101H34','阳101H35','阳101H37','阳101H43'],'阳101H51_75':['阳101H51','阳101H75']};gm={p:g for g,ps in groups.items() for p in ps}
# One feature record per directed source-stage-observer key; existing pressure records supply observed pairs.
keys={};pmap=[]
def well_value(v):
 try:
  f=float(v);return str(int(f)) if f.is_integer() else str(v)
 except:return str(v)
for r in P.itertuples():
 key=(str(r.platform_id),well_value(r.source_well),well_value(r.stage_id),str(r.monitor_platform),well_value(r.monitor_well));keys.setdefault(key,[]).append(r.sample_id);pmap.append(dict(sample_id=r.sample_id,key='|'.join(key)))
known={}
for r in st.itertuples():known.setdefault(r.platform,set()).add(str(int(r.well)))
for r in wh.itertuples():known.setdefault(r.platform,set()).add(str(int(r.well)))
for (p,w,s,np_,nw) in list(keys):
 if w.isdigit():known.setdefault(p,set()).add(w)
 if nw.isdigit():known.setdefault(np_,set()).add(nw)
for r in st.itertuples():
 for neighbor in known.get(r.platform,[]):
  if neighbor!=str(int(r.well)):keys.setdefault((r.platform,str(int(r.well)),str(int(r.stage)),r.platform,neighbor),[])
st_lookup={(r.platform,str(int(r.well)),str(int(r.stage))):r for r in st.itertuples()}
A=np.load(B/'seismic/horizon_arrays.npz');X=A['X'];Y=A['Y'];E={v:np.load(B/f'seismic/edges_{v:g}.npy') for v in [.1,.2,.3]};records=[];example={};edge_cache={};sensitivity=[]
for key,ids in sorted(keys.items()):
 p,w,s,np_,nw=key;record=dict(platform=p,source_well=p+'-'+w,stage=s,neighbor=np_+'-'+nw,feature_key='|'.join(key),pressure_curves=len(ids),pressure_sample_ids=';'.join(ids),sample_scope='压力观测井对' if ids else '设计候选井对',sgy_group=gm.get(p,''),status='',theta_deg=None,line_length_m=None,development_km_km2=None,area_m2=None,full_window_area_m2=None,coverage_fraction=None,theta_pair_deg=None,theta_sd_deg=None,edge_count=None,east=None,north=None,md_top=None,md_bottom=None,well_azimuth_deg=None,assumed_hf_azimuth_deg=None,geometry_status='',stage_source='',trajectory_source='',horizon_source='',seismic_source='')
 sr=st_lookup.get((p,w,s));issues=[]
 if p not in gm:issues.append('缺配套SGY')
 elif p!='泸201H5':issues.append('缺目标层位或时深转换')
 if sr is None:issues.append('缺可匹配分段区间')
 else:
  record.update(east=sr.east,north=sr.north,md_top=sr.md_top,md_bottom=sr.md_bottom,geometry_status=sr.geometry_status,stage_source=sr.file,trajectory_source=sr.trajectory_file,well_azimuth_deg=sr.azimuth)
  if sr.geometry_status!='已定位':issues.append(sr.geometry_status)
 nk=(np_,int(nw)) if nw.isdigit() else None
 if nk not in trajectories:issues.append('缺邻井轨迹')
 elif not trajectories[nk][0].east.notna().all():issues.append('邻井缺井口坐标')
 elif trajectories[nk][1]:issues.append('邻井轨迹一致性待核查')
 if not issues:
  g=trajectories[nk][0];ns_=st[(st.platform==np_)&(st.well==int(nw))]
  if len(ns_)==0:issues.append('缺邻井水平段测深范围')
  else:
   lo,hi=ns_.md_top.min(),ns_.md_bottom.max();md=np.r_[lo,g[(g.md>lo)&(g.md<hi)].md.to_numpy(),hi];line=np.column_stack((np.interp(md,g.md,g.east),np.interp(md,g.md,g.north)));center=np.array([sr.east,sr.north]);d=line[1:]-line[:-1];q=np.clip(np.sum((center-line[:-1])*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-12),0,1);nearest=line[:-1]+q[:,None]*d;j=np.argmin(np.linalg.norm(nearest-center,axis=1));neighbor=nearest[j];radian=np.radians(sr.azimuth);u=np.array([np.sin(radian),np.cos(radian)]);v=np.array([np.cos(radian),-np.sin(radian)]);cross=(neighbor-center)@v;along=(neighbor-center)@u;box=[min(-200,along-50),max(200,along+50),min(0,cross)-50,max(0,cross)+50];area=(box[1]-box[0])*(box[3]-box[2]);corners=np.array([center+a*u+b*v for a,b in [(box[0],box[2]),(box[1],box[2]),(box[1],box[3]),(box[0],box[3])]])
   observed_area=covered_area(corners,X.min()+40,X.max()-40,Y.min()+40,Y.max()-40)
   if observed_area<=0:issues.append('井间窗口无层位覆盖')
   else:
    stress=(sr.azimuth+90)%180;bearing=np.degrees(np.arctan2(*(neighbor-center)))%180;result,ed=metrics(E[.2],box,center,u,v,stress,bearing);record.update(theta_deg=result['theta'],line_length_m=result['length'],development_km_km2=result['length']/observed_area*1000,area_m2=observed_area,full_window_area_m2=area,coverage_fraction=min(1.,observed_area/area),theta_pair_deg=result['theta_pair'],theta_sd_deg=result['theta_sd'],edge_count=result['edges'],assumed_hf_azimuth_deg=stress,horizon_source='LU205H5-0603-O3w.hrzdat',seismic_source='LU205H5_MCANT.sgy');record['status']='已计算：参考层位/方向假设' if result['edges'] else '已计算：阈值下未检出线段，角度无定义';record['status']+=('；窗口覆盖不足95%' if observed_area/area<.95 else '');edge_cache[record['feature_key']]=ed.tolist()
    for threshold in [.1,.3]:
     alt,_=metrics(E[threshold],box,center,u,v,stress,bearing);sensitivity.append(dict(feature_key=record['feature_key'],threshold=threshold,line_length_m=alt['length'],theta_deg=alt['theta'],relative_length_change=(alt['length']/result['length']-1) if result['length']>0 else None))
    if p=='泸201H5' and w=='2' and s=='2' and nw=='1':example=dict(key=record['feature_key'],corners=corners.tolist(),center=center.tolist(),neighbor=neighbor.tolist(),edges=ed.tolist(),metrics=record)
 if issues:record['status']='；'.join(dict.fromkeys(issues))
 records.append(record)
q=pd.DataFrame(records);q.to_pickle(B/'pair_features.pkl');q.to_csv(O/'天然裂缝_井段邻井特征.csv',index=False,encoding='utf-8-sig');pd.DataFrame(pmap).to_csv(O/'压力样本与裂缝特征关联.csv',index=False,encoding='utf-8-sig');pd.DataFrame(sensitivity).to_csv(B/'threshold_sensitivity.csv',index=False,encoding='utf-8-sig');(B/'pair_edges.json').write_text(json.dumps(edge_cache),encoding='utf-8');(B/'example.json').write_text(json.dumps(example,ensure_ascii=False),encoding='utf-8')
coverage=[]
for p in sorted(set(P.platform_id)|set(st.platform)|set(gm)):
 z=q[q.platform==p];ss=st[st.platform==p];tt=qa[qa.platform==p]
 coverage.append(dict(platform=p,sgy_group=gm.get(p,''),horizon_available=p=='泸201H5',designed_stages=len(ss),survey_wells=len(tt),located_survey_wells=int(tt.coordinate_available.sum()),candidate_pairs=len(z),calculated_pairs=int(z.line_length_m.notna().sum()),pressure_curves=int(z.pressure_curves.sum()),matched_pressure_curves=int(z.loc[z.line_length_m.notna(),'pressure_curves'].sum()),next_input='已提取参考层位代理；需确认水力裂缝方向及层位归属' if p=='泸201H5' else '补目标层位/时深转换及未匹配的井段几何资料' if p in gm else '补配套SGY、目标层位及井段几何资料'))
cov=pd.DataFrame(coverage);cov.to_csv(O/'平台数据覆盖与待补项.csv',index=False,encoding='utf-8-sig');(B/'coverage.json').write_text(cov.to_json(orient='records',force_ascii=False),encoding='utf-8')
assert len(P)==sum(q.pressure_curves)==len(pmap);assert q.feature_key.is_unique;assert q.loc[q.line_length_m.notna(),'theta_deg'].dropna().between(0,90).all();assert (q.line_length_m.dropna()>=0).all();assert set(q.loc[q.line_length_m.notna(),'platform'])=={'泸201H5'}
checks=dict(feature_rows=len(q),calculated_rows=int(q.line_length_m.notna().sum()),unavailable_rows=int(q.line_length_m.isna().sum()),zero_line_rows=int((q.line_length_m==0).sum()),original_pressure_samples=len(P),matched_pressure_samples=int(q.loc[q.line_length_m.notna(),'pressure_curves'].sum()),all_pressure_samples_retained=True,unique_keys=True,angle_range_valid=True,source_data_read_only=True)
(B/'pair_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks));print(q[q.platform=='泸201H5'].status.value_counts().to_string());print(qa[qa.geometry_review_required].to_string(index=False))
