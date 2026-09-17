from pathlib import Path
p=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916/07_pair_features.py')
s=p.read_text(encoding='utf-8-sig')
insert='''def covered_area(poly,xmin,xmax,ymin,ymax):
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
'''
s=s.replace('def metrics(e,box,center,u,v,stress,bearing):',insert+'def metrics(e,box,center,u,v,stress,bearing):')
s=s.replace("area_m2=None,theta_pair_deg=None", "area_m2=None,full_window_area_m2=None,coverage_fraction=None,theta_pair_deg=None")
s=s.replace("box=[-200,200,min(0,cross)-50,max(0,cross)+50]", "box=[min(-200,along-50),max(200,along+50),min(0,cross)-50,max(0,cross)+50]")
old="""   if abs(along)>200:issues.append('邻井最近点超出沿井窗口')
   elif not (np.all((corners[:,0]>=X.min())&(corners[:,0]<=X.max())) and np.all((corners[:,1]>=Y.min())&(corners[:,1]<=Y.max()))):issues.append('井间窗口超出层位覆盖范围')
   else:"""
new="""   observed_area=covered_area(corners,X.min(),X.max(),Y.min(),Y.max())
   if observed_area<=0:issues.append('井间窗口无层位覆盖')
   else:"""
assert old in s;s=s.replace(old,new)
s=s.replace("development_km_km2=result['length']/area*1000,area_m2=area", "development_km_km2=result['length']/observed_area*1000,area_m2=observed_area,full_window_area_m2=area,coverage_fraction=min(1.,observed_area/area)")
s=s.replace("edge_cache[record['feature_key']]=ed.tolist()", "record['status']+=('；窗口覆盖不足95%' if observed_area/area<.95 else '');edge_cache[record['feature_key']]=ed.tolist()")
p.write_text(s,encoding='utf-8')
