from pathlib import Path
import json,ast,sys,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from adjustText import adjust_text
sys.stdout.reconfigure(encoding='utf-8');np.random.seed(20260916)
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');D=B/'地震叠合图';O=Path('F:/论文库/IGS/实验/结论/天然裂缝特征_20260916/各地震数据_裂缝与井轨迹叠合');O.mkdir(exist_ok=True)
tr=pd.read_pickle(B/'trajectories.pkl');qa=pd.read_csv(B/'trajectory_checks.csv');st=pd.read_csv(B/'stage_geometry.csv')
groups=next(ast.literal_eval(n.value) for n in ast.parse((B/'07_pair_features.py').read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='groups' for t in n.targets))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
colors=plt.get_cmap('tab20');allplatforms=sorted(tr.platform.unique());palette={p:(colors(i) if i<20 else plt.get_cmap('tab20b')(i-20)) for i,p in enumerate(allplatforms)}
palette={p:tuple(v*.68 for v in c[:3])+(1.,) for p,c in palette.items()}
def east(x):
 x=np.asarray(x,dtype=float);return np.where((x>18000000)&(x<19000000),x-18000000,x)
def short(w):return w.replace('泸','L').replace('阳','Y')
manifest=[];well_audit=[]
with PdfPages(O/'全部7组_裂缝指示属性与井轨迹叠合.pdf') as pdf:
 for index,(group,expected) in enumerate(groups.items(),1):
  z=np.load(D/(group+'.npz'));meta=json.loads((D/(group+'.json')).read_text(encoding='utf-8'));xx=east(z['x']);yy=z['y'];il=np.unique(z['il']);xl=np.unique(z['xl']);ii=np.searchsorted(il,z['il']);jj=np.searchsorted(xl,z['xl']);assert len(np.unique(ii*len(xl)+jj))==len(ii)
  sample=np.unique(np.linspace(0,len(ii)-1,min(10000,len(ii))).astype(int));design=np.column_stack([np.ones(len(sample)),ii[sample],jj[sample]]);cx=np.linalg.lstsq(design,xx[sample],rcond=None)[0];cy=np.linalg.lstsq(design,yy[sample],rcond=None)[0];R,C=np.indices((len(il),len(xl)));GX=cx[0]+cx[1]*R+cx[2]*C;GY=cy[0]+cy[1]*R+cy[2]*C;grid_error=float(np.max(np.hypot(GX[ii,jj]-xx,GY[ii,jj]-yy)));assert grid_error<2
  values=np.full(GX.shape,np.nan);values[ii,jj]=z['positive_mean'];horizon=group=='泸201H5'
  if horizon:
   a=np.load(B/'seismic/horizon_arrays.npz');GX=east(a['X']);GY=a['Y'];values=a['MCANT_nearest_0ms'].copy();values[:2]=np.nan;values[-2:]=np.nan;values[:,:2]=np.nan;values[:,-2:]=np.nan
   vmin=-1;vmax=1;cmap='Greys';mode='O3w参考层位：MCANT属性与提取骨架';colorlabel='MCANT原始属性值'
  else:
   finite=values[np.isfinite(values)];vmin=0;vmax=max(float(np.quantile(finite,.99)),1e-6);cmap='bone_r';mode=f"全时窗正属性均值投影：{meta['start'][0]}—{meta['start'][0]+(meta['ns']-1)*meta['dt']/1000:g} ms";colorlabel='正MCANT沿时窗平均值（无量纲原始属性）'
  xmin,xmax=float(GX.min()),float(GX.max());ymin,ymax=float(GY.min()),float(GY.max());shown=[];unlocated=[];outside=[]
  for key,g in tr.groupby(['platform','well']):
   w=f'{key[0]}-{key[1]}';valid=g.east.notna()&g.north.notna();q=qa[(qa.platform==key[0])&(qa.well==key[1])]
   if valid.sum()<2:
    if key[0] in expected:unlocated.append(w);well_audit.append(dict(group=group,well=w,plotted=False,reason='缺绝对坐标或有效轨迹'))
    continue
   g=g[valid].sort_values('md');x=east(g.east.to_numpy());y=g.north.to_numpy();inside=(x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax)
   if not inside.any():
    if key[0] in expected:outside.append(w);well_audit.append(dict(group=group,well=w,plotted=False,reason='轨迹不在SGY平面包络内'))
    continue
   review=bool(q.geometry_review_required.iloc[0]) if len(q) else True;shown.append((key,g,x,y,inside,review));well_audit.append(dict(group=group,well=w,plotted=True,reason='轨迹一致性待核查' if review else '可定位；投影基准待确认',included_by='目录指定平台' if key[0] in expected else '空间落入此地震范围'))
  fig=plt.figure(figsize=(16,11),facecolor='white');ax=fig.add_axes([.075,.14,.69,.74]);side=fig.add_axes([.79,.15,.195,.70]);side.axis('off')
  ax.set_facecolor('#f0f1f2');im=ax.pcolormesh(GX/1000,GY/1000,values,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto',rasterized=True,zorder=1)
  if horizon:
   e=np.load(B/'seismic/edges_0.2.npy').reshape(-1,2,2);e[:,:,0]=east(e[:,:,0]);ax.add_collection(LineCollection(e/1000,colors='#c23c31',linewidths=.8,zorder=2))
  texts=[]
  for key,g,x,y,inside,review in shown:
   col=palette[key[0]];ls='--' if review else '-';ax.plot(x/1000,y/1000,color='white',lw=3.8,alpha=.8,zorder=3);ax.plot(x/1000,y/1000,color=col,lw=1.9,ls=ls,zorder=4)
   endpoint=np.where(inside)[0][-1];ax.scatter(x[endpoint]/1000,y[endpoint]/1000,s=12,color=col,zorder=5);label=short(f'{key[0]}-{key[1]}')+('*' if review else '')
   texts.append(ax.text(x[endpoint]/1000,y[endpoint]/1000,label,fontsize=7.5,color='#24313a',zorder=6,bbox={'facecolor':'white','edgecolor':'none','pad':.4,'alpha':.75}))
  ax.set_xlim(xmin/1000,xmax/1000);ax.set_ylim(ymin/1000,ymax/1000);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('东坐标 / km（18带号前缀已统一）');ax.set_ylabel('北坐标 / km');ax.tick_params(labelsize=9);ax.grid(color='#7f8c8d',alpha=.15,lw=.4)
  cbar=fig.colorbar(im,ax=ax,location='bottom',fraction=.045,pad=.07,shrink=.75,extend='max' if not horizon else 'neither');cbar.set_label(colorlabel,fontsize=9)
  fig.canvas.draw()
  if texts:adjust_text(texts,ax=ax,expand=(1.08,1.2),force_text=(.12,.25),ensure_inside_axes=True,time_lim=3,arrowprops={'arrowstyle':'-','color':'#777777','lw':.5})
  fig.text(.075,.95,f'{index:02d}  天然裂缝指示属性与井轨迹叠合',fontsize=18,weight='bold',color='#233a4d');fig.text(.075,.915,group.replace('_',' / '),fontsize=12,color='#233a4d');fig.text(.075,.889,mode,fontsize=10,color='#626b73')
  plotted_platforms=sorted({k[0] for k,*_ in shown});counts={p:sum(k[0]==p for k,*_ in shown) for p in plotted_platforms};handles=[Line2D([0],[0],color=palette[p],lw=2,label=f'{p}  {counts[p]}口井') for p in plotted_platforms]
  side.text(0,1,f'范围内展示 {len(shown)} 口井\n共 {len(plotted_platforms)} 个平台',va='top',fontsize=12,weight='bold');side.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.04,.91),frameon=False,fontsize=9,handlelength=2,labelspacing=.65)
  yinfo=.86-.035*len(handles);notes=['井名：L=泸，Y=阳','虚线及*：轨迹一致性待核查','不同颜色区分平台','未将井深当作地震时间']
  if horizon:notes+=['红线：阈值0.2的解释骨架','网格外缘40 m已剔除']
  else:notes+=['浅灰空区：没有对应地震道','色阶上限采用本数据组P99','本图不是目标层位裂缝图']
  side.text(0,yinfo,'\n'.join(notes),va='top',fontsize=9,linespacing=1.6)
  absent=[p for p in expected if p not in plotted_platforms]
  missingtext=f'缺坐标或有效轨迹：{len(unlocated)}口\n目录所列但位于范围外：{len(outside)}口'
  if absent:missingtext+='\n未展示平台：'+'、'.join(absent)
  missingtext+='\n逐井原因见配套CSV清单'
  wrapped='\n'.join(textwrap.fill(line,width=23) for line in missingtext.splitlines());side.text(0,min(yinfo-.24,.28),wrapped,va='top',fontsize=8,color='#6b4f41',linespacing=1.4)
  footer='平面叠合采用现有坐标及去带号假设，投影基准仍待确认。'
  footer+=('参考层位解释线不等于已验证的天然裂缝。' if horizon else '全时窗投影叠加了不同深度响应，仅用于空间概览；不能据此判断与射孔段连通。')
  fig.text(.075,.035,footer,fontsize=9,color='#555555');name=f'{index:02d}_{group}_裂缝指示属性与井轨迹.png';fig.savefig(O/name,dpi=210,facecolor='white');pdf.savefig(fig,dpi=170);plt.close(fig)
  manifest.append(dict(group=group,image=name,mode=mode,source=meta['path'],wells_shown=len(shown),platforms_shown=counts,unlocated_wells=unlocated,outside_wells=outside,expected_platforms_without_display=absent,grid_shape=list(values.shape),missing_grid_cells=int(np.isnan(values).sum()),affine_coordinate_max_error_m=grid_error,color_max=vmax));print('PLOTTED',group,'wells',len(shown),'platforms',counts,flush=True)
(D/'overlay_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');pd.DataFrame(well_audit).to_csv(O/'各地震数据_井轨迹展示清单.csv',index=False,encoding='utf-8-sig')
(O/'图件说明.md').write_text('# 裂缝指示属性与井轨迹叠合图\n\n每组SGY属性数据输出一张PNG，七图合并于PDF。除目录标注的平台外，现有坐标轨迹空间上落入地震包络的其他平台也一并展示。\n\n泸201H5使用O3w参考层位MCANT及阈值0.2提取骨架（边界两格已剔除）。其他六组缺层位，展示全时窗mean(max(MCANT,0))投影，保留地震道缺口；并未从该投影生成井段裂缝特征或连通结论。\n\n除泸201H5采用原始属性固定色阶外，其余六图色阶按本组P99截色，超过上限使用最深色；原始数值在过程NPZ中保留。不同图颜色深浅不能作为跨平台裂缝发育程度比较。\n\n井轨迹按东/北坐标叠合；东坐标在1800万至1900万范围的去除1800万带号前缀。此为有标记的坐标约定，非已确认的坐标投影转换。\n\n缺坐标/轨迹的井未填造；已定位但一致性检查待核查的井以虚线及星号显示。逐井清单说明未显示原因。\n',encoding='utf-8')
