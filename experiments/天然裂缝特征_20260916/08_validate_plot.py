from pathlib import Path
import ast,json,sys
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');O=Path('F:/论文库/IGS/实验/结论/天然裂缝特征_20260916');D=O/'图';D.mkdir(exist_ok=True)
tree=ast.parse((B/'07_pair_features.py').read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['clip_edges','angle','covered_area','metrics']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'07_pair_features.py','exec'),globals())
E=np.load(B/'seismic/edges_reference_no_guard.npy');ref=pd.read_csv('E:/code/天然裂缝和压窜积液量的关系/PPT_天然裂缝定量表征_0915/stage_fracture_metrics.csv');errors=[]
for r in ref.itertuples():
 m,_=metrics(E,[r.x-200,r.x+200,r.y-200,r.y+200],np.zeros(2),np.array([1.,0]),np.array([0.,1]),r.stress_azimuth,0)
 errors.append(dict(well=r.well_id,stage=r.stage,length_error_m=m['length']-r.length_m,theta_error_deg=(m['theta']-r.theta_deg) if m['theta'] is not None else 0))
a=pd.DataFrame(errors);assert abs(a.length_error_m).max()<1e-6;assert abs(a.theta_error_deg).max()<1e-6
# Verify the source-based H5 stage coordinates against the independently cached reference.
st=pd.read_csv(B/'stage_geometry.csv');h5=st[st.platform=='泸201H5'].copy();h5['well_id']='H5-'+h5.well.astype(str);merged=h5.merge(ref,on=['well_id','stage'],validate='one_to_one');err=float(np.hypot(merged.east-merged.x,merged.north-merged.y).max());assert err<1e-6
assert abs(covered_area([[0,0],[2,0],[2,2],[0,2]],1,3,1,3)-1)<1e-10
q=pd.read_pickle(B/'pair_features.pkl');calc=q[q.line_length_m.notna()];assert len(calc)==294;assert np.allclose(calc.line_length_m/calc.area_m2*1000,calc.development_km_km2);assert len(calc[(calc.line_length_m==0)&calc.theta_deg.notna()])==0
checks={'reference_stages_reproduced':len(a),'max_length_error_m':float(abs(a.length_error_m).max()),'max_angle_error_deg':float(abs(a.theta_error_deg).max()),'stage_xy_max_error_m':err,'density_recalculation_pass':True,'zero_length_angle_undefined_pass':True,'full_pressure_mapping_pass':int(q.pressure_curves.sum())==3180,'partial_coverage_rows':int((calc.coverage_fraction<.95).sum()),'minimum_coverage_fraction':float(calc.coverage_fraction.min())};(B/'scientific_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
E=np.load(B/'seismic/edges_0.2.npy');A=np.load(B/'seismic/horizon_arrays.npz');background=A['MCANT_nearest_0ms'].copy();background[:2]=np.nan;background[-2:]=np.nan;background[:,:2]=np.nan;background[:,-2:]=np.nan;origin=np.array([A['X'].min(),A['Y'].min()]);tr=pd.read_pickle(B/'trajectories.pkl');example=json.loads((B/'example.json').read_text(encoding='utf-8'));corners=np.array(example['corners'])-origin;ed=np.array(example['edges']).reshape(-1,2,2)-origin;cent=np.array(example['center'])-origin;near=np.array(example['neighbor'])-origin
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
for ax in axes:
 im=ax.pcolormesh(A['X']-origin[0],A['Y']-origin[1],background,cmap='Greys',vmin=-1,vmax=1,shading='auto',rasterized=True,alpha=.55)
 ax.add_collection(LineCollection(E.reshape(-1,2,2)-origin,colors='#cf573e',linewidths=.9,label='阈值0.2提取线段'))
 for wi,g in tr[tr.platform=='泸201H5'].groupby('well'):
  ax.plot(g.east-origin[0],g.north-origin[1],lw=1.5,label=f'H5-{wi}井轨迹')
 ax.plot(*np.vstack([corners,corners[0]]).T,color='#3459a8',lw=1.8,ls='--',label='井间计算窗口');ax.set_aspect('equal');ax.set_xlabel('相对东坐标 / m');ax.set_ylabel('相对北坐标 / m')
axes[0].set_title('泸201H5：参考层位骨架与井轨迹');axes[0].legend(fontsize=8,loc='lower left');axes[1].set_title('H5-2第2段与H5-1：井间窗口');axes[1].set_xlim(corners[:,0].min()-120,corners[:,0].max()+120);axes[1].set_ylim(corners[:,1].min()-120,corners[:,1].max()+120);axes[1].add_collection(LineCollection(ed,colors='#a51d2d',linewidths=2.8));axes[1].scatter(*cent,c='#3459a8',s=45,zorder=5);axes[1].scatter(*near,c='#419065',s=45,zorder=5);m=example['metrics'];axes[1].text(.02,.02,f"逼近角（方向假设）：{m['theta_deg']:.1f}°\n窗口线长：{m['line_length_m']:.1f} m\n发育程度：{m['development_km_km2']:.2f} km/km²",transform=axes[1].transAxes,bbox={'facecolor':'white','alpha':.92,'edgecolor':'#ddd'},fontsize=10)
fig.suptitle('天然裂缝特征提取核查：地震解释线段',fontsize=15);fig.savefig(D/'01_裂缝提取与井间窗口核查.png',dpi=190);fig.savefig(D/'01_裂缝提取与井间窗口核查.pdf');plt.close(fig)
# Pair-specific heatmaps, use a shared scale and mask unavailable neighbor=self cells.
fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained');vmax=float(calc.development_km_km2.max())
for wi,ax in enumerate(axes.flat,1):
 z=calc[calc.source_well==f'泸201H5-{wi}'].copy();z['stage_num']=z.stage.astype(int);z['neighbor_num']=z.neighbor.str.extract(r'-(\d+)$').astype(int);mat=z.pivot(index='neighbor_num',columns='stage_num',values='development_km_km2');im=ax.imshow(mat,aspect='auto',cmap='YlOrRd',vmin=0,vmax=vmax);ax.set_xticks(range(0,len(mat.columns),2),mat.columns[::2]);ax.set_yticks(range(len(mat.index)),[f'H5-{x}' for x in mat.index]);ax.set_title(f'压裂井 H5-{wi}');ax.set_xlabel('设计施工段');ax.set_ylabel('邻井')
fig.colorbar(im,ax=axes,label='线长/有效覆盖面积 / (km/km²)',shrink=.85);fig.suptitle('泸201H5各井段—邻井的裂缝发育程度代理',fontsize=15);fig.savefig(D/'02_各井段邻井发育程度.png',dpi=190);fig.savefig(D/'02_各井段邻井发育程度.pdf');plt.close(fig)
print(json.dumps(checks));print(calc[['theta_deg','line_length_m','development_km_km2','coverage_fraction']].describe().to_string())
