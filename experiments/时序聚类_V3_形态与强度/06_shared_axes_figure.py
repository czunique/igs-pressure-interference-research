"""Additional figure: identical physical axes for the five existing V3 examples."""
import ast,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V3_形态与强度');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V3_形态与强度');OLD=B.parent/'时序聚类_V2_20260915'
spec=importlib.util.spec_from_file_location('report',B/'03_report.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
d=pd.read_pickle(B/'results.pkl');X=np.load(B/'pressure_delta101.npy');raw=pd.read_pickle(OLD/'raw_curves.pkl');proto=pd.read_csv(O/'08_典型曲线来源.csv');ct=pd.crosstab(d.morphology_code_v3,d.response_strength_grade).reindex(index=range(1,10),columns=['弱','中','强'],fill_value=0)
indices=proto[proto.code.between(1,5)].model_row.astype(int).tolist();duration=[];pressure=[]
for idx in indices:
 r=d.iloc[idx];o=raw[int(r.matrix_row)];duration.extend([(o['t'][-1]-o['t'][0])/60,r.window_duration_min]);pressure.extend([o['p']-r.p0_window_mpa,X[idx]])
xmax=float(np.ceil(max(duration)/50)*50);pmin=min(float(np.min(a)) for a in pressure);pmax=max(float(np.max(a)) for a in pressure);ymin=float(min(-1,np.floor(pmin)));ymax=float(max(1,np.ceil(pmax)));xticks=np.arange(0,xmax+.1,50);yticks=np.arange(ymin,ymax+.1,1)
checks={}
def shared_save(fig,name):
 axes=[a for a in fig.axes if a.get_ylabel()=='ΔP / MPa'];assert len(axes)==5
 for ax in axes:
  ax.set_xlim(0,xmax);ax.set_ylim(ymin,ymax);ax.set_xticks(xticks);ax.set_yticks(yticks)
 # Keep the inherited risk qualification; add the common axis range separately.
 fig.text(.5,.95,f'各子图统一：时间 0–{xmax:g} min；压力变化 {ymin:g}–{ymax:g} MPa',ha='center',fontsize=11,color='#4b5563')
 checks.update(xlim=[0,xmax],ylim=[ymin,ymax],xticks=xticks.tolist(),yticks=yticks.tolist(),five_axes_identical=all(np.array_equal(ax.get_xlim(),[0,xmax]) and np.array_equal(ax.get_ylim(),[ymin,ymax]) for ax in axes),all_observed_points_in_bounds=bool(pmin>=ymin and pmax<=ymax and max(duration)<=xmax),sample_ids=d.iloc[indices].sample_id.tolist())
 mod.save(fig,name)
# Reuse the existing figure layout without running other analysis or altering prior outputs.
main_node=next(n for n in ast.parse((B/'03_report.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='main');function=next(n for n in main_node.body if isinstance(n,ast.FunctionDef) and n.name=='prototype_figure')
namespace=dict(mod.__dict__);namespace.update(d=d,X=X,raw=raw,proto=proto,ct=ct,u=np.linspace(0,1,101),save=shared_save)
exec(compile(ast.Module(body=[function],type_ignores=[]),str(B/'03_report.py'),'exec'),namespace)
namespace['prototype_figure']([1,2,3,4,5],'五种目标形态：统一横纵坐标尺度','07_五种目标形态_统一横纵坐标尺度')
(B/'shared_axes_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(checks,ensure_ascii=False))
