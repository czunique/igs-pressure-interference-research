from pathlib import Path
import pandas as pd,numpy as np,importlib.util,json
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V3_形态与强度');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V3_形态与强度')
p=B/'02_validate_models.py';s=p.read_text(encoding='utf-8-sig');needle=" if len(clean):target=clean\n mid=";replacement=""" if len(clean):target=clean
 # Select clear, actual examples for illustration; population medians remain calculated on all class members.
 clear=target
 if code==1:clear=target[(target.positive_rise_p95_mpa>=.03)&target.amplitude_mpa.between(.04,.15)]
 if code==3:clear=target[target.middle_shape_rate>1.7*target[['early_shape_rate','late_shape_rate']].max(axis=1)]
 if code==5:clear=target[(target.minimum_dp_mpa<-.15)&(target.dip_depth_fraction>.15)&(target.positive_rise_p95_mpa>.2)]
 if len(clear):target=clear
 mid=""";s=s.replace(needle,replacement);p.write_text(s,encoding='utf-8')
# Run only revised example selection; completed classifications and sensitivity results stay fixed.
d=pd.read_pickle(B/'results.pkl');sp=importlib.util.spec_from_file_location('v3',B/'01_classify.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);block=s[s.index('proto=[]'):s.index("pd.DataFrame(proto).to_csv")];exec(block);pd.DataFrame(proto).to_csv(O/'08_典型曲线来源.csv',index=False,encoding='utf-8-sig')
p=B/'03_report.py';s=p.read_text(encoding='utf-8-sig');prefix=s.split('def main():')[0];body=s[s.index(' d=pd.read_pickle'):s.index(' # Cross map')];body='\n'.join(line[1:] if line.startswith(' ') else line for line in body.splitlines());body=body.replace("prototype_figure([6,7,8,9],'实际数据中的补充形态与待判记录','02_补充形态与实测曲线')",'');exec(prefix+body)
p=O/'V3_图示形态与三级响应分级报告.md';s=p.read_text(encoding='utf-8');note='\n图中典型曲线是按低拟合误差、身份可追溯及形态清楚程度选取的实际示例，不是每类的中位曲线；图右侧统计来自该类全部成员。M1示例显示微弱非零涨幅，M3示例突出中期速率峰值，M5示例显示超过0.15MPa的初始下降。全部曲线分布另见图06。\n'
if note.strip() not in s:s=s.replace('## 1. 实际分类结果与多对一映射',note+'\n## 1. 实际分类结果与多对一映射');p.write_text(s,encoding='utf-8')
# Controlled mathematical cases test intended definitions; these are tests only, not research observations.
u=np.linspace(0,1,181);sig=1/(1+np.exp(-12*(u-.5)));sig=(sig-sig[0])/(sig[-1]-sig[0]);curves={1:.02*u,2:5*(1-np.exp(-6*u))/(1-np.exp(-6)),3:5*sig,4:5*u,5:5*(-.3*(1-np.exp(-u/.05))+1.3*(np.maximum(u-.3,0)/.7)**1.5),6:5*u**3,7:np.where(u<=.5,10*u,5-8*(u-.5)),8:-5*u}
rows=[];yy=[]
for code,c in curves.items():
 o={'smooth_t':u*180*60,'smooth_p':50+c};r=pd.Series(dict(baseline_noise_mpa=.005,baseline_mpa=50));f,x,y=m.extract(o,r);rows.append(f);yy.append(y)
labs,*_=m.assign(pd.DataFrame(rows),np.array(yy));checks=[dict(expected=int(c),observed=int(l),passed=bool(c==l)) for c,l in zip(curves,labs)];(B/'controlled_shape_tests.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(checks);assert all(z['passed'] for z in checks)
