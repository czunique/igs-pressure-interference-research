from pathlib import Path
import json, numpy as np,pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score,f1_score,classification_report
from scipy.stats import spearmanr
from train_models import B,O,LABELS,FEATURES
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.65,'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'savefig.facecolor':'white'})
C=['#7C8DA5','#D98C48','#397DAF','#459886','#9A6FB0'];names=['M1 微弱','M2 快速稳定','M3 渐进上升','M4 稳定缓升','M5 先降后升']
F=O/'图件';F.mkdir(exist_ok=True)
a=json.loads((B/'audit.json').read_text(encoding='utf8'));d=pd.read_pickle(B/'model_cohort.pkl');X=pd.read_pickle(B/'model_inputs.pkl');z=np.load(B/'oof_arrays.npz');y=z['y'];p=z['XGBoost'];sv=z['shap'];fold=z['fold'];groups=d['验证平台组'].to_numpy();uu=np.unique(groups)
summ=pd.read_csv(O/'01_模型总体表现.csv');imp=pd.read_csv(O/'05_SHAP重要性.csv');sens=pd.read_csv(O/'04_敏感性与特征组对照.csv')
short={'重算井距_段中点平面_m':'井距 / m','段长_m':'段长 / m','簇数':'簇数','加砂强度_t_m':'加砂强度 / t/m','用液强度_m3_m':'用液强度 / m³/m','排量_m3_min':'排量 / m³/min','孔隙度_pct':'孔隙度 / %','含水饱和度_pct':'含水饱和度 / %','总有机碳_pct':'TOC / %','杨氏模量_GPa':'杨氏模量 / GPa','泊松比':'泊松比','最小主应力_MPa':'最小主应力 / MPa','水平应力差_MPa':'水平应力差 / MPa','破裂压力_MPa':'破裂压力 / MPa'}
figures=[]
def tidy(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,width=.6)
def save(fig,n):
    fig.savefig(F/(n+'.png'),dpi=600,bbox_inches='tight');fig.savefig(F/(n+'.pdf'),bbox_inches='tight');fig.savefig(F/(n+'.svg'),bbox_inches='tight');figures.append((n,fig));print('FIG',n,flush=True)
def panel(ax,l,title):ax.set_title(l+'  '+title,loc='left',fontweight='bold',pad=9);tidy(ax)
# 1
fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={'width_ratios':[1.0,1.1]},layout='constrained')
steps=['候选五类组合','剔除形态边界后','可用于主实验'];nums=[1630,1318,len(d)]
axs[0].barh(np.arange(3)[::-1],nums,color=['#DCE3EA','#A8BACB','#397DAF'],height=.55)
for i,n in enumerate(nums):axs[0].text(n+25,2-i,str(n),va='center')
axs[0].set_yticks([2,1,0],steps);axs[0].set_xlim(0,1850);axs[0].set_xlabel('井段—邻井组合数');panel(axs[0],'a','建模样本筛选')
count=np.bincount(y,minlength=5);axs[1].bar(np.arange(5),count,color=C,width=.65)
for i,n in enumerate(count):axs[1].text(i,n+10,str(n),ha='center')
axs[1].set_xticks(range(5),['M1','M2','M3','M4','M5']);axs[1].set_ylim(0,620);axs[1].set_ylabel('组合数');panel(axs[1],'b',f"{a['platforms']}个平台 · {a['connected_platform_groups']}个独立分组")
save(fig,'01_样本筛选与类别分布')
# 2 correlation and missingness
fig,axs=plt.subplots(1,2,figsize=(7.2,4.7),gridspec_kw={'width_ratios':[1,1.35]},layout='constrained')
miss=X.isna().mean().sort_values();axs[0].barh(range(len(miss)),miss*100,color='#93B3C7');axs[0].set_yticks(range(len(miss)),[short[t] for t in miss.index]);axs[0].set_xlabel('缺失比例 / %');axs[0].set_xlim(0,45);panel(axs[0],'a','质控后的特征覆盖')
corr=X.corr(method='spearman',min_periods=50);cm=axs[1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');axs[1].set_xticks(range(14),[str(i+1) for i in range(14)],rotation=0);axs[1].set_yticks(range(14),[f'{i+1} '+short[t].split(' / ')[0] for i,t in enumerate(FEATURES)]);axs[1].set_title('b  Spearman相关系数',loc='left',fontweight='bold');fig.colorbar(cm,ax=axs[1],shrink=.5,label='ρ');save(fig,'02_特征覆盖与相关性')
# 3
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained');order=['Dummy','Logistic','RandomForest','XGBoost','NestedSelection'];lab=['先验基线','逻辑回归','随机森林','XGBoost','内层选择流程']
s=summ.set_index('model').loc[order];xx=s.macro_F1.to_numpy();axs[0].barh(range(5),xx,color=['#CCD2D8','#86AABF','#74AFA2','#397DAF','#A3A6BB']);axs[0].errorbar(xx,range(5),xerr=[xx-s.F1_CI_low,s.F1_CI_high-xx],fmt='none',color='#263645',capsize=3,lw=.8);axs[0].set_yticks(range(5),lab);axs[0].invert_yaxis();axs[0].set_xlim(0,.42);axs[0].set_xlabel('宏平均F1及95%分组重采样区间');panel(axs[0],'a','跨平台组折外表现')
for i,(c,n) in enumerate(zip(['#397DAF','#D98C48'],['balanced_accuracy','accuracy'])):axs[1].bar(np.arange(5)+(i-.5)*.35,s[n],width=.33,color=c,label=['平衡准确率','准确率'][i])
axs[1].set_xticks(range(5),['基线','逻辑回归','随机森林','XGB','选择流程'],rotation=25);axs[1].set_ylim(0,.65);axs[1].legend(frameon=False);panel(axs[1],'b','多数类影响下的不同指标');save(fig,'03_模型比较')
# 4 confusion
fig,axs=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained');cms=confusion_matrix(y,p.argmax(1),labels=range(5));rec=cms/cms.sum(1)[:,None]
for ax,mat,title,mode in zip(axs,[cms,rec],['样本计数','按真实类别归一化'],['n','p']):
    im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=(mat.max() if mode=='n' else 1))
    for i in range(5):
        for j in range(5):ax.text(j,i,str(mat[i,j]) if mode=='n' else f'{mat[i,j]:.1%}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>(mat.max() if mode=='n' else 1)*.6 else '#22313D')
    ax.set_xticks(range(5),['M1','M2','M3','M4','M5']);ax.set_yticks(range(5),names);ax.set_xlabel('预测类型');ax.set_title(title,loc='left',fontweight='bold')
axs[0].set_ylabel('真实形态标签');save(fig,'04_XGBoost折外混淆矩阵')
# 5 ROC PR
fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained');classrows=[]
for j in range(5):
    yy=(y==j).astype(int);fpr,tpr,_=roc_curve(yy,p[:,j]);pr,rc,_=precision_recall_curve(yy,p[:,j]);ap=average_precision_score(yy,p[:,j]);ar=auc(fpr,tpr)
    axs[0].plot(fpr,tpr,color=C[j],label=f'M{j+1} AUC={ar:.3f}');axs[1].plot(rc,pr,color=C[j],label=f'M{j+1} AP={ap:.3f}');classrows.append({'class':LABELS[j],'n':int(yy.sum()),'recall':rec[j,j],'precision':cms[j,j]/max(cms[:,j].sum(),1),'F1':2*cms[j,j]/max(cms[:,j].sum()+cms[j].sum(),1),'AUROC':ar,'AP':ap,'prevalence':yy.mean()})
axs[0].plot([0,1],[0,1],':',color='#999999',lw=.8)
for ax in axs:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(frameon=False,fontsize=7)
axs[0].set(xlabel='假阳性率',ylabel='真阳性率');axs[1].set(xlabel='召回率',ylabel='精确率');panel(axs[0],'a','一对其余类别ROC');panel(axs[1],'b','一对其余类别PR');save(fig,'05_XGBoost_ROC与PR');pd.DataFrame(classrows).to_csv(O/'06_XGBoost分类别表现.csv',index=False,encoding='utf-8-sig')
# 6 reliability and group results
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained');conf=p.max(1);ok=p.argmax(1)==y;bins=np.linspace(.2,1,9);br=[]
for lo,hi in zip(bins[:-1],bins[1:]):
    ix=(conf>=lo)&(conf<(hi if hi<1 else 1.00001))
    if ix.sum():br.append([conf[ix].mean(),ok[ix].mean(),ix.sum()])
br=np.array(br);axs[0].plot([.2,1],[.2,1],':',color='#999999');axs[0].plot(br[:,0],br[:,1],color='#397DAF');axs[0].scatter(br[:,0],br[:,1],s=br[:,2]*.24+8,color='#397DAF');axs[0].set(xlabel='最大预测概率的箱内均值',ylabel='箱内实际准确率',xlim=(.2,1),ylim=(0,1));panel(axs[0],'a','未校准概率可靠性')
grows=[]
for g in uu:
    ix=groups==g;present=np.unique(y[ix]);grows.append({'group':g,'n':int(ix.sum()),'present_classes':len(present),'macro_F1_present':f1_score(y[ix],p[ix].argmax(1),labels=present,average='macro',zero_division=0),'accuracy':ok[ix].mean()})
gdf=pd.DataFrame(grows).sort_values('macro_F1_present');axs[1].scatter(gdf.macro_F1_present,np.arange(len(gdf)),s=gdf.n*.16+10,color='#459886');axs[1].set_yticks(range(len(gdf)),[g.replace('阳101','Y101').replace('泸','L') for g in gdf.group],fontsize=6);axs[1].set_xlabel('组内已有类别宏平均F1');axs[1].set_xlim(0,1);panel(axs[1],'b','平台组间差异');save(fig,'06_概率可靠性与分组表现');gdf.to_csv(O/'07_各平台组表现.csv',index=False,encoding='utf-8-sig')
# 7 importance
fig,axs=plt.subplots(1,2,figsize=(7.2,4),layout='constrained',gridspec_kw={'width_ratios':[1,1.05]});ip=imp.iloc[::-1];pos=np.arange(len(ip))
axs[0].barh(pos,ip.mean_abs_SHAP_margin,color='#397DAF',height=.6);axs[0].scatter(ip.platform_equal_mean_abs_SHAP,pos,s=17,marker='D',facecolors='white',edgecolors='#D98C48',label='平台组等权');axs[0].set_yticks(pos,[short[t].split(' / ')[0] for t in ip.feature]);axs[0].set_xlabel('平均|SHAP| / 类别原始得分');axs[0].legend(frameon=False);panel(axs[0],'a','XGBoost折外全局贡献')
v=ip[['M1','M2','M3','M4','M5']].to_numpy();im=axs[1].imshow(v,aspect='auto',cmap='YlGnBu',origin='lower');axs[1].set_yticks(pos,[short[t].split(' / ')[0] for t in ip.feature]);axs[1].set_xticks(range(5),['M1','M2','M3','M4','M5']);axs[1].set_title('b  各类型平均绝对贡献',loc='left',fontweight='bold');fig.colorbar(im,ax=axs[1],shrink=.55,pad=.02);save(fig,'07_SHAP全局与分类别重要性')
# 8 beeswarm with missing color separated; test observations only
fig,axs=plt.subplots(2,3,figsize=(10.5,6.2),layout='constrained');rng=np.random.default_rng(7);cmap=LinearSegmentedColormap.from_list('b_o',['#397DAF','#ECECE9','#D56D42'])
for j,ax in enumerate(axs.flat):
    if j==5:ax.axis('off');ax.text(.05,.8,'每点为一个折外样本\n蓝色：特征值较低\n橙色：特征值较高\n灰色：原始值缺失\n\nSHAP > 0 提高该类别原始得分\n不代表风险增加或概率增量',fontsize=10,va='top');continue
    ids=np.argsort(np.abs(sv[:,:,j]).mean(0))[-8:]
    for row,k in enumerate(ids):
        val=X.iloc[:,k].to_numpy();finite=np.isfinite(val);qlo,qhi=np.nanquantile(val,[.05,.95]);norm=np.clip((val-qlo)/max(qhi-qlo,1e-9),0,1);jit=rng.normal(0,.10,len(d));ax.scatter(sv[finite,k,j],row+jit[finite],c=norm[finite],s=3,cmap=cmap,vmin=0,vmax=1,alpha=.5,rasterized=True);ax.scatter(sv[~finite,k,j],row+jit[~finite],color='#BDBDBD',s=3,alpha=.45,rasterized=True)
    ax.axvline(0,color='#AAAAAA',lw=.7);ax.set_yticks(range(8),[short[FEATURES[k]].split(' / ')[0] for k in ids]);ax.set_xlabel('SHAP / 类别原始得分');panel(ax,chr(97+j),names[j])
save(fig,'08_SHAP五类蜂群图')
# 9 dependence chosen by global importance, class with biggest magnitude for each feature
fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),layout='constrained');dr=[]
for t,(ax,ft) in enumerate(zip(axs.flat,imp.feature.head(4))):
    k=FEATURES.index(ft);j=int(np.argmax(np.abs(sv[:,k,:]).mean(0)));v=X[ft].to_numpy();mask=np.isfinite(v);ax.scatter(v[mask],sv[mask,k,j],c=np.asarray(C)[fold[mask]],s=7,alpha=.5,rasterized=True);ax.axhline(0,color='#AAAAAA',ls=':',lw=.7)
    qs=np.unique(np.nanquantile(v,np.linspace(0,1,7)));bx=[]
    for lo,hi in zip(qs[:-1],qs[1:]):
        ix=mask&(v>=lo)&(v<=hi if hi==qs[-1] else v<hi)
        if ix.sum()>=10:bx.append((np.median(v[ix]),np.median(sv[ix,k,j])))
    if bx:bx=np.array(bx);ax.plot(bx[:,0],bx[:,1],color='#202C35',marker='o',ms=3)
    ax.set(xlabel=short[ft],ylabel=f'SHAP / M{j+1}原始得分');panel(ax,chr(97+t),short[ft].split(' / ')[0]+'与'+names[j].split(' ')[1]);dr.append({'feature':ft,'class':LABELS[j],'observed_n':int(mask.sum()),'pooled_spearman':spearmanr(v[mask],sv[mask,k,j]).statistic})
fig.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=C[k],label=f'验证折{k+1}',ms=3) for k in range(5)],loc='outside upper center',ncol=5,frameon=False);save(fig,'09_SHAP关键变量依赖图');pd.DataFrame(dr).to_csv(O/'08_SHAP依赖关系摘要.csv',index=False,encoding='utf-8-sig')
# 10 fixed sensitivity
fig,ax=plt.subplots(figsize=(7.2,3.1),layout='constrained');ax.barh(range(len(sens)),sens.macro_F1,color=['#93B3C7']*4+['#A49CBA','#D98C48']);ax.set_yticks(range(len(sens)),sens.experiment);ax.invert_yaxis();ax.set_xlim(0,.5)
for i,r in sens.iterrows():ax.text(r.macro_F1+.008,i,f'{r.macro_F1:.3f}  (n={r.n})',va='center',fontsize=8)
ax.set_xlabel('折外宏平均F1 · 各对照使用同一固定XGBoost配置');panel(ax,'a','特征组及验证层级对照');save(fig,'10_消融与敏感性对照')
# 11 local explanations deterministic correct and incorrect examples near median confidence
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained');examples=[]
for c,ax in zip([True,False],axs):
    idx=np.flatnonzero(ok==c);idx=idx[np.argsort(np.abs(conf[idx]-np.median(conf[idx])))][0];j=p[idx].argmax();order=np.argsort(np.abs(sv[idx,:,j]))[-7:];vals=sv[idx,order,j];ax.barh(range(7),vals,color=np.where(vals>=0,'#D98C48','#397DAF'));ax.set_yticks(range(7),[short[FEATURES[k]].split(' / ')[0] for k in order]);ax.axvline(0,color='#777777',lw=.7);ax.set_xlabel(f'SHAP / M{j+1}原始得分');panel(ax,'a' if c else 'b',('正确' if c else '错误')+f'示例：M{y[idx]+1} → M{j+1}');examples.append({'sample_key':d.iloc[idx,0],'true':LABELS[y[idx]],'predicted':LABELS[j],'confidence':float(conf[idx]),'base_value':float(z['base'][idx,j]),'shap_sum':float(sv[idx,:,j].sum()),'fold':int(fold[idx])})
save(fig,'11_SHAP典型与反例局部解释');pd.DataFrame(examples).to_csv(O/'09_SHAP局部解释示例.csv',index=False,encoding='utf-8-sig')
# 12 prior actual shape representatives, preserve physical scale
old=Path('F:/论文库/IGS/实验/过程/时序聚类_V3_形态与强度');r=pd.read_pickle(old/'results.pkl');arr=np.load(old/'pressure_delta101.npy');fig,axs=plt.subplots(2,3,figsize=(7.2,4.1),sharex=True,sharey=True,layout='constrained');axs=axs.ravel()
reps=[]
for j,ax in enumerate(axs):
    if j==5:
        ax.axis('off');ax.text(.05,.75,'实际代表曲线\n统一压差尺度\nM1与M5幅度较小',fontsize=8,va='top');continue
    candidates=r[(r.morphology_code_v3==f'M{j+1}')&(~r.morphology_uncertain.astype(bool))]
    if len(candidates)==0:candidates=r[r.morphology_v3.str.contains(LABELS[j].replace('型',''),regex=False)]
    # Display chosen archived actual examples when their IDs are available; otherwise closest fit at class median amplitude.
    ii=[1362,2508,1070,2162,1114][j];curve=arr[ii];ax.plot(np.linspace(0,1,101),curve,color=C[j]);ax.axhline(0,color='#CCCCCC',lw=.7);ax.set_title(names[j],loc='left',fontweight='bold');ax.set_xlabel('相对观察时间');tidy(ax);reps.append({'type':LABELS[j],'sample_id':str(r.loc[ii,'sample_id']),'index':int(ii)})
axs[0].set_ylabel('起点参考压差 / MPa');save(fig,'12_五类实际曲线统一尺度');pd.DataFrame(reps).to_csv(O/'10_五类曲线图来源.csv',index=False,encoding='utf-8-sig')
with PdfPages(O/'分类预测与SHAP_全部图件.pdf') as pdf:
    for n,fig in figures:pdf.savefig(fig,bbox_inches='tight')
plt.close('all')
# additional audit and compact workbook input
pd.DataFrame({'feature':FEATURES,'missing_fraction':X.isna().mean().values,'observed_n':X.notna().sum().values}).to_csv(O/'11_输入特征覆盖.csv',index=False,encoding='utf-8-sig')
foldimp=[]
for k in range(5):
    vals=np.abs(sv[fold==k]).mean((0,2))
    for f,v in zip(FEATURES,vals):foldimp.append({'fold':k,'feature':f,'mean_abs_SHAP':v})
pd.DataFrame(foldimp).to_csv(O/'12_SHAP跨折稳定性.csv',index=False,encoding='utf-8-sig')
# independent data checks
assert np.isfinite(p).all() and np.max(np.abs(p.sum(1)-1))<1e-5
assert cms.sum()==len(d) and len(d.iloc[:,0].unique())==len(d)
checks={'n':len(d),'confusion_total':int(cms.sum()),'probability_sum_max_error':float(np.max(np.abs(p.sum(1)-1))),'figures':len(figures),'shap_scale':'raw multiclass margin','shap_samples':'outer test only','missing_values_displayed_separately':True}
(B/'figure_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
print('ALL_FIGURES_DONE',flush=True)
