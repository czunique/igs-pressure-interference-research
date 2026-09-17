"""Publication figure exports and a concise, source-traceable Chinese results report."""
import sys,json,platform,importlib.metadata,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path('F:/论文库/IGS/实验/过程/时序聚类_20260915');OUT=Path('F:/论文库/IGS/实验/结论/时序聚类_20260915')
FIG=OUT/'图件';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':300,'pdf.fonttype':42,'svg.fonttype':'none','axes.labelcolor':'#25313c','text.color':'#25313c','axes.titleweight':'bold'})
COLORS=['#35658D','#C18A36','#A95237','#63774B','#975B83','#5F8290']
METHODS={'DTW_kmedoids':'受限DTW＋k-medoids','Euclidean_kmeans':'欧氏距离k-means','Feature_Ward':'物理特征＋Ward'}
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(FIG/(name+'.'+ext),bbox_inches='tight',facecolor='white')
 plt.close(fig)
def mdtable(df):
 def f(x):
  if isinstance(x,(float,np.floating)):return '—' if not np.isfinite(x) else f'{x:.3f}'
  return str(x).replace('|',' / ')
 return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(f(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))
def main():
 cfg=json.loads((BASE/'run_config.json').read_text(encoding='utf-8'));df=pd.read_pickle(BASE/'clustered_samples.pkl');full=pd.read_pickle(BASE/'samples.pkl');X=np.load(BASE/'primary_curves_201.npy');labels=np.load(BASE/'selected_labels.npy');meds=np.load(BASE/'prototype_indices.npy');raw=pd.read_pickle(BASE/'event_raw_curves.pkl')
 inv=pd.read_csv(BASE/'pressure_inventory.csv');unmatched=pd.read_csv(OUT/'02_未匹配及未解析记录.csv');comparison=pd.read_csv(OUT/'03_算法与类别数比较.csv');stab=pd.read_csv(OUT/'05_分组重采样稳定性.csv');summary=pd.read_csv(OUT/'07_类别响应特征汇总.csv');sens=pd.read_csv(OUT/'08_时间与幅度敏感性.csv');logo=pd.read_csv(OUT/'09_留一平台稳定性.csv')
 k=cfg['k'];u=np.linspace(0,100,201)
 coverage=inv.groupby('platform').agg(监测文件数=('path','size'),解析通道数=('channel_count','sum'));coverage['主分析独立样本数']=df.groupby('platform_id').size();coverage['补充样本数']=full[full.status=='supplementary'].groupby('platform_id').size();coverage=coverage.fillna(0).astype(int).sort_values('主分析独立样本数',ascending=False)
 coverage.to_csv(OUT/'11_平台覆盖汇总.csv',encoding='utf-8-sig')
 fig,ax=plt.subplots(figsize=(10,7));sub=coverage.iloc[::-1];ax.barh(sub.index,sub['主分析独立样本数'],color=COLORS[0],label='主分析');ax.barh(sub.index,sub['补充样本数'],left=sub['主分析独立样本数'],color='#D9DFE5',label='补充');ax.set_xlabel('去重后的施工段—邻井样本数');ax.set_title('各平台进入主分析与补充分析的样本');ax.legend(frameon=False,loc='lower right');ax.grid(axis='x',alpha=.15);save(fig,'图01_平台样本覆盖')
 fig,axes=plt.subplots(1,2,figsize=(11,4.4))
 for i,(m,z) in enumerate(comparison.groupby('method',sort=False)):
  axes[0].plot(z.k,z.common_dtw_silhouette,'o-',color=COLORS[i],label=METHODS[m]);axes[1].plot(z.k,z.ari_median,'o-',color=COLORS[i],label=METHODS[m]);axes[1].fill_between(z.k,z.ari_p10,z.ari_median,color=COLORS[i],alpha=.1)
 for ax in axes:ax.set_xticks(range(2,7));ax.set_xlabel('类别数 K');ax.grid(alpha=.18)
 axes[0].set_ylabel('共同DTW距离下的轮廓系数');axes[1].set_ylabel('成组重采样 ARI');axes[0].set_title('相同样本、相同评价距离');axes[1].set_title('100次成组抽样：中位数与P10');axes[0].legend(frameon=False,fontsize=8);axes[1].set_ylim(-.05,1.05);save(fig,'图02_算法与类别数比较')
 fig,axes=plt.subplots(k,1,figsize=(10,2.7*k),squeeze=False,sharex=True)
 for j in range(k):
  ax=axes[j,0];y=X[labels==j];q=np.quantile(y,[.1,.5,.9],axis=0)
  for line in y:ax.plot(u,line,color=COLORS[j],alpha=min(.15,15/len(y)),lw=.45)
  ax.fill_between(u,q[0],q[2],color=COLORS[j],alpha=.2,label='P10–P90样本分位带');ax.plot(u,q[1],color=COLORS[j],lw=2,label='逐时刻中位数');ax.plot(u,X[meds[j]],color='#1F2630',lw=1.4,linestyle='--',label='真实代表曲线');ax.axhline(0,color='#999999',lw=.6);ax.set_ylabel('Δp / MPa');ax.set_title(f'C{j+1}  |  n={len(y)}，{summary.iloc[j].platforms}个平台',loc='left');ax.grid(alpha=.12)
 axes[0,0].legend(frameon=False,ncol=3,fontsize=8);axes[-1,0].set_xlabel('完整施工事件的归一化时间 / %');fig.suptitle('压力幅度保留的时序聚类（各面板纵轴独立）',y=1.005,fontsize=14);fig.tight_layout();save(fig,'图03_各类曲线叠加与代表样本')
 fig,axes=plt.subplots(1,3,figsize=(12,4.3));ticks=['C'+str(j+1) for j in range(k)]
 for ax,f,title,unit in [(axes[0],'a60_mpa','持续响应幅度','A60 / MPa'),(axes[1],'onset_min','首次持续起涨时滞','min'),(axes[2],'max_rate60_mpa_min','最大一分钟涨速','MPa/min')]:
  vals=[df.loc[labels==j,f].dropna().to_numpy() for j in range(k)]
  b=ax.boxplot(vals,tick_labels=ticks,patch_artist=True,showfliers=True,flierprops={'marker':'.','markersize':3,'alpha':.3})
  for patch,col in zip(b['boxes'],COLORS):patch.set_facecolor(col);patch.set_alpha(.5)
  ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.15)
 fig.text(.5,-.025,'箱体为四分位范围；时滞图仅含检出起涨的样本，未检出比例见结果表。',ha='center',fontsize=9);fig.tight_layout();save(fig,'图04_原始尺度响应特征')
 fig,ax=plt.subplots(figsize=(7,4));bottom=np.zeros(k)
 for column,col,title in [('grade_lt1_fraction','#CEDAE6','A60 < 1 MPa'),('grade_1to5_fraction','#C18A36','1 ≤ A60 < 5 MPa'),('grade_ge5_fraction','#A95237','A60 ≥ 5 MPa')]:
  heights=summary[column].to_numpy()*100;ax.bar(ticks,heights,bottom=bottom,color=col,label=title)
  for i,v in enumerate(heights):
   if v>=8:ax.text(i,bottom[i]+v/2,f'{v:.0f}%',ha='center',va='center',color='white' if col!='#CEDAE6' else '#25313c')
  bottom+=heights
 ax.set_ylim(0,100);ax.set_ylabel('类内样本比例 / %');ax.set_title('各类压力响应幅度组成');ax.legend(frameon=False,ncol=1,bbox_to_anchor=(1.02,1));fig.text(.02,-.03,'1、5 MPa为试验方案的描述阈值；尚未用微地震或现场处置结果验证风险。',fontsize=9);save(fig,'图05_类别内响应幅度组成')
 cross=pd.crosstab(df.platform_id,df.cluster);fig,ax=plt.subplots(figsize=(max(6,k*1.1),max(4,.3*len(cross)+1)));im=ax.imshow(cross.values,cmap='Blues',aspect='auto');ax.set_xticks(range(k),cross.columns);ax.set_yticks(range(len(cross)),cross.index)
 for i in range(len(cross)):
  for j in range(k):ax.text(j,i,str(cross.iloc[i,j]),ha='center',va='center',fontsize=9,color='white' if cross.iloc[i,j]>cross.values.max()*.55 else '#25313c')
 ax.set_title('类别的平台分布（计数）');fig.colorbar(im,ax=ax,label='独立样本数',shrink=.7);save(fig,'图06_类别跨平台支持')
 fig,axes=plt.subplots(k,1,figsize=(10,3*k),squeeze=False)
 for j in range(k):
  row=df.iloc[meds[j]];r=raw[int(row.matrix_row)];ax=axes[j,0]
  ax.plot(r['pre_t_min'],r['pre_p_mpa'],color='#969DA5',lw=.7,label='起泵前压力');ax.plot(r['t_min'],r['p_mpa'],color=COLORS[0],lw=.8,label='施工期间原始压力');ax.axhline(row.baseline_mpa,color='#444444',linestyle='--',lw=.8,label='基线');ax.axvline(0,color='#777777',linestyle=':',lw=.8);ax.set_ylabel('邻井压力 / MPa');ax.set_xlabel('相对起泵时间 / min');ax.set_title(f'C{j+1}：{row.platform_id}，{row.source_well}井第{row.stage_id}段 → {row.monitor_well}井',loc='left')
  a2=ax.twinx();a2.plot(r['pump_t_min'],r['pump_q'],color='#9B7535',alpha=.7,lw=.85,linestyle='--',label='施工排量（右轴）');a2.set_ylabel('排量 / (m³/min)');ax.set_xlim(min(-10,min(r['pre_t_min'],default=0)),row.duration_min+2)
 from matplotlib.lines import Line2D
 handles,leglabels=axes[0,0].get_legend_handles_labels();handles.append(Line2D([0],[0],color='#9B7535',ls='--',lw=.85));leglabels.append('施工排量（右轴）');axes[0,0].legend(handles,leglabels,frameon=False,ncol=2,fontsize=8);fig.tight_layout();save(fig,'图07_代表样本原始压力与排量')
 fig,ax=plt.subplots(figsize=(8,4));z=stab[(stab.method==cfg['selected_method'])&(stab.k==k)];ax.hist(z.ari_to_full,bins=np.linspace(-.1,1,23),color=COLORS[0],edgecolor='white');ax.set_xlabel('成组重采样结果相对全样本分组的 ARI');ax.set_ylabel('重复次数');ax.set_title(f'选定方案稳定性：100次施工段成组抽样');ax.axvline(z.ari_to_full.median(),color='#A95237',ls='--');save(fig,'图08_选定方案稳定性分布')
 simple=summary[['cluster','n','platforms','pairs','stages','a60_mpa_median','onset_min_median','grade_ge5_fraction']].copy();simple.columns=['类别','样本数','平台数','井对数','施工段数','A60中位数(MPa)','已检出时滞中位数(min)','A60≥5MPa比例(0–1)']
 tops=comparison.sort_values(['selected','common_dtw_silhouette'],ascending=[False,False])[['method','k','common_dtw_silhouette','ari_median','ari_p10','support_gate','selected']].copy();tops['method']=tops.method.map(METHODS);tops.columns=['方法','K','共同轮廓系数','ARI中位数','ARI-P10','支持度通过','选定']
 counts=full.status.value_counts().to_dict();un=unmatched.reason.value_counts().rename_axis('原因').reset_index(name='记录数')
 report=f'''# 邻井压力时序聚类：首轮全数据探索结果

## 一、结论

本轮已完成对 **{len(inv):,} 份监测文件、{inv.platform.nunique()} 个目录平台**的读取检查，以及施工事件匹配、去重、质量控制、多方法比较和稳定性检验。

严格质量口径下，纳入 **{cfg['n']} 条独立施工段—邻井响应，覆盖 {cfg['platforms']} 个平台、{cfg['pairs']} 个物理井对、{cfg['stages']} 个施工段组**。在本轮比较的算法及参数范围内，选定 **{METHODS[cfg['selected_method']]}，K={k}**。共同DTW轮廓系数为 **{cfg['silhouette']:.3f}**，100次成组重采样ARI中位数为 **{cfg['ari_median']:.3f}**，P10为 **{cfg['ari_p10']:.3f}**。

本轮类别是压力曲线的统计分组，证据级别为“仅压力观测”。当前未进行微地震验证，不能将类别直接命名为已验证的流体连通机制、恶性压窜类型或安全等级。

## 二、数据范围及纳入情况

源数据：`E:/压裂窜扰项目/02 规范化`，递归检查各平台邻井压力目录，并包含“不重要的”子目录。示例模板目录“xxx平台”不计入研究数据。原始文件保持只读。

- 解析得到的压力通道记录：{int(inv.channel_count.sum()):,}。这一数量含重复导出、多工作表和低频记录，不能用作独立样本量。
- 未提取出可识别时序通道的文件：{int((inv.channel_count==0).sum())}，逐文件说明见过程目录的压力文件索引。
- 匹配施工后进入主分析的独立样本：{counts.get('primary',0)}。
- 仅作补充复核的独立样本：{counts.get('supplementary',0)}。
- 基线非平稳或近零压力跳变、需核实工况/仪表的独立样本：{counts.get('operational_review',0)}，单列保存，不进入主聚类也不标为低风险。
- 匹配后因质量不足排除的独立样本：{counts.get('exclude',0)}。
- 匹配候选中重复导出或重复工作表：{counts.get('duplicate_export',0)}。

上述文件、通道和独立样本是不同统计单位，不应直接相加。未匹配记录可能对应一个文件、一个压力通道或一个候选事件，明细中保留其粒度。

{mdtable(un)}

平台覆盖明细见 `11_平台覆盖汇总.csv`；全部候选曲线质量见 `01_全部候选曲线与质控.csv`。

## 三、处理与比较方法

1. 以“平台—施工井—施工段—监测邻井”为样本键；同段多个邻井共同归入施工段组。中间停泵保留在整段窗口内，本轮不将停泵后复施工拆成新的独立样本。
2. 按有效排量自动识别完整施工窗口：排量阈值为 max(0.5 m³/min, 该记录P95排量的5%)，有效段需持续至少60 s。使用首个与末个有效泵注区间界定窗口，优先选择采样更密的有效施工表。
3. 双方有绝对日期时按绝对时间匹配。对有绝对日期支持的监测邻井同期施工和多源施工重叠单独排除。只有时分秒、无法核实跨井同期作业的记录保留这一限制。同一已知井段只有时分秒时，仅按同名井段和时钟对齐，允许整日平移处理跨日，不移动起涨点、不做最大相关平移。主分析的对齐方式构成为：{df.alignment.value_counts().to_dict()}。时钟对齐样本仍需施工日志核实日期。
4. 起泵前10 min内至少5个有效点、覆盖180 s，取中位数作为基线。施工前基线不足时只生成“初始窗代理基线”的补充记录，不进入主分析。
5. 主分析要求施工窗首尾覆盖率≥95%、最大观测缺口≤30 s、原始采样间隔≤30 s；全零占位压力不纳入。起泵前P90-P10>1 MPa、末60 s中位压力与基线相差>0.5 MPa、或基线斜率绝对值>0.1 MPa/min者，列为基线非平稳待复核；原始压力出现<1 MPa且在≤5 s采样中有>5 MPa跳变者列为仪表/工况待复核。这些是本轮质量审查阈值，并不证明该响应虚假，也不能据此判为低风险。人工低频记录不会通过插值变成高频观测。
6. 原始尺度保留MPa和min。施工内使用约15 s中心中位滤波，随后在共同施工窗口内重采样为201点。保留压力涨幅而不逐条标准化，起涨时滞按 max(0.1 MPa,3倍基线稳健波动)持续60 s定义；未检出时保留删失标记。
7. 比较受限DTW＋k-medoids、欧氏k-means、物理特征＋Ward，每种K=2～6。DTW半径10点，距离为累计平方误差平方根再除以√201。Ward使用A60、终值、下降幅度、平均面积、一分钟最大涨速、相对时滞和删失标记的稳健缩放特征。
8. 所有方案用同一受限DTW距离矩阵计算共同轮廓系数。主要随机算法20次初始化；100次稳定性重拟合每次5次初始化。按平台分层，每次抽取约80%的施工段组，同段邻井不拆散；报告对全样本聚类的ARI和相邻重复共同样本上的ARI。
9. 先检查每类至少20个施工段、2个井对和2个平台的支持度，以及ARI-P10≥0.7；在通过的候选中选共同轮廓系数较高的方案，差值≤0.02时优先较少类别。若没有通过方案，则结果仅为探索性分型，不强行赋予成熟类型含义。

**当前支持度门槛通过：{cfg['support_gate_passed']}；稳定性门槛通过：{cfg['stability_gate_passed']}。**

### 相对试验方案V2的本轮差异

本轮按本次“对现有全部曲线聚类”的任务开展全数据探索，尚未按D0/D1/D2封存外部平台。Soft-DTW k-means本轮未运行，以欧氏k-means作为时间不变形基线；因此不能声称已证明优于Soft-DTW。100次稳定性重拟合采用5次随机初始化，主比较为20次。采样间隔、基线阈值、起涨阈值等尚未全部穷尽敏感性组合。本轮可用于建立候选类型，论文中的最终独立测试仍需重新设计冻结发现集或使用后续新增平台。

## 四、算法比较结果

以下列出全部15个候选，选定方案置于首行；完整结果见 `03_算法与类别数比较.csv`。较高轮廓系数同时受类数和类间幅度分离影响，需结合支持度和稳定性解释。

{mdtable(tops)}

![算法比较](图件/图02_算法与类别数比较.png)

### 为什么没有只选轮廓系数最高的方案

物理特征＋Ward的K=2轮廓系数较高，但把仅5条曲线单独分成一类，未达到20个施工段的支持门槛。欧氏k-means的K=2在部分成组抽样中分组变化较大，ARI-P10约0.375。受限DTW的K=2兼顾了类别支持、稳定性与分离程度。DTW的K=3也通过门槛，但共同轮廓系数约0.672、稳定性低于K=2，因此保留为细分备选而不强行增加类别。

## 五、类别的实际响应特征

类别编号按A60中位数升序排列，编号顺序不等同于经验证的风险顺序。A60指完整60 s窗口内压力增量中位数的最大值。下表时滞只统计检出持续起涨的曲线；未检出数量和四分位范围保存在类别汇总表中。

{mdtable(simple)}

![曲线分型](图件/图03_各类曲线叠加与代表样本.png)

**暂定的描述名称**：C1为“微弱响应为主型”，C2为“持续增压为主型”。这两个名称描述当前类内多数曲线，不保证覆盖所有边界曲线。C1仍含少量晚期较大涨幅样本，所以C1不能整体等同于无压窜或低风险。C2内部也有不同上升形态，当前不足以分别定为不同连通机制。

逐条z-score后与主分型的ARI约0.037，说明“只比较形状”与“保留幅度”的分组差异很大。现有结果最稳健地支持强弱响应的粗分，不足以声称已发现多种独立、稳定的机制形态。泸201H5贡献242/505条主分析曲线，跨平台样本并不均衡；移除该平台后ARI约0.882，须在论文中同时披露。

各类代表样本来自真实观测，文件、井段、邻井和时间信息见 `07_类别响应特征汇总.csv` 及 `06_主分析样本聚类标签.csv`。1和5 MPa仅用于报告试验方案中的幅度组成，当前不构成通用压窜判据。

## 六、稳健性和边界样本

{mdtable(sens)}

敏感性表固定选定类别数，并以DTW＋k-medoids检查时间约束、序列长度、逐条标准化及是否平滑的影响。其中逐条标准化回答“仅按形状分组是否一致”，与保留幅度的研究问题不同。

留一平台后，在其余样本上重新拟合的ARI范围为 **{logo.ari_retained.min():.3f}～{logo.ari_retained.max():.3f}**。这是类型发现对平台构成的敏感性检查，不是监督分类模型的外部预测性能。

按前两类原型距离差占第二近距离的比例<0.1标记边界样本，比例为 **{cfg['boundary_fraction']:.1%}**。最近真实原型重分配与原聚类标签的一致率为 **{cfg['prototype_reassignment_agreement']:.1%}**；若采用Ward或k-means，后续冻结字典时必须保留这一赋值差异。补充曲线的临时归类只作复核，不能直接混入正式训练标签。

## 七、论文中可以写到什么程度

可以报告本轮数据覆盖、明确的纳入排除流程、三方法的共同距离评价、K选择依据、真实代表曲线及稳定性。样本来自已有监测资料，可能偏向曾被关注或已出现异常的井段，类别比例不代表全区块压窜发生率。

当前不能写为已验证的机制分型、地质因素因果效应或施工前预测能力。微地震、天然裂缝参数提取、分类预测与SHAP均未在本轮执行。任何仅由压力指标定义的等级与压力指标的差异，属于描述而非独立验证。

下一步应先结合真实代表曲线和边界样本核实井号、基线与施工工况，确定候选类型名称，再按用户计划进入微地震处理与独立验证。

## 八、交付文件与复算

结论目录包含本报告、CSV结果表，以及300 dpi PNG、矢量PDF和SVG图件。代码、源文件哈希索引、原始尺度片段、201点矩阵、距离矩阵、抽样分组、配置和环境版本全部保存在同名“过程”目录。

源文件未被写入、移动或删除。结果标签文件带有原始文件路径、压力工作表、施工工作表和样本键，可逐条追溯。

### 方法资料

- [DTW时序聚类原理与实现说明](https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html)
- [scikit-learn 轮廓系数定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [scikit-learn ARI定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
'''
 (OUT/'时序聚类_首轮结果报告.md').write_text(report,encoding='utf-8')
 versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','dtaidistance','python-calamine']};versions['python']=sys.version;(BASE/'environment_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
 checks={'primary_unique_sample_id':df.sample_id.is_unique,'matrix_finite':bool(np.isfinite(X).all()),'labels_cover_all_primary':len(df)==len(labels),'cluster_counts_equal_primary':int(summary.n.sum())==len(df),'primary_coverage_ge95':bool((df.coverage>=.95).all()),'primary_gap_le30s':bool((df.max_gap_s<=30.01).all()),'primary_prepump_baseline':bool((df.baseline_source=='pre_pump_10min').all()),'source_file_sizes_mtimes_unchanged':True}
 for _,r in inv.iterrows():
  p=Path(r.path)
  if p.stat().st_size!=r['size'] or p.stat().st_mtime_ns!=int(r.mtime):checks['source_file_sizes_mtimes_unchanged']=False
 (BASE/'validation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print('CHECKS',checks);print('REPORT',OUT/'时序聚类_首轮结果报告.md')
if __name__=='__main__':main()
