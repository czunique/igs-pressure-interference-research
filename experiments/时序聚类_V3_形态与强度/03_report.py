"""V3 report, real curve figures, feature dictionary and independently checkable delivery."""
import sys,json,re,hashlib,warnings,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore')
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V3_形态与强度');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V3_形态与强度');OLD=B.parent/'时序聚类_V2_20260915';G=O/'图';G.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('v3',B/'01_classify.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190,'pdf.fonttype':42,'svg.fonttype':'none'})
COL=['#8292a4','#d48544','#377abd','#56987c','#996db0','#b85b4d','#d6aa45','#648077','#939393'];DES={1:'整体变化较小；先按幅度识别',2:'早期上升较快，尾段速率明显降低',3:'中期上升最快；前、后期速率较低',4:'三个阶段的上升速率较接近',5:'明显下降后转为持续上升',6:'尾段持续加速，尚未形成明显平台',7:'压力升高后出现明显回落',8:'观察窗口内以压力下降为主',9:'单一模板解释不足或指标不满足'}
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(G/(name+'.'+ext),bbox_inches='tight')
 plt.close(fig)
def table(x):
 return '\n'.join(['| '+' | '.join(x.columns)+' |','|'+'|'.join(['---']*len(x.columns))+'|']+['| '+' | '.join(str(z).replace('|','/') for z in row)+' |' for row in x.itertuples(index=False,name=None)])
def main():
 d=pd.read_pickle(B/'results.pkl');X=np.load(B/'pressure_delta101.npy');Y=np.load(B/'shape101.npy');P=np.load(B/'fitted_shape101.npy');raw=pd.read_pickle(OLD/'raw_curves.pkl');proto=pd.read_csv(O/'08_典型曲线来源.csv');cfg=json.loads((B/'config.json').read_text(encoding='utf-8'));u=np.linspace(0,1,101);ct=pd.crosstab(d.morphology_code_v3,d.response_strength_grade).reindex(index=range(1,10),columns=['弱','中','强'],fill_value=0)
 summary=d.groupby(['morphology_code_v3','morphology_v3']).agg(n=('sample_id','size'),platforms=('platform_id','nunique'),uncertain=('morphology_uncertain','sum'),amplitude=('amplitude_mpa','median'),rise=('positive_rise_p95_mpa','median'),early=('early_rate_mpa_min_v3','median'),middle=('middle_rate_mpa_min_v3','median'),late=('late_rate_mpa_min_v3','median'),auc=('positive_auc_mpa_min','median'),rise_time=('rising_duration_min','median'),fit_error=('template_normalized_rmse','median'));summary.to_csv(O/'03_形态特征汇总.csv',encoding='utf-8-sig')
 def prototype_figure(codes,title,filename):
  fig=plt.figure(figsize=(15,2.55*len(codes)));gs=fig.add_gridspec(len(codes),3,width_ratios=[1.2,2.15,2.3],hspace=.8,wspace=.3)
  for row,code in enumerate(codes):
   z=d[d.morphology_code_v3==code];idx=int(proto[proto.code==code].iloc[0].model_row);r=d.iloc[idx];o=raw[int(r.matrix_row)];t=(o['t']-o['t'][0])/60;c=COL[code-1]
   ax=fig.add_subplot(gs[row,0]);ax.axis('off');ax.text(0,.85,f'M{code} {m.NAMES[code]}',fontsize=15,color=c,weight='bold');ax.text(0,.25,f'{len(z):,}条 | {z.platform_id.nunique()}个平台\n占全部 {len(z)/len(d):.1%}',fontsize=11,linespacing=1.8)
   ax=fig.add_subplot(gs[row,1]);ax.plot(t,o['p']-r.p0_window_mpa,color='#bcc3ca',lw=.65,label='原始观测');ax.plot(u*r.window_duration_min,X[idx],color=c,lw=2,label='趋势曲线');ax.axhline(0,color='#777',lw=.5);ax.grid(alpha=.17);ax.set(xlabel='观察起点后 / min',ylabel='ΔP / MPa');ax.set_title(f'{r.platform_id} / 源井{r.source_well or "?"} / 段{r.stage_id or "?"} / 邻井{r.monitor_well}',fontsize=9)
   if row==0:ax.legend(fontsize=7)
   ax=fig.add_subplot(gs[row,2]);ax.axis('off');med=z.median(numeric_only=True);txt=f"{DES[code]}\n\n正向涨幅中位数：{med.positive_rise_p95_mpa:.3f} MPa\n早/中/后期速率：{med.early_rate_mpa_min_v3:.3g} / {med.middle_rate_mpa_min_v3:.3g} / {med.late_rate_mpa_min_v3:.3g}\n速率单位：MPa/min\n正向积分中位数：{med.positive_auc_mpa_min:.1f} MPa·min\n弱 / 中 / 强响应：{ct.loc[code,'弱']} / {ct.loc[code,'中']} / {ct.loc[code,'强']}"
   ax.text(0,.95,txt,va='top',fontsize=10.5,linespacing=1.5)
  fig.suptitle(title,fontsize=19,y=.99);fig.text(.5,.018,'全部为实际曲线示例；形态采用参考模板与指标约束识别。强度是暂定压力响应分级，尚未验证为实际压窜风险。',ha='center',fontsize=10);fig.subplots_adjust(top=.93,bottom=.095);save(fig,filename)
 prototype_figure([1,2,3,4,5],'图示五种目标形态：实际数据识别结果','01_五种目标形态与实测曲线')
 prototype_figure([6,7,8,9],'实际数据中的补充形态与待判记录','02_补充形态与实测曲线')
 # Cross map proves shape is not risk and keeps population counts visible.
 fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(ct.to_numpy(),cmap='Blues',aspect='auto');ax.set(xticks=range(3),xticklabels=['弱响应','中响应','强响应'],yticks=range(9),yticklabels=[f'M{k} {m.NAMES[k]}' for k in range(1,10)]);fig.colorbar(im,ax=ax,label='曲线观测数')
 for i in range(9):
  for j in range(3):ax.text(j,i,str(ct.iloc[i,j]),ha='center',va='center',color='white' if ct.iloc[i,j]>550 else '#243345',fontsize=12)
 ax.set_title('同一形态可对应不同响应强度');fig.text(.5,.01,'M7–M9及质量/边界问题记录，风险标签保持待核查；本图仅统计压力响应强度。',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));save(fig,'03_形态乘响应强度')
 # 3 phase rates and acceleration shape: scale-independent class median trend.
 fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 for code in [2,3,4,5,6]:
  z=d[d.morphology_code_v3==code];med=z[['early_shape_rate','middle_shape_rate','late_shape_rate']].median().to_numpy();axs[0].plot([1,2,3],med,'o-',label=m.NAMES[code],color=COL[code-1]);axs[1].plot([1,2],np.diff(med),'o-',color=COL[code-1],label=m.NAMES[code])
 axs[0].set(xticks=[1,2,3],xticklabels=['前1/3','中1/3','后1/3'],ylabel='分阶段斜率 × 观察时长 / 幅度尺度',title='分阶段速率决定主要形态差异');axs[0].legend(fontsize=8);axs[1].set(xticks=[1,2],xticklabels=['前期→中期','中期→后期'],ylabel='阶段归一化斜率的变化',title='加速与减速趋势（描述性，不是SHAP）')
 for ax in axs:ax.axhline(0,lw=.7,color='#777');ax.grid(alpha=.2)
 fig.tight_layout();save(fig,'04_分阶段速率与加速趋势')
 # Response score distribution and physical evidence scatter.
 fig,axs=plt.subplots(1,2,figsize=(12,4.5));cmap={'弱':'#6c9e86','中':'#d8a754','强':'#bd665e'}
 for grade in ['弱','中','强']:
  z=d[d.response_strength_grade==grade];axs[0].hist(z.response_strength_score,bins=np.linspace(0,1.5,46),alpha=.75,color=cmap[grade],label=f'{grade}响应 n={len(z)}');axs[1].scatter(z.positive_rise_p95_mpa,z.positive_auc_mpa_min,s=8,alpha=.3,c=cmap[grade],label=grade)
 for cut in cfg['intensity_cuts']:axs[0].axvline(cut,color='#444',ls='--',lw=1);axs[0].text(cut,axs[0].get_ylim()[1]*.85,f'{cut:.3f}',rotation=90,ha='right')
 axs[0].set(xlabel='综合响应强度分数（当前样本尺度）',ylabel='曲线数',title='三组分数由一维KMeans分割');axs[0].legend(fontsize=8);axs[1].set_xscale('symlog',linthresh=.1);axs[1].set_yscale('symlog',linthresh=1);axs[1].set(xlabel='观察窗正向涨幅 / MPa（symlog）',ylabel='正向面积 / MPa·min（symlog）',title='幅度与持续累积共同影响强度');axs[1].grid(alpha=.15);fig.tight_layout();save(fig,'05_响应强度分数与物理量')
 # Population profiles of all target curves, not only selected exemplars.
 fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
 for code,ax in zip(range(1,7),axs.ravel()):
  ii=np.flatnonzero(d.morphology_code_v3==code)
  for j in ii:ax.plot(u,Y[j],color=COL[code-1],lw=.45,alpha=.05,rasterized=True)
  lo,md,hi=np.quantile(Y[ii],[.25,.5,.75],axis=0);ax.fill_between(u,lo,hi,color=COL[code-1],alpha=.25);ax.plot(u,md,color=COL[code-1],lw=2);ax.set_ylim(-.8,1.7);ax.set_title(f'M{code} {m.NAMES[code]} | n={len(ii)}');ax.grid(alpha=.15);ax.set_xlabel('观察窗口相对时间')
 axs[0,0].set_ylabel('按幅度尺度归一化ΔP');axs[1,0].set_ylabel('按幅度尺度归一化ΔP');fig.suptitle('目标形态及后期加速型：全部曲线与四分位范围');fig.tight_layout();save(fig,'06_目标形态全量曲线')
 # Feature definitions support user-requested metric set.
 dictionary=[('positive_rise_p95_mpa','正向涨幅','max(P95[P(t)−P0],0)','MPa'),('amplitude_mpa','变化幅度','P95[P(t)]−P05[P(t)]','MPa'),('overall_rate_mpa_min','整体回归涨幅速率','全观察窗口压力对时间的最小二乘斜率','MPa/min'),('endpoint_rate_mpa_min','端点平均涨幅速率','尾段压差/观察时长','MPa/min'),('early_rate_mpa_min_v3','前期速率','观察窗前1/3线性回归斜率','MPa/min'),('middle_rate_mpa_min_v3','中期速率','观察窗中1/3线性回归斜率','MPa/min'),('late_rate_mpa_min_v3','后期速率','观察窗后1/3线性回归斜率','MPa/min'),('early_middle_acceleration_mpa_min2','前中期加速度趋势','(中期速率−前期速率)/(T/3)','MPa/min²'),('middle_late_acceleration_mpa_min2','中后期加速度趋势','(后期速率−中期速率)/(T/3)','MPa/min²'),('acceleration_p95_native_mpa_min2','瞬时加速度P95','在原生或更粗时间上差分，不以插值增多点估计','MPa/min²'),('positive_auc_mpa_min','正向积分面积','积分max(P(t)−P0,0)dt，时间为分钟','MPa·min'),('signed_auc_mpa_min','有符号积分面积','积分(P(t)−P0)dt','MPa·min'),('positive_auc_mean_mpa','时长归一化正向面积','正向积分/T','MPa'),('rising_duration_min','累计上涨时间','至多21点趋势中，增量>max(0.01MPa,噪声)的区间时长之和','min'),('longest_rising_duration_min','最长连续上涨时间','连续满足上涨条件的最长区间时长','min'),('positive_duration_min','压力持续高于起点时间','趋势区间中点压差>max(0.05MPa,3×噪声)的时间','min'),('onset_fraction_v3','相对起涨位置','观察窗内首次连续两点超过噪声阈值的位置；未检出另有删失标记','0–1')]
 pd.DataFrame(dictionary,columns=['字段','指标','定义','单位']).to_csv(O/'09_指标定义.csv',index=False,encoding='utf-8-sig')
 morph_sens=pd.read_csv(O/'04_形态参数敏感性.csv');risk_sens=pd.read_csv(O/'07_响应强度权重敏感性.csv');free=pd.read_csv(O/'05_新增特征自由聚类对照.csv');br=[]
 for code in range(1,10):
  z=d[d.morphology_code_v3==code];br.append({'类型':f'M{code} {m.NAMES[code]}','数量':len(z),'弱响应':int(ct.loc[code,'弱']),'中响应':int(ct.loc[code,'中']),'强响应':int(ct.loc[code,'强'])})
 rules=[['M1 涨幅微弱','变化幅度≤0.2MPa且最大正向涨幅≤0.3MPa；优先识别','0.2MPa为探索性规则，已检验0.1–0.3MPa'],['M2 快速稳定','前期归一化斜率>0.25；后期斜率<前/中期较大值的55%；拟合早期饱和曲线','“稳定”指后段相对趋缓，不要求严格水平'],['M3 渐进上升','中期斜率>前/后期较大值的1.15倍；拟合S形曲线','前中期加速、中后期减速'],['M4 稳定缓升','各阶段持续上升，与直线模板更接近','“缓”是相对形态描述，不能据此定为弱响应'],['M5 先降后升','下降超过max(噪声阈值,幅度尺度8%)；谷值在观察窗2.5–65%；随后充分回升','初降必须超过可辨认噪声'],['M6 后期加速','后期斜率>前期斜率的1.25倍，拟合延迟幂函数上涨','保留尚未进入后期平台的形态'],['M7 冲高回落','峰值出现在窗内，回落比例>30%，尾段斜率为负','风险映射保持待核查'],['M8 降压主导','尾端压差明显为负，正向上涨有限','不能把降压直接解释为无压窜'],['M9 复杂/待判','形态条件均不满足或归一化拟合RMSE>0.30','不强制归入理想形态']]
 pd.DataFrame(rules,columns=['类型','判据','解释边界']).to_csv(O/'10_形态判定规则.csv',index=False,encoding='utf-8-sig')
 counts=d.response_strength_grade.value_counts();review=int(d.risk_mapping_review_required.sum());unc=int(d.morphology_uncertain.sum())
 report=f'''# V3：参考形态分类与强、中、弱响应分级

## 结论

**可以按图中的思路分类。本轮已在全部3180条曲线观测上完成重算，识别出五种目标形态，并保留实际数据中的额外形态。** 推荐采用“曲线形态 × 响应强度”的两层标签。当前强、中、弱描述压力响应，只有完成微地震、实际压窜及工况验证后，才能标定为风险等级。

这里使用的是**参考形态约束分类**：先定义图中要研究的形态家族，再由可计算指标和曲线拟合确定所属类型。它与完全无监督聚类不同，不能在论文中写成“算法无先验地发现了图示五种机理”。

![五种目标形态](图/01_五种目标形态与实测曲线.png)

## 1. 实际分类结果与多对一映射

{table(pd.DataFrame(br))}

五种目标类型合计 **{int((d.morphology_code_v3<=5).sum())}条**；后期加速型等额外记录保留独立去向。所有3180条均有结果，包括复杂待判记录，未因不符合示例形态而删除。

同一形态出现多种强度，说明仅凭形态名字推断风险会损失信息。例如稳定缓升型中，强响应 **{ct.loc[4,'强']}条**，通常具有更大的涨幅或更长的持续累积；先降后升型也并非必然属于强风险。

![交叉表](图/03_形态乘响应强度.png)

## 2. 指标体系

令P0为本条观察窗起始5%邻域的压力中位数，ΔP(t)=P(t)−P0，T为实际观察时长。统一采用起点参考便于比较缺少施工前基线的曲线；它反映观察窗内变化，可能低估记录开始前已经发生的响应。原先基于施工前基线的指标仍保留，不能混淆两种口径。

{table(pd.DataFrame(dictionary,columns=['字段','指标','定义','单位'])[['指标','定义','单位']])}

前、中、后期统一按**观察窗三等分**，与V2的前25%/中50%/后25%不同。所有速度、加速度和积分使用实际分钟尺度。整体斜率和端点平均速率都已导出；前10%和后10%速率另存用于检查起始和尾端。

形态曲线按max(P95−P05, 0.2MPa, 3倍噪声)归一化。趋势平滑在V2原生或更粗尺度上进行，窗口约为max(3分钟,观察时长5%)，不把101点插值当作新增观测。998条原生采样间隔>120秒的记录仍可用于总体趋势，但瞬时加速度精度有限。

“累计上涨时间”“最长连续上涨时间”和“压力高于起点的持续时间”是不同指标：压力进入平台时可能不再上涨，却仍维持高压力。强度分级同时考虑这种区别。持续时间采用至多21个趋势点估计，不声称亚采样间隔精度。

## 3. 图示形态如何被识别

{table(pd.DataFrame(rules,columns=['类型','判据','解释边界']))}

对非微弱曲线，在通过指标约束的家族内比较直线、早期指数饱和、S形、初降再上升、延迟幂函数、冲高回落和下降模板。拟合允许非负幅度缩放与常数偏移，最小化归一化残差，并施加每形状参数0.0006的复杂度惩罚，避免更灵活模板无条件胜出。网格参数、候选误差及选中模板逐条保存在过程目录。

**{unc}条标为形态边界/待核查**：拟合误差偏大、两种类型拟合接近、参数敏感或无法用单一模板解释。保留自动标签，同时标明不确定性，不把它们当作无误差真值。

### 自由聚类对照

在新增指标空间另行运行KMeans的5～9类方案。当前K={int(free.loc[free.silhouette.idxmax(),'k'])}的轮廓系数最高，但其分组与目标形态分类的ARI仅约{free.ari_to_guided_types.min():.2f}～{free.ari_to_guided_types.max():.2f}。这表明单纯自由聚类仍会混合形态与幅度差异，不能保证直接复现图中的命名体系。

{table(free.round(3).rename(columns={'k':'类别数','silhouette':'轮廓系数','min_cluster_n':'最小类数量','conditional_scale_group_ari_median':'分组重采样ARI中位数','ari_to_guided_types':'与参考分类ARI'}))}

自由聚类稳定性使用30次按平台分层、施工段整体抽取80%的重采样，特征尺度固定于全量数据，属于当前尺度条件下的结果；轮廓系数使用同一固定1600条随机样本。参考模板分类使用规则/平滑敏感性检验，不能用固定规则重复预测制造“100%稳定性”。

## 4. 强、中、弱如何确定

**形态标签不直接决定风险。** 对每条曲线，用六项单调增加的响应指标构成探索性强度分数：

- 正向涨幅：35%。
- 时间归一化正向面积：20%。
- 正向累计面积：10%。
- 高于起点的持续时间：10%。
- 累计上涨时间：5%。
- 前、中、后期最大正向速率：20%。

各指标先按固定物理尺度log1p变换，再除以当前样本P90并截断至1.5；保留各分量与贡献，能够逐条解释分数。面积、涨幅和时长有关联，因此用较小的累计面积/时长权重，并用去除累计量的替代权重检查重复计权影响。加速度趋势用于形态判定；没有把采样分辨率差异很大的瞬时加速度直接用于强度打分。

对强度分数进行一维KMeans三组划分，按组中心由小到大命名。当前分界为 **{cfg['intensity_cuts'][0]:.4f}、{cfg['intensity_cuts'][1]:.4f}**，得到弱响应 **{counts['弱']}条**、中响应 **{counts['中']}条**、强响应 **{counts['强']}条**。这些分界是本批数据下的探索性分界，**不是工程安全阈值，也不是MPa阈值**。

![强度分数](图/05_响应强度分数与物理量.png)

所有曲线都有计算响应等级。但 **{review}条的风险标签保持“待核查”**，涉及下降/回落/复杂形态、分类边界、身份不清、片段、冲突、操作突变或权重敏感；不能因为算出的正向涨幅小就直接标成低风险。其余{len(d)-review}条可作暂定风险标签候选，仍需要独立证据验证。

## 5. 稳健性与局限

### 形态参数敏感性

{table(morph_sens.round(4).rename(columns={'test':'改变条件','agreement':'同标签比例','ari':'ARI'}))}

### 强度分级敏感性

{table(risk_sens.round(4).rename(columns={'test':'替代方案','n':'记录数','agreement':'同等级比例','ari':'ARI','weak':'弱','medium':'中','strong':'强'}))}

这些比例衡量参数变化后的结果一致性，不是分类准确率。形态阈值、模板网格及权重是本轮明确公开的设计选择，尚未经过独立人工标签或微地震证据校准。可先作为论文的候选方法，后续冻结参数再做独立验证。

- 本轮沿用V2的全量3180条可辨认观测及其去重、片段、源文件身份审计，没有重新定义独立压窜事件数。
- 观察窗可能缺少早期或尾部。晚起记录会把“渐进”看成“稳定”，尚未观测到平台的S形可能被归为“后期加速”。
- 响应“强”不必然等于损害“重”；井筒容积、井口开关状态、压力恢复、注采工况等也影响压力。
- 初降后升不能仅凭曲线证明“应力激活天然裂缝”；快速稳定也不能单独证明“压裂缝直接沟通”。
- 图中的液强、排量、段长等措施参数未沿用，没有足够证据据此给出施工建议。

## 6. 后续风险验证怎么接

建议后续使用两套标签：**M1–M8形态标签（M9待判）**与**弱/中/强暂定响应等级**。用微地震连通性、距邻井最近事件/裂缝距离、裂缝逼近角、实际压窜记录和操作日志，核查“不同形态与强度组合是否对应不同风险”。验证后再生成正式风险标签。

地质工程预测模型可分别预测形态与风险，或优先预测最终验证后的三类风险；同一施工段的邻井曲线必须成组划分训练/验证集，避免泄漏。SHAP用于解释地质工程参数与已验证标签的关系，不能用本轮人工加权分数的分量贡献冒充SHAP。

## 7. 文件

- [全量形态与强度标签](01_全量曲线形态与强度标签.csv)
- [形态×强度交叉表](02_形态与强度交叉表.csv)
- [各类指标汇总](03_形态特征汇总.csv)
- [指标定义](09_指标定义.csv)与[明确判据](10_形态判定规则.csv)
- [自由聚类对照标签](06_自由聚类逐条标签.csv)
- [典型曲线原始来源](08_典型曲线来源.csv)
- 图目录提供6组PNG、PDF、SVG。
- 过程目录：`{B}`；代码顺序为01_classify.py、02_validate_models.py、03_report.py。V2和原始数据保留。

**本轮可以作为“参考形态约束分类＋响应强度分级”的候选方法与结果交付；不能作为已经验证的三类压窜风险结论。**
'''
 (O/'V3_图示形态与三级响应分级报告.md').write_text(report,encoding='utf-8')
 # Independent arithmetic verification of all integrals and segment slopes from the exported representation.
 integralerr=[];slopeerr=[]
 for i,r in d.iterrows():
  f,x,y=m.extract(raw[int(r.matrix_row)],r);o=raw[int(r.matrix_row)];t=(o['smooth_t']-o['smooth_t'][0])/60;p=o['smooth_p'].copy();dt=np.median(np.diff(t));win=min(max(3,int(round(max(3.,.05*t[-1])/dt))|1),len(p)//2*2-1)
  if win>=5:p=m.savgol_filter(p,win,2,mode='interp')
  delta=p-r.p0_window_mpa;v=np.maximum(delta,0);independent=float(np.sum((v[:-1]+v[1:])*.5*np.diff(t)));integralerr.append(abs(independent-r.positive_auc_mpa_min));mask=t<=t[-1]/3;tt=t[mask];pp=p[mask]
  if len(tt)>=3:slopeerr.append(abs(np.polyfit(tt,pp,1)[0]-r.early_rate_mpa_min_v3))
 checks=dict(n=len(d),unique_ids=d.sample_id.nunique()==len(d),all_labels_present=d.morphology_v3.notna().all(),cross_total=int(ct.to_numpy().sum()),area_all_rows_max_error=float(max(integralerr)),early_slope_all_eligible_max_error=float(max(slopeerr)),all_finite_scores=bool(np.isfinite(d.response_strength_score).all()),risk_all_unvalidated=bool((~d.risk_validated).all()),reference_count=int((d.morphology_code_v3<=5).sum()),source_snapshot_sha256=hashlib.sha256((OLD/'final_labels.pkl').read_bytes()).hexdigest(),raw_curve_snapshot_sha256=hashlib.sha256((OLD/'raw_curves.pkl').read_bytes()).hexdigest())
 checks={k:bool(v) if isinstance(v,np.bool_) else v for k,v in checks.items()};(B/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');assert max(integralerr)<1e-8 and max(slopeerr)<1e-8 and checks['unique_ids'] and checks['cross_total']==3180
 (B/'复算说明.md').write_text('''# V3复算

运行环境：`../时序聚类_20260915/venv/Scripts/python.exe`，保持V2的原生曲线与标签快照不变。

依次运行01_classify.py、02_validate_models.py、03_report.py。01提取指标并执行模板形态判定/强度分组；02比较自由聚类和规则/权重敏感性，保存质量标志与真实典型曲线；03生成图表、中文报告和独立积分/斜率核验。

config.json包含所有设计权重和阈值；template_library.json保留全部模板参数。results.pkl与结论01号CSV含3180条完整标签和指标。pressure_delta101.npy是相对观察起点压力；shape101.npy是形态归一化曲线；fitted_shape101.npy是模型拟合，不得冒充实测。所有图中的实线来自实际曲线，灰线为原始观测。

不要把参考形态分类称为完全无监督发现；不要把response_strength_grade称为已验证风险。risk_grade_provisional为待核查或暂定，risk_validated全部为False。风险分数权重人为指定、三组分界由当前样本估计，最终要由独立证据标定。

计算时长单位min，速率MPa/min，加速度MPa/min²，面积MPa·min。累计上涨时间与高于起点持续时间不同；shape归一化分母有0.2MPa与噪声下限，微弱曲线不会被任意放大。

本轮没有修改E盘源文件、V1和V2输出；只读取V2快照，原始来源沿用V2逐文件审计。全部中间过程与代码位于本目录，结果与科学图在对应结论目录。
''',encoding='utf-8')
 print('DONE',json.dumps(checks,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
