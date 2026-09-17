"""Scientific figures and source-backed Chinese report; all writes stay in experiment folders."""
import sys,json,html,warnings,textwrap
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.stdout.reconfigure(encoding='utf-8');warnings.filterwarnings('ignore')
B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915');G=O/'图';G.mkdir(exist_ok=True)
font=next((x for x in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if any(f.name==x for f in font_manager.fontManager.ttflist)),'DejaVu Sans')
plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'pdf.fonttype':42,'svg.fonttype':'none'})
colors=['#688397','#dc8651','#3564ae','#779450'];names={1:'低幅波动型',2:'早期上升趋缓型',3:'迟缓响应持续上升型',4:'降压主导型'}
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(G/(name+'.'+ext),bbox_inches='tight')
 plt.close(fig)
def table(df):
 cols=list(df.columns);out=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join(str(v).replace('|','/') for v in row)+' |')
 return '\n'.join(out)
def main():
 d=pd.read_pickle(B/'final_labels.pkl');sel=json.loads((B/'selected_model.json').read_text(encoding='utf-8'));assert len(d)==len(pd.read_pickle(B/'model_samples.pkl'))
 q=pd.read_csv(O/'11_四类形态特征汇总.csv');met=pd.read_csv(O/'05_算法与类别数比较.csv');st=pd.read_csv(O/'07_100次施工段分组稳定性.csv');sens=pd.read_csv(O/'10_质量与平台敏感性.csv');proto=pd.read_csv(O/'09_真实典型曲线索引.csv');audit=pd.read_csv(O/'03_逐通道利用去向.csv');inv=pd.read_csv(O/'01_全量文件覆盖.csv');full=pd.read_pickle(B/'broad_samples.pkl')
 raw=pd.read_pickle(B/'raw_curves.pkl');X=np.load(B/'curves101.npy')[d.matrix_row.astype(int)];S=np.load(B/'shapes101.npy')[d.matrix_row.astype(int)];u=np.linspace(0,1,101)
 # Refine lineage to final unique observations, resolving exact duplicates without inflating counts.
 duplicate=dict(zip(full.loc[~full.included,'sample_id'],full.loc[~full.included,'duplicate_sample_of']))
 audit['final_sample_id']=audit.sample_id.replace(duplicate);audit.loc[audit.sample_id.isin(duplicate),'status']='相同数值重复样本已合并';audit['final_cluster']=audit.final_sample_id.map(d.set_index('sample_id').cluster)
 audit.to_csv(O/'03_逐通道利用去向.csv',index=False,encoding='utf-8-sig')
 coverage=[]
 for r in inv.to_dict('records'):
  z=audit[audit.file==r['path']];ids=z.final_sample_id.dropna().unique();used=[i for i in ids if i in set(d.sample_id)]
  status='已用于形态聚类或重复合并' if len(used) else '未形成聚类样本'
  reason='；'.join((z.status.astype(str)+': '+z.reason.fillna('')).unique()) if len(z) else '非时序统计/施工表、空表、无法解析或时间压力配对尚不可靠，详见逐工作表审计'
  coverage.append(dict(file=r['path'],channel_count=r['channel_count'],final_status=status,related_unique_samples=len(used),reason=reason))
 cov=pd.DataFrame(coverage);cov.to_csv(O/'12_每个文件最终利用清单.csv',index=False,encoding='utf-8-sig')
 # 1: all curves in two physically distinct views. Ribbons are IQR, not confidence intervals.
 fig,axs=plt.subplots(2,4,figsize=(16,7.2),sharex=True)
 for j,cl in enumerate(range(1,5)):
  ix=np.flatnonzero(d.cluster==cl);c=colors[j]
  for row,M in enumerate([S,X]):
   ax=axs[row,j]
   for z in M[ix]:ax.plot(u,z,color=c,alpha=.035,lw=.45,rasterized=True)
   p25,md,p75=np.quantile(M[ix],[.25,.5,.75],axis=0);ax.fill_between(u,p25,p75,color=c,alpha=.3,label='25–75%分位');ax.plot(u,md,color=c,lw=2,label='中位曲线');ax.axhline(0,color='#777',lw=.5);ax.grid(alpha=.15)
   if row==0:ax.set_title(f'C{cl} {names[cl]}\nn={len(ix)}');lo,hi=np.quantile(M[ix],[.01,.99]);ax.set_ylim(min(-.2,lo),max(.2,hi));ax.text(.02,.02,'纵轴显示1–99%范围',transform=ax.transAxes,fontsize=7)
   else:lo,hi=np.quantile(M[ix],[.01,.99]);ax.set_ylim(min(-.2,lo),max(.2,hi));ax.set_xlabel('观测窗口相对时间（0–1）')
   if j==0:ax.set_ylabel('起点对齐的形态值' if row==0 else '相对基线压力 / MPa')
 axs[0,0].legend(fontsize=7);fig.suptitle(f'全量邻井压力曲线的四类形态（{len(d):,}条；每条均绘制）',fontsize=16,y=1.02);fig.text(.5,-.015,'阴影为样本四分位范围；各图纵轴范围不同。上排按幅度及噪声下限归一化；下排保留MPa。',ha='center',fontsize=9);fig.tight_layout();save(fig,'01_全量曲线四类形态')
 # 2: four-row interpretation table with real medoids and measurable features.
 fig=plt.figure(figsize=(15,12));gs=fig.add_gridspec(4,3,width_ratios=[1.35,2.1,2.3],hspace=.85,wspace=.32)
 for j,cl in enumerate(range(1,5)):
  z=d[d.cluster==cl];r=q[q.cluster==cl].iloc[0];pid=proto[proto.cluster==cl].iloc[0];idx=int(pid.model_row);ob=raw[int(d.iloc[idx].matrix_row)];a=fig.add_subplot(gs[j,0]);a.axis('off');a.text(0,.85,f'C{cl}  {names[cl]}',fontsize=14,color=colors[j],weight='bold');a.text(0,.28,f'{len(z):,} 条 | {z.platform_id.nunique()} 个平台\n占全部 {len(z)/len(d):.1%}',fontsize=11,linespacing=1.8)
  ax=fig.add_subplot(gs[j,1]);tm=(ob['t']-ob['t'][0])/60;yy=ob['p']-d.iloc[idx].baseline_mpa;ax.plot(tm,yy,lw=.65,color='#acb3bb',label='原始观测');ax.plot((ob['smooth_t']-ob['t'][0])/60,ob['smooth_p']-d.iloc[idx].baseline_mpa,lw=1.8,color=colors[j],label='平滑曲线');ax.set_ylabel('ΔP / MPa');ax.set_xlabel('观测起点后 / min');ax.grid(alpha=.2);ax.set_title(f"真实典型曲线：{d.iloc[idx].platform_id} / 源井{d.iloc[idx].source_well or '?'} / 段{d.iloc[idx].stage_id or '?'} / 邻井{d.iloc[idx].monitor_well}",fontsize=9)
  if j==0:ax.legend(fontsize=7)
  a=fig.add_subplot(gs[j,2]);a.axis('off');desc={1:'整体变化小；部分曲线存在晚期响应。',2:'较早出现上升，后段上升速率降低。',3:'早段变化较慢，后段持续或加速上升。',4:'观察窗口内降压占主导，需结合工况判读。'}[cl]
  txt=f"{desc}\n\n压力变化幅度：{r.amplitude_range_mpa_median:.3f} MPa\n早段速率：{r.early_rate_mpa_min_median:.3g} MPa/min\n尾段速率：{(0 if abs(r.tail_rate_mpa_min_median)<1e-10 else r.tail_rate_mpa_min_median):.3g} MPa/min\n幅度与速率均为本类中位数"
  a.text(0,.95,txt,va='top',fontsize=11,linespacing=1.6)
 fig.suptitle('邻井压力曲线形态分类与可测特征',fontsize=20,y=.98);fig.text(.5,.018,'形态类别尚未等同于压窜风险或裂缝连通机理；需结合微地震及开关井工况验证。',ha='center',fontsize=11);fig.subplots_adjust(top=.9,bottom=.085);save(fig,'02_四类典型曲线与分类依据')
 # 3 method comparison and group stability.
 fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 for method,g in met.groupby('method'):axs[0].plot(g.k,g.common_hybrid_silhouette,marker='o',label=method)
 axs[0].axvspan(2.8,5.2,color='#f0f3f8',zorder=-10);axs[0].set(xlabel='类别数 K',ylabel='共同混合距离的轮廓系数',xticks=range(2,7));axs[0].legend(fontsize=7);axs[0].grid(alpha=.2)
 for k in [3,4,5]:
  ar=st[st.k==k].ari;axs[1].boxplot(ar,positions=[k],widths=.45,tick_labels=[str(k)],showfliers=False);axs[1].text(k,.1,f'中位 {ar.median():.3f}\nP10 {ar.quantile(.1):.3f}',ha='center',fontsize=9)
 axs[1].set(xlabel='均衡特征 KMeans 的类别数',ylabel='100次按施工段分组重采样 ARI',ylim=(0,1.03),xticks=[3,4,5]);axs[1].grid(alpha=.2);fig.suptitle('类别数选择：分离度、稳定性和解释粒度共同判断');fig.tight_layout();save(fig,'03_方法类别数与稳定性')
 # 4 feature profile: class medians standardized across all observations.
 feats=['amplitude_range_mpa','end_dp_mpa','early_rate_mpa_min','middle_rate_mpa_min','tail_rate_mpa_min','early_to_late_acceleration_mpa_min2','onset_fraction'];labels=['变化幅度','尾端压差','早段速率','中段速率','尾段速率','早尾段速率变化/时间','观察窗内响应位置']
 a=d[feats];center=a.median();scale=(a.quantile(.9)-a.quantile(.1)).replace(0,1);vals=np.array([((d[d.cluster==cl][feats].median()-center)/scale).to_numpy() for cl in range(1,5)])
 fig,ax=plt.subplots(figsize=(11,4));im=ax.imshow(vals,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto');ax.set(xticks=range(len(feats)),xticklabels=labels,yticks=range(4),yticklabels=[f'C{i} {names[i]}' for i in range(1,5)]);plt.setp(ax.get_xticklabels(),rotation=18,ha='right');fig.colorbar(im,ax=ax,label='相对总体中位数 / 总体P90−P10')
 for i in range(4):
  for j in range(len(feats)):ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',fontsize=9)
 ax.set_title('四类形态的特征差异（描述性统计，非SHAP或因果重要性）');fig.tight_layout();save(fig,'04_分类特征热图')
 # 5 platform mix.
 ct=pd.crosstab(d.platform_id,d.cluster).reindex(columns=[1,2,3,4],fill_value=0);ct=ct.loc[ct.sum(axis=1).sort_values().index];fig,ax=plt.subplots(figsize=(11,9));left=np.zeros(len(ct))
 for cl in range(1,5):ax.barh(ct.index,ct[cl],left=left,label=f'C{cl} {names[cl]}',color=colors[cl-1]);left+=ct[cl].to_numpy()
 ax.set_xlabel('去重后曲线观测数');ax.set_title('各平台数据利用及类别分布');ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'05_各平台分类分布')
 # 6 sensitivity.
 fig,ax=plt.subplots(figsize=(11,6));z=sens[~sens.test.str.startswith('留一平台')];ax.barh(z.test,z.ari,color='#527dac');ax.set(xlim=(0,1.05),xlabel='与全量四类结果的 ARI',title='质量条件与特征权重敏感性（在保留样本上重新拟合）')
 for i,r in enumerate(z.itertuples()):ax.text(r.ari+.01,i,f'{r.ari:.3f}',va='center',fontsize=9)
 ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'06_敏感性检查')
 # Per-platform full population curve panels.
 pg=G/'逐平台全曲线';pg.mkdir(exist_ok=True)
 for plat,z in d.groupby('platform_id'):
  fig,axs=plt.subplots(1,4,figsize=(14,3),sharex=True)
  for cl,ax in zip(range(1,5),axs):
   ix=z[z.cluster==cl].index.to_numpy()
   for i in ix:ax.plot(u,S[i],color=colors[cl-1],alpha=.16,lw=.7)
   ax.set_title(f'C{cl} {names[cl]} | {len(ix)}');ax.set_ylim(-2.2,2.2);ax.set_xlabel('相对时间');ax.grid(alpha=.15)
  fig.suptitle(plat+'：全部已用曲线（统一显示形态范围 −2.2～2.2）');fig.tight_layout();fig.savefig(pg/(plat+'.png'),bbox_inches='tight',dpi=150);plt.close(fig)
 # Compact inspectable interactive curve catalogue using local data only.
 records=[]
 for i,r in d.iterrows():records.append(dict(id=r.sample_id,platform=r.platform_id,source=str(r.source_well),stage=str(r.stage_id),monitor=str(r.monitor_well),cluster=int(r.cluster),name=r.morphology_type,file=r.file,sheet=r.sheet,dt=round(r.native_dt_s,2),fragment=bool(r.fragment_only),conflict=bool(r.alternate_record_flag),agreement=round(r.label_agreement,3),duration=round(r.duration_min,2),y=np.round(X[i],4).tolist()))
 payload=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
 htmltext='''<!DOCTYPE html><html lang="zh"><meta charset="UTF-8"><title>全量曲线查询</title><style>body{font:16px 'Microsoft YaHei',sans-serif;max-width:1150px;margin:35px auto;background:#f4f6f9;color:#24364b}section{background:white;border-radius:10px;padding:25px;margin-top:20px}input,select{padding:10px;font:inherit;max-width:100%}svg{width:100%;height:410px}small{color:#586b80}#info{line-height:1.8;overflow-wrap:anywhere}button{padding:8px 15px}</style><h1>全量邻井压力曲线查询</h1><p>按平台、井段或文件名搜索；每条记录对应最终聚类标签。图中为101点重采样曲线，源文件路径可在下方核查。</p><input id="search" placeholder="输入平台、井段、文件名" style="width:65%"><select id="class"><option value="">全部类别</option><option value="1">C1 低幅波动</option><option value="2">C2 早期上升趋缓</option><option value="3">C3 迟缓响应持续上升</option><option value="4">C4 降压主导</option></select><section><p id="count"></p><select id="curve" style="width:100%"></select><svg id="plot" viewBox="0 0 1050 410" role="img" aria-label="所选曲线的相对基线压力"></svg><div id="info"></div></section><p><small>横轴为观测窗口内时间；缺少施工起点或基线时，不等于真实施工响应时间。曲线形态尚未验证为压窜风险标签。数据全部保存在本地。</small></p><script>const DATA=PAYLOAD;let selected=[];const $=id=>document.getElementById(id);function filter(){let t=$('search').value.toLowerCase(),c=$('class').value;selected=DATA.filter(r=>(!c||r.cluster==c)&&JSON.stringify([r.platform,r.source,r.stage,r.monitor,r.file]).toLowerCase().includes(t));$('count').textContent='匹配 '+selected.length+' / '+DATA.length+' 条';$('curve').replaceChildren(...selected.map((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.platform+' | 源井 '+r.source+' | 段 '+r.stage+' | 邻井 '+r.monitor+' | C'+r.cluster+' | '+r.id;return o}));draw()}function draw(){let r=selected[+$('curve').value];if(!r){$('plot').replaceChildren();$('info').textContent='无匹配曲线';return}let lo=Math.min(0,...r.y),hi=Math.max(0,...r.y);if(hi-lo<.02){hi+=.01;lo-=.01}let x=i=>70+i*9.3,y=v=>350-(v-lo)/(hi-lo)*310;let s='<path d="M70 40V350H1000" fill="none" stroke="#789"/>';for(let j=0;j<=5;j++){let v=lo+(hi-lo)*j/5;s+='<text x="60" y="'+(y(v)+5)+'" text-anchor="end" font-size="12">'+v.toFixed(2)+'</text><path d="M70 '+y(v)+'H1000" stroke="#e5eaf0"/>';s+='<text x="'+(70+j*186)+'" y="373" text-anchor="middle" font-size="12">'+(r.duration*j/5).toFixed(0)+'</text>'}s+='<text x="75" y="25" font-size="14">ΔP / MPa</text><text x="850" y="400" font-size="14">观测起点后 / min</text>';s+='<path d="'+r.y.map((v,i)=>(i?'L':'M')+x(i)+' '+y(v)).join(' ')+'" fill="none" stroke="#3564ae" stroke-width="2"/>';$('plot').innerHTML=s;$('info').replaceChildren();for(let t of ['C'+r.cluster+' '+r.name+'；分组重采样标签一致率 '+(100*r.agreement).toFixed(1)+'%','原生采样间隔 '+r.dt+' 秒；连续片段 '+r.fragment+'；同事件记录冲突 '+r.conflict,'源文件：'+r.file,'工作表：'+r.sheet+'；样本 ID：'+r.id]){let p=document.createElement('div');p.textContent=t;$('info').append(p)}}$('search').oninput=filter;$('class').onchange=filter;$('curve').onchange=draw;const init=new URLSearchParams(location.hash.slice(1));$('search').value=init.get('q')||'';$('class').value=init.get('class')||'';filter();</script></html>'''.replace('PAYLOAD',payload)
 (O/'全量曲线查询.html').write_text(htmltext,encoding='utf-8')
 # Report values computed from final outputs.
 ar=st[st.k==4].ari;chosen=met[(met.method=='Balanced_feature_KMeans')&(met.k==4)].iloc[0];ps=sens[sens.test.str.startswith('留一平台')];brief=[]
 for r in q.itertuples():brief.append({'类别':f'C{r.cluster} {r.name}','曲线数':r.n,'幅度中位数/MPa':f'{r.amplitude_range_mpa_median:.3f}','早段速率/MPa·min⁻¹':f'{r.early_rate_mpa_min_median:.4f}','尾段速率/MPa·min⁻¹':f'{r.tail_rate_mpa_min_median:.4f}'})
 no=audit[~audit.final_sample_id.isin(d.sample_id)];reasons=no.groupby(['status','reason'],dropna=False).size().reset_index(name='通道记录数');reasons.to_csv(O/'13_剩余未用记录原因.csv',index=False,encoding='utf-8-sig')
 stats=dict(curves=len(d),files=len(inv),files_with_channels=int((inv.channel_count>0).sum()),files_used=int((cov.related_unique_samples>0).sum()),platforms=d.platform_id.nunique(),lowfreq=int(d.low_resolution.sum()),fragments=int(d.fragment_only.sum()),alternates=int(d.alternate_record_flag.sum()),unknown_event=int((~d.event_identity_known).sum()),unknown_well=int((~d.well_identity_known).sum()),operational=int(d.operational_flag.sum()),boundary=int(d.boundary_flag.sum()),median_ari=float(ar.median()),p10_ari=float(ar.quantile(.1)),silhouette=float(chosen.common_hybrid_silhouette),source_channel_records=len(audit))
 (B/'report_numbers.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
 report=f'''# 邻井压力时序聚类：V2 全量利用与四类形态结果

## 1. 本轮结论

本轮逐一读取 **{len(inv):,} 个监测目录文件**，最终获得 **{len(d):,} 条去重后的压力曲线观测**，覆盖 **{d.platform_id.nunique()} 个来源平台**。相对首轮505条严格主分析样本，扩大至 **{len(d)/505:.2f}倍**。两轮纳入标准不同，这不是简单新增独立压窜事件数。

建议论文的**当前形态解释采用四类**：低幅波动、早期上升趋缓、迟缓响应持续上升、降压主导。这里的“迟缓”主要是观测窗口内的相对描述。四类是分离度与物理可解释粒度的折中，不是统计指标的全局最优解，也不是已经验证的风险等级。

{table(pd.DataFrame(brief))}

变化幅度定义为平滑压力的P95−P05；早段和尾段分别为观察窗口前25%及后25%的回归斜率。单位为MPa与MPa/min。统计为全类中位数，不是判定阈值。

![真实典型曲线与分类依据](图/02_四类典型曲线与分类依据.png)

## 2. 为什么首轮没有用上部分数据

首轮以可靠的施工起点、至少180秒施工前基线、较高时间覆盖度、较小采样间隔和缺口为主要门槛，适合对齐后的强度比较，却不适合作为全部形态数据的唯一入口。例如528条候选仅因施工前基线不足被放入补充组。重复导出也不应重复计数。

本轮改进：

- 合并新旧解析器，补充中文井号、TimeStamp、日期时间分列、No/通道编号、ch通道字典、多井分组表头和跨平台监测身份识别。
- 原始数据不移动、不删除。重复且一致的记录合并；不同读数的同事件记录保留并标记，避免强行拼接。
- 缺少施工前基线时，用观察起点邻域压力作参考，仍参与形态聚类；标签表明确记录参考方式。
- 低频记录保留整体变化趋势；不把插值生成的点当成新增观测。
- 长缺口曲线仅回收具有至少6点、至少5分钟、最长18小时的最长可解释连续片段；余下片段没有被自动解释为完整施工事件。
- 排除明确的施工井自身压力、砂浓/排量等非压力变量、阶段汇总指标、全零占位和无法可靠构造时序的记录。

### 本轮利用口径

- 至少成功提取压力通道的文件：**{stats['files_with_channels']} / {len(inv)}**；实际关联到最终样本或其一致重复记录的文件：**{stats['files_used']} / {len(inv)}**。
- 可用曲线：**{len(d)}条**，其中低频记录（原生间隔>120秒）**{stats['lowfreq']}条**、局部连续片段 **{stats['fragments']}条**。
- 同事件读数不一致标记：**{stats['alternates']}条**；源井或段号未明确：**{stats['unknown_event']}条**；监测井身份未明确：**{stats['unknown_well']}条**；近零压力大跳变工况标记：**{stats['operational']}条**。这些集合可以重叠，不能相加。
- 文件窗口 {int((d.window_basis=='observed_file_window').sum())}条；依段号和时钟匹配施工窗口 {int((d.window_basis=='same_stage_clock').sum())}条；日期时间匹配施工窗口 {int((d.window_basis=='absolute').sum())}条。后两类也受原始记录真实性及多井同时施工影响，不自动视为因果证据。

**“全部利用”在这里是全量扫描、尽量提取、逐项说明去向；并非把所有文件都强行变成样本。** 未使用数据和重复记录的准确去向见[每个文件最终利用清单](12_每个文件最终利用清单.csv)、[逐通道审计](03_逐通道利用去向.csv)、[剩余未用原因](13_剩余未用记录原因.csv)。工作表层面的多轮补读审计保存在02系列CSV。

## 3. 分类依据和算法设计

每条曲线保留实际压力、原生时间、采样分辨率与来源。处理顺序为时间去重与排序、有限缺口插值、原生或更粗时间尺度平滑、101点相对时间表示、特征提取。

1. **形态块（60%）**：10/25/50/75/90%时刻相对压力、终点值、早中尾段归一化斜率、初始下探、回落程度、峰谷位置、观察窗口内响应位置。形态归一化分母取P95−P05、0.2 MPa和3倍噪声估计三者最大值，避免把微弱噪声放大。
2. **幅度块（25%）**：压力范围、相对起点最大上升/下降、尾端相对参考基线压力；带符号log1p变换。
3. **动力学块（15%）**：早中尾段MPa/min速率、早尾段斜率差除以时间间隔。分别采用0.01 MPa/min和0.0001 MPa/min²尺度做带符号log1p变换。
4. 各块用总体P10–P90稳健尺度并截断至±3，再按特征数量归一化、乘块权重的平方根。

用户提出的起始/尾端涨幅、速率、加速度和响应时间均有对应导出字段。瞬时加速度分位数另行导出供审查，**没有把不同采样频率下的瞬时加速度直接混入主模型**。主模型采用跨早尾段的加速趋势，适合当前混合采样频率。观察窗内响应位置包含右删失标记；未超过噪声阈值不等于真实无响应。

对比 **均衡特征KMeans、均衡特征Ward、形态KMeans、形态DTW与特征混合距离K-medoids**，各自K=2～6，共20个方案。DTW采用51点、±5点约束；共同评价距离为中位距离归一化后的50%形态DTW与50%均衡特征欧氏距离。轮廓系数在固定随机抽取的1600条曲线上计算，各方案使用同一批点。最终模型使用全部曲线。

## 4. 为什么推荐四类，而不是强行五类

- **三类**主要区分上升、低幅和下降，上升组里不同起涨与尾端速率仍被合并。
- **四类**进一步区分“早升后缓”和“迟缓但持续上升”，回应了时序形态的解释目标。
- **五类**的主要增量是把持续上升曲线按幅度继续细分；其标签已完整保存在候选标签文件中，可作为论文补充结果。当前不能据此声称得到五种独立压窜机理。
- 某些纯形态方案出现很小的突变曲线组，因此不能只看较高的轮廓系数。

所选四类的共同距离轮廓系数为 **{chosen.common_hybrid_silhouette:.3f}**。100次分平台、按施工段整体抽取80%组的重采样中，ARI中位数 **{ar.median():.3f}**、第10百分位 **{ar.quantile(.1):.3f}**。ARI衡量重新拟合后分组的一致性，不是预测准确率，更不是风险识别正确率。

![方法比较与稳定性](图/03_方法类别数与稳定性.png)

[全部20个候选方案](05_算法与类别数比较.csv)；[100次分组重采样](07_100次施工段分组稳定性.csv)。同一施工段及同段的冲突导出作为组保留，避免单条曲线随机拆分夸大稳定性；未知事件只能按平台及记录日粗分组。KMeans重采样时重新估计特征尺度；混合DTW模型的筛选稳定性以全量距离尺度为条件。

## 5. 质量敏感性与解释边界

{table(sens[~sens.test.str.startswith('留一平台')].assign(ari=lambda x:x.ari.map(lambda v:f'{v:.3f}')).rename(columns={'test':'检验','n':'保留观测数','ari':'ARI'}))}

留一平台检验（仅对样本≥20的平台）ARI范围 **{ps.ari.min():.3f}～{ps.ari.max():.3f}**。这些检验衡量当前形态划分对样本构成的敏感性，不是外部平台验证。

**{stats['boundary']}条曲线被标为分类边界**：分组重采样标签一致率<80%，或共同距离轮廓系数<0。全部仍有标签，后续建立地质工程预测模型时，应先核查边界、工况、事件对应与监测井身份，不能把所有自动标签当成无误差真值。

- 四类是形态标签，不能直接对应高/中/低风险；C1低幅和C4降压也不能直接解释为无压窜。
- 没有可靠施工起点时，只能计算观测起点后的相对响应时间。依时钟匹配的记录不能提供独立可靠日期证据。
- 不同采样分辨率可能影响微幅组与降压组占比；原生间隔、平滑尺度和质量标志已逐条导出。
- 形态名称不是对所有成员的严格规则；每类内部仍有幅度和工况差异。
- 初降后升可作为附加描述字段进行核查，当前聚类没有证明其能独立形成稳定、充足的第五类。
- 微地震、裂缝逼近角和长度、地质工程预测、SHAP均尚未在本轮执行；不得把特征热图写成SHAP重要性。
- 本轮按用户要求使用全量探索数据，没有保留用于声称外部泛化的独立盲测集；后续预测模型应按平台或施工井组划分训练/验证集。

## 6. 可用于论文的结果表述

> 对监测目录内全部文件进行逐文件读取、通道识别与质量审查后，共形成{len(d)}条可辨认的邻井压力曲线观测。通过综合压力变化幅度、分段变化速率、相对起涨位置与曲线形态，对四种聚类方法及2～6个类别进行对比。兼顾样本支持度、重采样稳定性与解释粒度，采用均衡特征KMeans的四类结果作为当前形态描述方案，得到低幅波动型、早期上升趋缓型、迟缓响应持续上升型和降压主导型。四类结果在施工段分组重采样中的ARI中位数为{ar.median():.3f}。该分类反映压力动态响应差异，尚需结合微地震与生产操作记录验证其与裂缝连通及压窜风险的对应关系。

## 7. 交付文件与复算

- [全量标签与质量字段](08_全量曲线聚类标签.csv)
- [四类特征汇总](11_四类形态特征汇总.csv)
- [真实典型曲线索引](09_真实典型曲线索引.csv)
- [本地逐曲线查询页面](全量曲线查询.html)：搜索平台/井段/文件，查看每条压力曲线和来源。
- 图目录提供6组PNG/PDF/SVG；逐平台全曲线目录保留每个平台的全部曲线形态面板。
- 全部代码、中间数组、原生曲线、模型和缓存：`{B}`。
- 复算顺序和字段定义见过程目录的`复算说明.md`；原始E盘源文件只读。

## 方法资料

- [scikit-learn：聚类方法及评价](https://scikit-learn.org/stable/modules/clustering.html)
- [tslearn：时序聚类与DTW](https://tslearn.readthedocs.io/en/latest/user_guide/clustering.html)
- [SciPy：Savitzky–Golay滤波](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)

以上资料用于方法说明；本报告数值均来自本地实际数据运算。方案文档和示例图作为研究意图参考，其机理与措施未被当成已证实结论。
'''
 (O/'时序聚类_V2_全量结果报告.md').write_text(report,encoding='utf-8')
 print('REPORT',json.dumps(stats,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
