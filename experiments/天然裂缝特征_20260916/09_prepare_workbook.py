from pathlib import Path
import json,sys,re
import numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');O=Path('F:/论文库/IGS/实验/结论/天然裂缝特征_20260916')
REF=Path('E:/code/天然裂缝和压窜积液量的关系')
def branches(edges):
 adj={};lens=[];nodes=[]
 for i,e in enumerate(edges):
  a=tuple(np.round(e[:2],5));b=tuple(np.round(e[2:],5));nodes.append((a,b));lens.append(float(np.linalg.norm(np.array(e[2:])-e[:2])));adj.setdefault(a,[]).append(i);adj.setdefault(b,[]).append(i)
 used=set();out=[]
 def walk(start,i):
  length=0.;node=start
  while i not in used:
   used.add(i);length+=lens[i];a,b=nodes[i];node=b if node==a else a
   if len(adj[node])!=2:break
   candidates=[j for j in adj[node] if j not in used]
   if not candidates:break
   i=candidates[0]
  out.append(length)
 for node,es in adj.items():
  if len(es)!=2:
   for i in es:
    if i not in used:walk(node,i)
 for i,(node,_) in enumerate(nodes):
  if i not in used:walk(node,i)
 assert len(used)==len(edges);assert abs(sum(out)-sum(lens))<1e-6
 return dict(branch_count=len(out),branch_max_m=max(out) if out else 0.,branch_mean_m=float(np.mean(out)) if out else 0.)
assert branches([[0,0,10,0],[10,0,20,0]])['branch_max_m']==20
assert branches([[0,0,10,0],[10,0,20,0],[10,0,10,10]])['branch_count']==3
q=pd.read_pickle(B/'pair_features.pkl');edges=json.loads((B/'pair_edges.json').read_text(encoding='utf-8'))
for c in ['branch_count','branch_max_m','branch_mean_m']:q[c]=np.nan
for i,r in q.iterrows():
 if r.feature_key in edges:
  for k,v in branches(edges[r.feature_key]).items():q.loc[i,k]=v
q['usable_for_exploratory_model']=q.line_length_m.notna() & (q.coverage_fraction>=.95)
q.to_pickle(B/'pair_features.pkl');q.to_csv(O/'天然裂缝_井段邻井特征.csv',index=False,encoding='utf-8-sig')
# Workbook inputs keep measured/derived quantities numeric and leave unavailable values blank.
q['stage_sort']=pd.to_numeric(q.stage,errors='coerce');q=q.sort_values(['platform','source_well','stage_sort','neighbor'],na_position='last');q=pd.concat([q[q.line_length_m.notna()],q[q.line_length_m.isna() & (q.pressure_curves>0)],q[q.line_length_m.isna() & (q.pressure_curves==0)]])
columns={'platform':'平台','source_well':'压裂井','stage':'施工段','neighbor':'邻井','theta_deg':'天然裂缝逼近角\n°；方向假设','line_length_m':'天然裂缝段长\nm；窗口总线长','development_km_km2':'天然裂缝发育程度\nkm/km²','branch_max_m':'最长骨架分支\nm','branch_mean_m':'平均骨架分支\nm','coverage_fraction':'窗口有效覆盖率','area_m2':'有效面积\nm²','full_window_area_m2':'完整窗口面积\nm²','theta_pair_deg':'相对井间方向夹角\n°','pressure_curves':'关联压力曲线数','status':'计算状态及限制','sample_scope':'记录来源','feature_key':'关联键','assumed_hf_azimuth_deg':'假定水力裂缝方位\n°','well_azimuth_deg':'井眼网格方位\n°','md_top':'分段顶界MD\nm','md_bottom':'分段底界MD\nm','geometry_status':'几何定位状态','stage_source':'分段源文件','trajectory_source':'井轨迹源文件'}
main=q[list(columns)].rename(columns=columns);main['施工段']=main['施工段'].map(lambda v:int(v) if str(v).isdigit() else v)
cov=pd.read_csv(O/'平台数据覆盖与待补项.csv').fillna('');cov['覆盖≥95%已计算组合']=[int(((q.platform==p)&q.usable_for_exploratory_model).sum()) for p in cov.platform];cov=cov.rename(columns={'platform':'平台','sgy_group':'SGY数据组','horizon_available':'有配套层位','designed_stages':'已解析设计段','survey_wells':'有轨迹井数','located_survey_wells':'可定位井数','candidate_pairs':'候选井段邻井组合','calculated_pairs':'已计算组合','pressure_curves':'压力曲线数','matched_pressure_curves':'已关联特征压力曲线数','next_input':'后续所需资料'})
sgy=json.loads((B/'sgy_inventory.json').read_text(encoding='utf-8'));seismic=[]
for r in sgy:
 p=Path(r['path']);text=(B/(p.stem+'_header.txt')).read_text(encoding='utf-8');offset=re.search(r'CDP X\s*:\s*position\s*=\s*(\d+)',text)
 seismic.append({'数据组':p.parent.name,'属性文件':p.name,'道数':r['n'],'采样点数':r['ns'],'采样间隔_ms':r['dt']/1000,'起始时间_ms':r['start'][0],'结束时间_ms':r['start'][0]+(r['ns']-1)*r['dt']/1000,'样值格式':'大端IBM浮点','坐标X字节位置':int(offset[1]) if offset else None,'字节数':r['size'],'目标层位文件':'LU205H5-0603-O3w.hrzdat' if p.parent.name=='泸201H5' else '未检索到','本次用途':'层位采样及骨架提取' if p.parent.name=='泸201H5' else '数据覆盖与格式核查；等待层位','完整源路径':str(p)})
st=pd.read_csv(B/'stage_geometry.csv');geo=st[['platform','well','stage','md_top','md_bottom','east','north','azimuth','geometry_status','file','trajectory_file']].rename(columns={'platform':'平台','well':'井号','stage':'施工段','md_top':'顶界MD_m','md_bottom':'底界MD_m','east':'东坐标_m','north':'北坐标_m','azimuth':'网格方位_°','geometry_status':'定位状态','file':'分段源文件','trajectory_file':'轨迹源文件'})
notes=[
['结果范围','本表为现有资料下的阶段性结果。其他平台缺目标层位/时深转换等关键资料，保留空值及原因。'],
['记录粒度','一条记录为一个压裂井、一个设计施工段、一个邻井。设计候选组合不表示发生过压窜或有监测。'],
['逼近角','局部60 m邻域PCA方向与假定水力裂缝方向的锐角，按线段长度加权，范围0—90°。'],
['水力裂缝方向假设','沿用参考算法：施工段井眼网格方位+90°（模180°）；尚无已确认的逐段最大水平主应力方位。'],
['井间方向夹角','同一骨架方向相对源段中点至邻井水平段最近点连线的长度加权锐角；与应力逼近角含义不同。'],
['天然裂缝段长','指计算窗口与有效覆盖范围内提取的地震线骨架总长度；不是单条地下天然裂缝的真实三维长度。'],
['最长/平均骨架分支','骨架网络中端点或分叉点之间连续路径的长度统计；闭合环作为一条路径。分支不等同于独立地质裂缝。'],
['天然裂缝发育程度','二维线密度P21代理：总线长(m)/有效覆盖面积(m²)×1000，单位km/km²；未作地质标定。'],
['属性与阈值','MCANT参考层位最近样点值≥0.2；保留至少4个像元的8邻域连通域，再细化骨架。'],
['边界剔除','空间叠合发现规则边界直线伪影；最终提取前剔除网格外缘两格（40 m），有效面积同步扣除。'],
['窗口定义','以源段中点为原点，沿井方向初始±200 m，横向覆盖源井与邻井最近点并各外扩50 m；错列时沿井方向扩展至包含邻井点并留50 m。'],
['窗口面积','源井与邻井有错列或相距较远时面积变化；发育程度用有效覆盖面积归一化。不同支持范围仍会影响结果。'],
['覆盖率','有效覆盖面积/完整窗口面积；不足95%加标记。局部观测量不外推至未覆盖区域，建模应单独筛选。'],
['零与空值','线长为0表示当前阈值未检出线段，此时逼近角空值；其他空值为缺输入或未完成计算，不作为0。'],
['层位限制','O3w层位为参考层位，不能直接当作各射孔深度；SGY与层位标注时间域，其他平台不可按井深数值直接切片。'],
['地震命名','数据归档于泸201H5，SGY文件名为LU205H5；沿用示例空间关联，正式论文前需确认归属、投影和垂向基准。'],
['阈值敏感性','额外计算0.1和0.3；0.1与0.2在本属性级别上结果相同，不能据此认为已完成广泛稳定性验证。'],
['复现验证','原算法98段固定方窗线长/角度和井段坐标已复现；最终结果另采用井间窗口并剔除边界伪影，因此不应与旧固定方窗值直接等同。'],
['压力关联','全部3180条压力样本的关联键已保留在配套CSV；重复观测共享静态特征，不增加独立地质样本数。'],
['参考代码',str(REF/'PPT_天然裂缝定量表征_0915/analyze_fractures.py')],
['井间窗口参考',str(REF/'PPT_天然裂缝定量表征_0915/修订版/analyze_v2.py')],
['SGY规范','SEG-Y Rev1：坐标倍率与CDP字节位置。https://seg.org/wp-content/uploads/2025/11/seg_y_rev1.pdf'],
]
frames=[('井段邻井特征',main),('平台覆盖',cov),('井段定位',geo),('地震数据',pd.DataFrame(seismic)),('指标说明',pd.DataFrame(notes,columns=['项目','定义及依据']))]
payload={name:{'columns':list(df.columns),'rows':json.loads(df.to_json(orient='values',force_ascii=False))} for name,df in frames}
(B/'workbook_data.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
qcalc=q[q.line_length_m.notna()];stats={'total_pair_rows':len(q),'calculated_pair_rows':len(qcalc),'positive_line_rows':int((qcalc.line_length_m>0).sum()),'zero_line_rows':int((qcalc.line_length_m==0).sum()),'partial_coverage_rows':int((qcalc.coverage_fraction<.95).sum()),'coverage95_rows':int(qcalc.usable_for_exploratory_model.sum()),'pressure_curves_linked':int(qcalc.pressure_curves.sum()),'platform_rows':len(cov),'branch_max_m':float(qcalc.branch_max_m.max())};(B/'delivery_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(stats)
report=f'''# 天然裂缝特征提取：阶段性结果\n\n## 已完成范围\n\n- 泸201H5：98个设计井段、{len(qcalc)}条有方向的井段—邻井组合，关联295条压力曲线。\n- {stats['positive_line_rows']}条组合检出线段；{stats['zero_line_rows']}条未检出（角度无定义）。\n- {stats['partial_coverage_rows']}条窗口有效覆盖不足95%；建模用表中覆盖率筛选。\n- 全表共{len(q)}条设计候选及实测井对记录。候选组合不是已发生压窜样本。\n- 其他平台尚未完成裂缝数值提取；主要缺目标层位/时深转换，另有井段或轨迹缺失及版本冲突。\n\n## 方法及解释边界\n\n'''+ '\n\n'.join(f'### {a}\n\n{b}' for a,b in notes[:19])+'''\n\n## 验证\n\n原始SGY层位属性与旧缓存逐点一致。原算法98段固定400 m方窗结果已复现；线长、角度、坐标误差均低于1e-6。主结果采用明确的井间窗口和两格边界剔除。P21独立复算、分支线长守恒、零线长角度空值、唯一键及3180条压力样本全量映射均通过。\n\n## 补齐后继续\n\n为每个SGY数据组补目标层位网格（含XY与时间）或经确认的时深转换，核对井口坐标和射孔分段区间。提供实际最大水平主应力方位后可替换逼近角方向假设。不能用其他平台的层位、压力类别或模拟裂缝补造缺失值。\n'''
(O/'天然裂缝特征提取说明.md').write_text(report,encoding='utf-8')
