from pathlib import Path
import pandas as pd,numpy as np,json,sys,pickle,re
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');O=Path('F:/论文库/IGS/实验/结论/地质工程因素_20260916')
d=pd.read_pickle(B/'stage_factors_final.pkl');p=pd.read_pickle(B/'pair_factors.pkl');cl=pd.read_pickle(B/'classification_linkage.pkl');g=pd.read_pickle(B/'layers.pkl');src=pd.read_pickle(B/'source_registry.pkl');comp=pd.read_pickle(B/'legacy_comparison.pkl');si=pd.read_pickle(B/'source_issues.pkl');co=pd.read_pickle(B/'factor_conflicts.pkl');cover=pd.read_pickle(B/'platform_coverage.pkl')
# Ensure numeric missing cells are blanks, preserving zero separately.
rename={'platform':'平台','well':'井号','stage':'压裂段','sample_id':'曲线样本ID','monitor_platform':'邻井平台','monitor_well':'邻井号','join_valid':'可关联','join_issue':'关联问题','identity_correction':'身份修正依据','original_platform':'原解析平台','original_source_well':'原解析井号','morphology_code':'原形态编号','original_class':'原曲线分类','five_class':'目标5类','response_strength':'压力响应强度','morphology_uncertain':'形态边界待核查','source':'来源定位','sheet':'表名','top':'小层顶深_m','bottom':'小层底深_m','layer':'小层','key':'井段键','field':'指标','issue':'核查问题','values':'候选值及来源','raw_value':'异常原值','raw_nu':'异常泊松比'}
# Pair outputs omit duplicated detailed source-stage mechanics; complete details remain in stage sheet.
p['井距_m']=p.get('井距_m',np.nan)
cols=list(p.columns);cols.remove('井距_m');cols.insert(11,'井距_m');p=p[cols]
for frame in [p,d]:
 for c in frame:
  if c.endswith('来源') or c in ['地质来源','射孔_地质来源']:continue
  if frame[c].dtype==object:frame[c]=frame[c].map(lambda x:None if isinstance(x,str) and x=='nan' else x)
# Keep all numerical stage factors, with source IDs at the end.
d=d.drop(columns=[c for c in ['地质来源','射孔_地质来源','分段来源','深度基准说明'] if c in d])
front=['井段键','platform','well','stage','段顶测深_m','段底测深_m','测深段长_m','地质计算状态','解释覆盖率'];d=d[front+[c for c in d if c not in front]].rename(columns=rename)
cl=cl.rename(columns=rename);g=g.rename(columns=rename);si=si.rename(columns=rename);co=co.rename(columns=rename)
# Long source descriptions stay in source table/CSV, not repeated in each output record.
quality=pd.concat([si,co],ignore_index=True).fillna('');quality=quality[['井段键','指标','核查问题','异常原值','异常泊松比','候选值及来源','来源定位']]
# Independent hand-check example with unequal elastic moduli across intersected layers.
stagefull=pd.read_pickle(B/'stage_factors_final.pkl');layers=pd.read_pickle(B/'layers.pkl');example=None
for _,r in stagefull[(stagefull.platform=='泸201H5')&(stagefull.well==1)].iterrows():
 q=layers[(layers.platform==r.platform)&(layers.well==r.well)&(layers.top<r['段底测深_m'])&(layers.bottom>r['段顶测深_m'])]
 if len(q)>1 and q['杨氏模量_GPa'].nunique()>1:example=(r,q);break
r,q=example;a,b=float(r['段顶测深_m']),float(r['段底测深_m']);w=np.minimum(q.bottom,b)-np.maximum(q.top,a);calc=float(np.dot(w,q['杨氏模量_GPa'])/sum(w));assert abs(calc-r['杨氏模量_GPa'])<1e-8
ex=pd.DataFrame({'井段键':[r['井段键']]*len(q),'段顶测深_m':[a]*len(q),'段底测深_m':[b]*len(q),'小层顶深_m':q.top.to_list(),'小层底深_m':q.bottom.to_list(),'杨氏模量_GPa':q['杨氏模量_GPa'].to_list(),'重叠长度_m':[None]*len(q),'长度乘模量':[None]*len(q),'来源定位':q.source.to_list()})
notes=[
('记录范围','总表包含同平台已知井的候选邻井组合，以及曲线/原统计明确出现的跨平台组合。候选组合不表示已发生压窜。'),
('分类列','第三列是目标5类：涨幅微弱、快速稳定、渐进上升、稳定缓升、先降后升。保留原V3分类；其他形态标为待归入5类。未重新训练或强制归并类别。'),
('无曲线与冲突','暂无可关联曲线不等于无压窜或弱风险。同一组合多记录分类不一致时不投票合并。全部3180条曲线均保留在曲线关联表。'),
('井段地质加权','各小层参数以其与分段测深区间的重叠长度为权重。均值=Σ(参数×有效重叠长度)/Σ有效重叠长度。每个指标单独计算覆盖率，缺值不当作0。'),
('深度基准','小层顶底深按测深与分段匹配。多数源表只写顶深/底深，MD基准仍需核实；未将垂深直接用于此重叠计算。超出解释范围不外推。'),
('射孔簇参数','井段表中射孔_前缀表示仅在逐簇实际记录区间内加权。射孔区间取并集防止重复计算。设计孔数及簇中心距不冒充实际施工参数。'),
('工程参数优先级','压窜数据统计—1.0.xlsx优先，缺项以平台实际施工表补充。原表同一源井段在不同邻井行的参数一致时共享；同一组合有直接值时优先。冲突值和来源另列。'),
('工程版本冲突','平台施工表存在多个版本时，优先记录较完整文件并参考文件名日期；这不保证其为最终修订版。814条多来源字段差异需复核。零排量/零强度保留原值，应核查是否未施工记录。'),
('工程计量口径','平均排量、主体排量、最低/最高排量原义保留；范围排量未擅取中点。净液量和总液量分列，用液强度与不同液量口径的差异不能直接认定为录入错误。'),
('力学派生参数','在每个有效小层先计算水平应力差、差系数、平面应变模量E/(1-ν²)、剪切模量E/[2(1+ν)]及体积模量E/[3(1-2ν)]，再按区间加权。后3项采用各向同性弹性近似，保留测井解释的动态/静态口径不明限制。'),
('应力字段','源表写最大/最小主应力，未全部明确水平主应力。本表沿用源字段；水平应力差按两者差值暂算，不将其视为已校准地应力。'),
('异常值','泊松比超出0—0.5、杨氏模量与标注GPa量级明显不符等值列为待核查。未凭数值大小自动改单位或移动原表列。'),
('井距','井距_m来自原统计。重算井距为源井段中点至邻井轨迹最短平面距离，含坐标基准假设；缺坐标或轨迹一致性异常时留空。两种井距口径并不必然一致。'),
('天然裂缝','带入既有泸201H5参考层位结果，其逼近角含井眼方位+90°方向假设。解释线总长是统计窗口内线长，不等于单条裂缝长度；其余平台不以全时窗属性投影填充真实裂缝指标。'),
('来源与复算','来源编号对应来源定位表。逐项权重计算、字段有效覆盖和来源明细保存在过程目录的逐项计算与来源.csv。当前Excel为本次数据处理快照；源资料更新需复算。'),
('后续建模','第三列属于压力曲线形态标签，强中弱仅为压力响应强度。尚未通过微地震等独立证据验证为风险等级。训练时排除候选无标签和冲突记录，按平台/源井段分组验证。')]
frames={'井段邻井因素':p,'全部井段参数':d,'平台覆盖':cover,'曲线关联':cl,'原统计对比':comp,'小层解释输入':g,'质量核查':quality,'来源定位':src,'加权计算示例':ex,'方法说明':pd.DataFrame(notes,columns=['项目','说明'])}
# Independent integrity controls.
assert len(cl)==3180 and cl['曲线样本ID'].nunique()==3180
assert len(p)==p['压裂井-压裂段-压窜井'].nunique()
assert p['关联曲线数'].sum()==cl['可关联'].sum()
assert list(p.columns)[2]=='压窜分类_目标5类'
for c in d:
 if '覆盖率' in c:assert d[c].dropna().between(0,1.000001).all(),c
payload={}
for name,frame in frames.items():
 rows=json.loads(frame.to_json(orient='values',force_ascii=False,double_precision=12));payload[name]={'columns':list(frame.columns),'rows':rows}
(B/'workbook_data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
checks={'passed':True,'sheets':{k:len(v) for k,v in frames.items()},'example_key':r['井段键'],'example_young_GPa':calc,'classification_identity_partition':[int(cl['可关联'].sum()),int((~cl['可关联']).sum())],'fracture_computed_pairs':int(p['天然裂缝解释线总长_m'].notna().sum())}
(B/'data_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(checks,ensure_ascii=False))
(O/'计算口径与缺项说明.md').write_text('# 地质工程因素汇总\n\n'+'\n\n'.join('## '+k+'\n\n'+v for k,v in notes),encoding='utf-8')
