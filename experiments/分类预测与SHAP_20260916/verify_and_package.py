from pathlib import Path
import sys,json,numpy as np,pandas as pd,joblib
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import train_models as tm
# Re-save script-created pickles with an importable class reference.
import __main__
__main__.FoldColumns=tm.FoldColumns
B,O=tm.B,tm.O;d=pd.read_pickle(B/'model_cohort.pkl');x=pd.read_pickle(B/'model_inputs.pkl');z=np.load(B/'oof_arrays.npz');fold=z['fold'];y=z['y'];sv=z['shap'];g=d['验证平台组'].to_numpy()
for path in B.glob('*.joblib'):
    m=joblib.load(path);joblib.dump(m,path)
    if path.name.startswith('fold'):
        k=int(path.name[4]);name=path.stem.split('_',1)[1];assert np.max(np.abs(m.predict_proba(x.iloc[fold==k])-z[name][fold==k]))<1e-7
s=pd.read_csv(B/'hyperparameter_search.csv')
for name in ['Logistic','RandomForest']:
    best=int(s[s.model==name].groupby('candidate').inner_macro_F1.mean().idxmax());p,w=tm.CAND[name][best];m=tm.fit_model(name,p,x,y,w);joblib.dump(m,B/(name+'_final_pipeline.joblib'))
rows=[]
for feature,j in [('重算井距_段中点平面_m',1),('最小主应力_MPa',0),('水平应力差_MPa',0),('加砂强度_t_m',0)]:
    k=tm.FEATURES.index(feature)
    for group in np.unique(g):
        mask=(g==group)&x[feature].notna().to_numpy();xx=x.loc[mask,feature].to_numpy();yy=sv[mask,k,j]
        # Within-group descriptive association, report only adequately varying support.
        if len(xx)>=15 and len(np.unique(xx))>=5 and np.std(yy)>1e-9:
            rows.append({'feature':feature,'class':'M'+str(j+1),'group':group,'observed_n':len(xx),'rho':spearmanr(xx,yy).statistic})
pd.DataFrame(rows).to_csv(O/'13_SHAP组内方向核查.csv',index=False,encoding='utf-8-sig')
# Compact data delivery with clear distinctions between observations and fitted outputs.
def pack(df):return {'columns':list(df.columns),'rows':json.loads(df.to_json(orient='values',force_ascii=False))}
rename={'model':'模型','macro_F1':'宏平均F1','balanced_accuracy':'平衡准确率','accuracy':'准确率','log_loss':'对数损失','Brier_multiclass':'多类Brier','macro_AUROC':'宏平均AUROC','F1_CI_low':'F1区间下限','F1_CI_high':'F1区间上限','feature':'特征','mean_abs_SHAP_margin':'平均绝对SHAP_原始得分','platform_equal_mean_abs_SHAP':'平台组等权SHAP','missing_fraction':'缺失比例','experiment':'对照实验','n':'样本数','groups':'分组数','class':'类型','recall':'召回率','precision':'精确率','prevalence':'样本占比'}
files={'模型比较':'01_模型总体表现.csv','分类别表现':'06_XGBoost分类别表现.csv','SHAP重要性':'05_SHAP重要性.csv','对照实验':'04_敏感性与特征组对照.csv','折外预测':'03_逐样本折外预测.csv','分组表现':'07_各平台组表现.csv'}
data={n:pack(pd.read_csv(O/f,dtype={'曲线样本ID':str}).rename(columns=rename)) for n,f in files.items()}
inputs=pd.concat([d[['压裂井-压裂段-压窜井','验证平台组','压窜分类_目标5类']].reset_index(drop=True),x.reset_index(drop=True)],axis=1);data['质控后模型输入']=pack(inputs)
notes=[['研究目标','利用历史地质工程参数预测五类压力曲线形态；当前不是经过独立验证的损害风险模型。'],['样本口径','1630个一致五类组合，排除312个形态边界与302个少于5项有效输入的组合，主实验1016个组合、560个井段、23个平台、21个连通平台组。'],['验证设计','外层5折、内层3折，跨平台邻井相关平台合并。外层训练测试无共享井；另报按井段分组的同平台迁移对照。'],['参数处理','14项候选输入；训练折内保留覆盖至少35%且有变化的列，中位数插补及标准化。地质解释覆盖不足95%和冲突字段置空；未执行的零工程量不作有效输入。'],['时间口径','输入含历史实际施工参数，不保证在施工前可获得，不能称为已完成前瞻性预测。'],['分类字典','既有分类字典曾使用全量曲线开发。本次验证评价给定字典后的监督学习，尚无完全独立的平台外部验证。'],['概率与区间','概率未校准；95%区间按21个平台组重采样1000次，条件于已拟合的折外预测，未包含类型字典和重新训练不确定性。'],['SHAP','解释外层测试样本，TreeSHAP tree_path_dependent，类别原始得分尺度；不是概率百分点，不是因果效应。灰点代表原始输入缺失。'],['天然裂缝','仅泸201H5具备阶段性定量结果，未纳入跨平台主模型。地震属性不是微地震事件验证。'],['来源','模型输入来自20260916版压裂井段邻井地质工程因素与五类形态数据集；曲线标签来自V3参考形态约束分类。'],['局限','XGBoost M2召回率为0；五类区分能力仍弱。不能依据SHAP排行直接制定排量或加砂处方。'],['垂向应力质控','阳101H35-1设计文档表6第三行垂向应力15.4MPa与深度及相邻层显著不一致，暂不使用垂向应力候选列；源数据未修改。'],['统计表口径','数据为本次运行的静态统计快照。Excel中修改输入不会重新训练模型。']]
data['方法与来源']=pack(pd.DataFrame(notes,columns=['项目','说明']))
(B/'model_workbook.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')
(B/'reloaded_model_checks.json').write_text(json.dumps({'all_fold_reloaded_predictions_match':True,'final_models':['Logistic','RandomForest','XGBoost'],'importable_transformer':'train_models.FoldColumns'},indent=2),encoding='utf8')
print(pd.DataFrame(rows).groupby(['feature','class']).rho.agg(['count','median','min','max']).to_string())
