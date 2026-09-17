from pathlib import Path
import json, re, hashlib, sys, platform, warnings
import numpy as np, pandas as pd, joblib
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, log_loss, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import shap, sklearn, xgboost
warnings.filterwarnings('ignore',category=UserWarning)
B=Path('F:/论文库/IGS/实验/过程/分类预测与SHAP_20260916'); O=Path('F:/论文库/IGS/实验/结论/分类预测与SHAP_20260916')
LABELS=['涨幅微弱型','快速稳定型','渐进上升型','稳定缓升型','先降后升型']
GEO=['孔隙度_pct','含水饱和度_pct','总有机碳_pct','杨氏模量_GPa','泊松比','最小主应力_MPa','水平应力差_MPa','破裂压力_MPa']
ENG=['段长_m','簇数','加砂强度_t_m','用液强度_m3_m','排量_m3_min']
FEATURES=['重算井距_段中点平面_m']+ENG+GEO

def clean(d):
    x=d[FEATURES].apply(pd.to_numeric,errors='coerce').copy()
    # Quality screening is label-independent and uses source audits only.
    for c in GEO:
        coverage=c+'_覆盖率' if c+'_覆盖率' in d else '解释覆盖率'
        x.loc[pd.to_numeric(d[coverage],errors='coerce').lt(.95),c]=np.nan
    for c in ENG:
        x.loc[d['工程冲突字段'].fillna('').str.contains(c,regex=False),c]=np.nan
        x.loc[x[c]<=0,c]=np.nan
    x.loc[d['分段冲突'].fillna(False).astype(bool),GEO+['段长_m']]=np.nan
    x.loc[d['井距说明'].fillna('').str.contains('待核查'),FEATURES[0]]=np.nan
    return x.replace([np.inf,-np.inf],np.nan)

class FoldColumns(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        a=np.asarray(X,float); self.keep_=(np.isfinite(a).mean(0)>=.35)&(np.nanstd(a,axis=0)>1e-10)
        if not self.keep_.any(): raise ValueError('No eligible training features')
        return self
    def transform(self,X):return np.asarray(X,float)[:,self.keep_]

def make(name,params):
    if name=='Dummy':m=DummyClassifier(strategy='prior')
    elif name=='Logistic':m=LogisticRegression(max_iter=2000,**params)
    elif name=='RandomForest':m=RandomForestClassifier(n_estimators=240,n_jobs=4,random_state=20260916,**params)
    else:m=XGBClassifier(objective='multi:softprob',num_class=5,tree_method='hist',n_jobs=4,random_state=20260916,eval_metric='mlogloss',**params)
    return Pipeline([('columns',FoldColumns()),('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('model',m)])

def fit_model(name,params,x,y,weighted):
    m=make(name,params);kw={}
    if weighted:kw['model__sample_weight']=compute_sample_weight('balanced',y)
    m.fit(x,y,**kw);return m

def metrics(y,p):
    q=p.argmax(1);one=np.eye(5)[y]
    return {'macro_F1':f1_score(y,q,labels=range(5),average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,q),'accuracy':accuracy_score(y,q),'log_loss':log_loss(y,p,labels=range(5)),'Brier_multiclass':np.mean(np.sum((one-p)**2,axis=1)), 'macro_AUROC':roc_auc_score(y,p,multi_class='ovr',labels=range(5)) if len(np.unique(y))==5 else np.nan}

CAND={'Dummy':[({},False)],'Logistic':[({'C':c},w) for c,w in [(0.1,False),(1,True),(10,True)]], 'RandomForest':[({'max_depth':d,'min_samples_leaf':l,'max_features':'sqrt'},w) for d,l,w in [(5,8,True),(None,4,True),(8,6,False)]], 'XGBoost':[({'n_estimators':n,'max_depth':d,'learning_rate':.04,'min_child_weight':5,'subsample':.85,'colsample_bytree':.85,'reg_lambda':5},w) for n,d,w in [(180,2,True),(260,3,True),(180,2,False)]]}

def connected_groups(raw):
    parent={}
    def root(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=root(parent[x])
        return parent[x]
    for a,b in zip(raw['平台'],raw['压窜井'].str.rsplit('-',n=1).str[0]):
        ra,rb=root(a),root(b)
        if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
    return {g:root(g) for g in parent}

def main():
    src=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916/pair_factors.pkl');raw=pd.read_pickle(src)
    all5=raw[raw['压窜分类_目标5类'].isin(LABELS)].copy(); Xall=clean(all5)
    enough=Xall.notna().sum(1)>=5
    strict=all5['分类说明'].fillna('').eq('')
    d=all5.loc[strict&enough].reset_index(names='source_index');X=clean(d); y=d['压窜分类_目标5类'].map(dict(zip(LABELS,range(5)))).to_numpy();component=connected_groups(raw[raw['关联曲线数'].gt(0)]); d['验证平台组']=d['平台'].map(component).fillna(d['平台']); groups=d['验证平台组'].to_numpy()
    d.to_pickle(B/'model_cohort.pkl');X.to_pickle(B/'model_inputs.pkl')
    audit={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'all_pairs':len(raw),'five_labels':len(all5),'label_boundary':int((~strict).sum()),'strict_before_feature_screen':int(strict.sum()),'strict_insufficient_features':int((strict&~enough).sum()),'n':len(d),'platforms':int(d['平台'].nunique()),'connected_platform_groups':int(d['验证平台组'].nunique()),'connected_mapping':component,'stages':int(d['井段键'].nunique()),'classes':d['压窜分类_目标5类'].value_counts().to_dict(),'features':FEATURES,'versions':{'python':sys.version,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'shap':shap.__version__},'protocol':'Frozen existing morphology taxonomy; nested grouped 5 outer / 3 inner by connected source/monitor platform components. Vertical stress excluded because at least one source has unresolved column ambiguity. No independent taxonomy discovery split; retrospective conditional evaluation. Historical actual engineering values, not validated pre-job forecast.'}
    (B/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(audit,ensure_ascii=False),flush=True)
    # Freeze all outer partitions before any fitted model.
    outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=20260916).split(X,y,groups))
    folds=np.full(len(d),-1); checks=[]
    for k,(tr,te) in enumerate(outer):
        folds[te]=k;assert set(groups[tr]).isdisjoint(groups[te]);assert set(d.iloc[tr]['压裂井']).union(d.iloc[tr]['压窜井']).isdisjoint(set(d.iloc[te]['压裂井']).union(d.iloc[te]['压窜井']));assert set(d.iloc[tr]['井段键']).isdisjoint(d.iloc[te]['井段键']);assert len(np.unique(y[tr]))==5
        checks.append({'fold':k,'train_n':len(tr),'test_n':len(te),'train_platforms':sorted(set(groups[tr])),'test_platforms':sorted(set(groups[te])),'test_class_counts':np.bincount(y[te],minlength=5).tolist()})
    (B/'frozen_splits.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    oof={n:np.zeros((len(d),5)) for n in CAND};rows=[];search=[];select_oof=np.zeros((len(d),5));selected=[]
    sv=np.zeros((len(d),len(FEATURES),5));base=np.zeros((len(d),5));add_errors=[]
    for k,(tr,te) in enumerate(outer):
        inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=120+k).split(X.iloc[tr],y[tr],groups[tr])); scores={}
        for name,opts in CAND.items():
            candidates=[]
            for oi,(p,w) in enumerate(opts):
                vals=[]
                for itr,iva in inner:
                    assert set(groups[tr][itr]).isdisjoint(groups[tr][iva])
                    m=fit_model(name,p,X.iloc[tr].iloc[itr],y[tr][itr],w)
                    vals.append(f1_score(y[tr][iva],m.predict(X.iloc[tr].iloc[iva]),labels=range(5),average='macro',zero_division=0))
                score=float(np.mean(vals));candidates.append(score);search.append({'outer_fold':k,'model':name,'candidate':oi,'inner_macro_F1':score,'params':json.dumps(p),'weighted':w})
            best=int(np.argmax(candidates));p,w=opts[best];m=fit_model(name,p,X.iloc[tr],y[tr],w);proba=m.predict_proba(X.iloc[te]);oof[name][te]=proba
            row={'fold':k,'model':name,'candidate':best,'inner_score':candidates[best],**metrics(y[te],proba)};rows.append(row);scores[name]=(candidates[best],proba)
            joblib.dump(m,B/f'fold{k}_{name}.joblib')
            if name=='XGBoost':
                z=m[:-1].transform(X.iloc[te]); ex=shap.TreeExplainer(m[-1],feature_perturbation='tree_path_dependent',model_output='raw');v=ex.shap_values(z);v=np.asarray(v);assert v.shape==(len(te),z.shape[1],5),v.shape
                keep=m['columns'].keep_;sv[np.ix_(te,np.where(keep)[0],range(5))]=v;base[te]=np.asarray(ex.expected_value)
                margin=m[-1].predict(z,output_margin=True);err=float(np.max(np.abs(v.sum(1)+np.asarray(ex.expected_value)-margin)));add_errors.append(err);assert err<1e-4,err
            print('OUTER',k,name,'F1',round(row['macro_F1'],3),'inner',round(row['inner_score'],3),flush=True)
        chosen=max(scores,key=lambda n:scores[n][0]);select_oof[te]=scores[chosen][1];selected.append(chosen)
    oof['NestedSelection']=select_oof
    np.savez_compressed(B/'oof_arrays.npz',y=y,fold=folds,shap=sv,base=base,**oof)
    pd.DataFrame(rows).to_csv(O/'02_各折模型表现.csv',index=False,encoding='utf-8-sig');pd.DataFrame(search).to_csv(B/'hyperparameter_search.csv',index=False,encoding='utf-8-sig')
    # Platform bootstrap keeps correlated stage-neighbor rows together. CI conditional on fitted OOF models.
    rng=np.random.default_rng(101);unique=np.unique(groups);boot=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(1000)]
    summary=[]
    for n,p in oof.items():
        bs=[f1_score(y[b],p[b].argmax(1),labels=range(5),average='macro',zero_division=0) for b in boot]
        summary.append({'model':n,**metrics(y,p),'F1_CI_low':np.quantile(bs,.025),'F1_CI_high':np.quantile(bs,.975)})
    pd.DataFrame(summary).to_csv(O/'01_模型总体表现.csv',index=False,encoding='utf-8-sig')
    pred=d[['压裂井-压裂段-压窜井','平台','井段键','压窜分类_目标5类','曲线样本ID','验证平台组']].copy();pred['outer_fold']=folds
    for n,p in oof.items():
        pred[n+'_预测类型']=[LABELS[i] for i in p.argmax(1)]
        for j in range(5):pred[n+'_P_M'+str(j+1)]=p[:,j]
    pred.to_csv(O/'03_逐样本折外预测.csv',index=False,encoding='utf-8-sig')
    # Fixed-parameter sensitivity studies are deliberately not a second search for a better headline score.
    fixed=CAND['XGBoost'][0];sensitivity=[];sensitivity_arrays={}
    variants={'仅工程参数':ENG,'仅地质参数':GEO,'不含井距':ENG+GEO,'全特征固定参数':FEATURES}
    for name,cols in variants.items():
        p=np.zeros((len(d),5))
        for k,(tr,te) in enumerate(outer):p[te]=fit_model('XGBoost',*fixed[:1],X[cols].iloc[tr],y[tr],fixed[1]).predict_proba(X[cols].iloc[te])
        sensitivity.append({'experiment':name,'n':len(d),'groups':len(unique),**metrics(y,p)});sensitivity_arrays[name]=p
    for name,sub,xx,gg in [('加入形态边界',all5.loc[enough].reset_index(drop=True),Xall.loc[enough].reset_index(drop=True),'平台'),('按井段分组',d,X,'井段键')]:
        yy=sub['压窜分类_目标5类'].map(dict(zip(LABELS,range(5)))).to_numpy();gr=(sub['平台'].map(component).fillna(sub['平台']) if gg=='平台' else sub[gg]).to_numpy();p=np.zeros((len(sub),5))
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=20260916).split(xx,yy,gr):p[te]=fit_model('XGBoost',fixed[0],xx.iloc[tr],yy[tr],fixed[1]).predict_proba(xx.iloc[te])
        sensitivity.append({'experiment':name,'n':len(sub),'groups':len(np.unique(gr)),**metrics(yy,p)})
    pd.DataFrame(sensitivity).to_csv(O/'04_敏感性与特征组对照.csv',index=False,encoding='utf-8-sig');np.savez_compressed(B/'ablation_oof.npz',**sensitivity_arrays)
    # SHAP on outer test samples, importance equal-weighted by platform as robustness check.
    imp=np.abs(sv).mean((0,2));eq=np.mean([np.abs(sv[groups==g]).mean((0,2)) for g in unique],axis=0)
    importance=pd.DataFrame({'feature':FEATURES,'mean_abs_SHAP_margin':imp,'platform_equal_mean_abs_SHAP':eq,'missing_fraction':X.isna().mean().to_numpy()})
    for j,l in enumerate(LABELS):importance['M'+str(j+1)]=np.abs(sv[:,:,j]).mean(0)
    importance.sort_values('mean_abs_SHAP_margin',ascending=False).to_csv(O/'05_SHAP重要性.csv',index=False,encoding='utf-8-sig')
    audit.update({'outer_selected_models':selected,'shap_max_additivity_error':max(add_errors),'outer_group_overlap':False})
    (B/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    # Final candidate fit for later inference; final training score is never reported as validation.
    s=pd.DataFrame(search);best=int(s[s.model=='XGBoost'].groupby('candidate').inner_macro_F1.mean().idxmax());p,w=CAND['XGBoost'][best];final=fit_model('XGBoost',p,X,y,w);joblib.dump(final,B/'XGBoost_final_pipeline.joblib');final[-1].save_model(B/'XGBoost_final.ubj')
    (B/'final_model_metadata.json').write_text(json.dumps({'candidate':best,'params':p,'weighted':w,'features':FEATURES,'classes':LABELS,'scope':'Retrospective exploratory morphology classification; probabilities uncalibrated; not operational risk prediction'},ensure_ascii=False,indent=2),encoding='utf8')
    print('DONE',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
