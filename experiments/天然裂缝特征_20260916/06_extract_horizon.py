from pathlib import Path
import ast,struct,json,hashlib,sys
import numpy as np,pandas as pd
from collections import deque
from scipy.spatial import cKDTree
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');OUT=B/'seismic';OUT.mkdir(exist_ok=True)
REF=Path('E:/code/天然裂缝和压窜积液量的关系/L201H5_code');SRC=Path('E:/压裂窜扰项目/00 原始数据/压裂所数据/地震/泸201H5')
def load_functions(p,names,namespace):
 tree=ast.parse(p.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
load_functions(REF/'analyze_sgy.py',['ibm_to_float','stats','read_sgy'],globals())
h=np.loadtxt(SRC/'LU205H5-0603-O3w.hrzdat',comments='#');arrays={};checks={};meta={}
for name in ['MCANT','max_curvature']:
 cube,il,xl,X,Y,t,m=read_sgy(SRC/f'LU205H5_{name}.sgy');ii=np.searchsorted(il,h[:,0]);jj=np.searchsorted(xl,h[:,1]);assert np.allclose(X[ii,jj],h[:,2]) and np.allclose(Y[ii,jj],h[:,3]);H=np.empty(X.shape);H[ii,jj]=h[:,4];pos=(H-t[0])/(t[1]-t[0]);assert pos.min()>=0 and pos.max()<len(t)-1;r,c=np.indices(X.shape)
 if name=='MCANT':a=cube[r,c,np.rint(pos).astype(int)];key='MCANT_nearest_0ms'
 else:lo=np.floor(pos).astype(int);f=pos-lo;a=cube[r,c,lo]*(1-f)+cube[r,c,lo+1]*f;key='max_curvature_offset_0ms'
 arrays[key]=a;meta[name]=m;old=np.load(REF/'sgy_analysis/extracted_arrays.npz')[key];checks[name+'_max_abs_difference_vs_reference']=float(np.max(abs(old-a)));assert np.allclose(old,a);del cube
np.savez_compressed(OUT/'horizon_arrays.npz',X=X,Y=Y,H=H,**arrays)
(B/'sgy_reproduction_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');(OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
load_functions(REF.parent/'PPT_天然裂缝定量表征_0915/analyze_fractures.py',['prune','skeleton','clip_segment'],globals())
def extract_edges(A,threshold,guard=2):
 mask=A>=threshold-1e-6
 if guard:mask[:guard]=False;mask[-guard:]=False;mask[:,:guard]=False;mask[:,-guard:]=False
 S=skeleton(prune(mask,4));edges=[]
 for r,c in zip(*np.where(S)):
  for dr,dc in [(0,1),(1,-1),(1,0),(1,1)]:
   nr,nc=r+dr,c+dc
   if not(0<=nr<S.shape[0] and 0<=nc<S.shape[1] and S[nr,nc]):continue
   if dr and dc and (S[r,nc] or S[nr,c]):continue
   edges.append([X[r,c],Y[r,c],X[nr,nc],Y[nr,nc]])
 return np.asarray(edges).reshape(-1,4)
np.save(OUT/'edges_reference_no_guard.npy',extract_edges(arrays['MCANT_nearest_0ms'],.2,0))
for threshold in [.1,.2,.3]:
 e=extract_edges(arrays['MCANT_nearest_0ms'],threshold);np.save(OUT/f'edges_{threshold:g}.npy',e);print('threshold',threshold,'edges',len(e))
print(checks)
