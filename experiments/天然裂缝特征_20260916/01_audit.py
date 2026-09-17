from pathlib import Path
import struct,json,numpy as np
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916')
root=Path('E:/压裂窜扰项目/00 原始数据/压裂所数据')
rows=[]
for p in root.rglob('*.sgy'):
 with p.open('rb') as f: h=f.read(3840)
 text=max([h[:3200].decode(c,errors='replace') for c in ['ascii','cp500']],key=lambda s:sum(x.isascii() and x.isprintable() for x in s))
 ns=struct.unpack_from('>H',h,3220)[0];dt=struct.unpack_from('>H',h,3216)[0];fmt=struct.unpack_from('>H',h,3224)[0]
 stride=240+ns*4;n=(p.stat().st_size-3600)//stride
 m=np.memmap(p,mode='r',dtype='u1'); col=lambda off,ty:np.ndarray((n,),dtype=ty,buffer=m,offset=3600+off,strides=(stride,))
 scalar=col(70,'>i2').astype(float);fac=np.where(scalar>0,scalar,1/np.maximum(abs(scalar),1));x=col(72,'>i4')*fac;y=col(76,'>i4')*fac
 r=dict(path=str(p),size=p.stat().st_size,ns=ns,dt=dt,fmt=fmt,n=n,remainder=(p.stat().st_size-3600)%stride,x=[float(x.min()),float(x.max())],y=[float(y.min()),float(y.max())],start=np.unique(col(108,'>i2')).tolist(),scalar=np.unique(scalar).tolist(),units=np.unique(col(88,'>i2')).tolist())
 rows.append(r);(B/(p.stem+'_header.txt')).write_text('\n'.join(text[i:i+80] for i in range(0,3200,80)),encoding='utf-8');print(json.dumps(r,ensure_ascii=False))
(B/'sgy_inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
files=[str(p) for p in Path('E:/压裂窜扰项目/02 规范化').rglob('*') if p.is_file() and '地质工程因素' in str(p)]
(B/'geology_inventory.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
print('geology files',len(files))
