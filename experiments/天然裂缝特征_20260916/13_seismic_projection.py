from pathlib import Path
import struct,re,json,sys,hashlib,time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');D=B/'地震叠合图';D.mkdir(exist_ok=True)
inv=json.loads((B/'sgy_inventory.json').read_text(encoding='utf-8'))
def ibm(words):
 u=words.astype(np.uint32);v=np.ldexp((u&0xffffff).astype(np.float32),4*((u>>24)&127).astype(np.int32)-280);return np.where((u>>31)!=0,-v,v)
assert np.allclose(ibm(np.array([0,0x41100000,0xc1200000,0x40800000],dtype=np.uint32)),[0,1,-2,.5])
for meta in inv:
 p=Path(meta['path'])
 if 'ant' not in p.name.lower():continue
 out=D/(p.parent.name+'.npz')
 if out.exists():print('cached',p.parent.name,flush=True);continue
 start=time.time();n=meta['n'];ns=meta['ns'];stride=240+4*ns
 text=(B/(p.stem+'_header.txt')).read_text(encoding='utf-8')
 def position(pattern,default):
  m=re.search(pattern,text);return int(m[1])-1 if m else default
 xoff=position(r'CDP X\s*:\s*position\s*=\s*(\d+)',72);yoff=position(r'CDP Y\s*:\s*position\s*=\s*(\d+)',76)
 fields={k:np.empty(n,dtype=dtype) for k,dtype in [('x','f8'),('y','f8'),('il','i4'),('xl','i4'),('positive_mean','f4'),('maximum','f4'),('positive_fraction','f4')]};sha=hashlib.sha256();bad=0;rawmin=np.inf;rawmax=-np.inf;done=0;last=0;sampling_checks=[]
 with p.open('rb') as f:
  header=f.read(3600);sha.update(header);assert struct.unpack_from('>h',header,3504)[0]==0 and struct.unpack_from('>H',header,3224)[0]==1
  while done<n:
   count=min(2048,n-done);buf=f.read(count*stride);assert len(buf)==count*stride;sha.update(buf)
   def col(off,dt):return np.ndarray((count,),dtype=dt,buffer=buf,offset=off,strides=(stride,))
   assert np.all(col(114,'>u2')==ns) and np.all(col(116,'>u2')==meta['dt'])
   scalar=col(70,'>i2').astype(float);factor=np.where(scalar>0,scalar,1/np.maximum(abs(scalar),1));sl=slice(done,done+count)
   fields['x'][sl]=col(xoff,'>i4')*factor;fields['y'][sl]=col(yoff,'>i4')*factor;fields['il'][sl]=col(8,'>i4');fields['xl'][sl]=col(20,'>i4')
   words=np.ndarray((count,ns),dtype='>u4',buffer=buf,offset=240,strides=(stride,4));values=ibm(words);bad+=int((~np.isfinite(values)).sum());assert np.all(np.isfinite(values));rawmin=min(rawmin,float(values.min()));rawmax=max(rawmax,float(values.max()))
   fields['maximum'][sl]=values.max(axis=1);fields['positive_mean'][sl]=np.maximum(values,0).mean(axis=1,dtype=np.float64);fields['positive_fraction'][sl]=(values>0).mean(axis=1)
   if done==0:
    for a,b in [(0,0),(0,ns//2),(count-1,ns-1)]:
     word=int(words[a,b]);v=(-1 if word&0x80000000 else 1)*(word&0xffffff)/(16**6)*16.0**(((word>>24)&127)-64);assert np.isclose(values[a,b],v);sampling_checks.append(v)
   done+=count
   if time.time()-last>25:print(p.parent.name,round(done/n*100,1),'%',flush=True);last=time.time()
  assert not f.read(1)
 np.savez_compressed(out,**fields)
 report=dict(meta,x_byte=xoff+1,y_byte=yoff+1,sha256=sha.hexdigest(),raw_value_range=[rawmin,rawmax],positive_mean_quantiles=np.quantile(fields['positive_mean'],[0,.5,.9,.95,.99,1]).tolist(),maximum_quantiles=np.quantile(fields['maximum'],[0,.5,.9,.99,1]).tolist(),nonfinite_samples=bad,scalar_ibm_reference_checks=sampling_checks,elapsed_s=time.time()-start,projection='mean(max(attribute,0)) over the complete provided time interval; no horizon alignment')
 (D/(p.parent.name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('DONE',p.parent.name,report['positive_mean_quantiles'],flush=True)
