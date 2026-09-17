from pathlib import Path
import olefile,struct,re,json,sys,zipfile,lxml.etree as ET
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');D=B/'document_text';D.mkdir(exist_ok=True)
def oldtext(p):
 o=olefile.OleFileIO(p);w=o.openstream('WordDocument').read();t=o.openstream('1Table' if struct.unpack_from('<H',w,10)[0]&512 else '0Table').read();fc,sz=struct.unpack_from('<II',w,418);cl=t[fc:fc+sz];pos=0
 while pos<len(cl) and cl[pos]==1:pos+=3+struct.unpack_from('<H',cl,pos+1)[0]
 if pos>=len(cl) or cl[pos]!=2:raise ValueError('No piece table')
 l=struct.unpack_from('<I',cl,pos+1)[0];pl=cl[pos+5:pos+5+l];n=(l-4)//12;cps=struct.unpack_from('<'+'I'*(n+1),pl);out=[]
 for i in range(n):
  f=struct.unpack_from('<I',pl,4*(n+1)+8*i+2)[0];cnt=cps[i+1]-cps[i];compressed=bool(f&0x40000000);f&=0x3fffffff
  if compressed:out.append(w[f//2:f//2+cnt].decode('cp1252',errors='replace'))
  else:out.append(w[f:f+cnt*2].decode('utf-16le',errors='replace'))
 o.close();return ''.join(out)
out=[]
for p in Path('E:/压裂窜扰项目/02 规范化').rglob('*'):
 if '地质工程因素' not in str(p) or p.suffix.lower() not in ['.doc','.docx']:continue
 try:
  if not olefile.isOleFile(p) and zipfile.is_zipfile(p):
   with zipfile.ZipFile(p) as z:r=ET.fromstring(z.read('word/document.xml'))
   ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
   tx='\n'.join(''.join(x.itertext()) for x in r.findall('.//w:p',ns))
  else:tx=oldtext(p)
  name=str(len(out))+'.txt';(D/name).write_text(tx,encoding='utf-8')
  snippets=[tx[max(0,m.start()-60):m.end()+170] for m in re.finditer('井口.{0,6}坐标|纵坐标|最大水平.{0,6}方位',tx)]
  out.append(dict(path=str(p),cache=name,snippets=snippets))
 except Exception as e:out.append(dict(path=str(p),error=str(e)))
(B/'document_index.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('documents',len(out),'errors',sum('error' in r for r in out),'with_spatial_snippets',sum(bool(r.get('snippets')) for r in out))
for r in out:
 if r.get('snippets'): print(Path(r['path']).name,repr(r['snippets'][0])[:330])
