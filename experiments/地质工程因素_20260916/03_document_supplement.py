from pathlib import Path
import sys,json,pickle,re,zipfile
from lxml import etree as ET
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');O=B.parent/'天然裂缝特征_20260916'
rs=pickle.load(open(B/'source_tables.pkl','rb'));rs=[r for r in rs if not r.get('document')]
def wid(s):
 m=re.search(r'([泸阳自]\d+H\d+)[-－—](\d+)',str(s),re.I);return (m[1].upper(),int(m[2])) if m else None
def num(x):
 try:return float(str(x).strip())
 except:return float('nan')
have={wid(r['file']) for r in rs if r['kind']=='geology'};extras=[];audit=[]
for doc in json.loads((O/'document_index.json').read_text(encoding='utf-8')):
 if 'cache' not in doc:continue
 p=Path(doc['path']);w=wid(p.name)
 if not w or (w in have and w not in [('泸203H6',2),('泸203H6',3),('泸203H9',6)]):continue
 tables=[]
 if p.suffix.lower()=='.docx':
  try:
   with zipfile.ZipFile(p) as z:root=ET.fromstring(z.read('word/document.xml'))
   ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
   for ti,t in enumerate(root.findall('.//w:tbl',ns)):
    rows=[]
    for tr in t.findall('w:tr',ns):
     row=[]
     for tc in tr.findall('w:tc',ns):
      v=''.join(tc.xpath('.//w:t/text()',namespaces=ns));row.append(v);span=tc.find('w:tcPr/w:gridSpan',ns)
      if span is not None:row.extend(['']*(int(span.get('{'+ns['w']+'}val'))-1))
     rows.append(row)
    prev=t.getprevious();caption=''.join(prev.xpath('.//w:t/text()',namespaces=ns)) if prev is not None else ''
    tables.append((str(ti+1),rows,caption))
  except Exception as e:audit.append({'file':str(p),'error':str(e)})
 else:
  text=(O/'document_text'/doc['cache']).read_text(encoding='utf-8');rows=[a.strip().split('\x07') for a in text.split('\x07\x07')]
  for i,row in enumerate(rows):
   if '顶深' in str(row) and '底深' in str(row) and any('孔隙度' in c or 'TOC' in c for c in row):
    if len(row)<7:continue
    caption=row[0];row=row.copy();row[0]=row[0].split('\n')[-1]
    following=[]
    for rr in rows[i+1:i+200]:
     if len(rr)>=5 and 1000<num(rr[2])<num(rr[3])<12000:following.append(rr)
     elif following:break
    tables.append((f'文本表{i+1}',[row]+following,caption))
 for ti,rows,caption in tables:
  if not rows or len(rows[0])<7:continue
  h=rows[0]
  if not ('顶深' in str(h[2]) and '底深' in str(h[3]) and any('孔隙度' in str(c) or 'TOC' in str(c) for c in h)):continue
  other=wid(caption)
  if other and other!=w:audit.append({'file':str(p),'table':ti,'reason':'表题为其他井，未采用'});continue
  valid=[row for row in rows[1:] if len(row)>=5 and 1000<num(row[2])<num(row[3])<12000]
  if not valid:continue
  h=[str(c).replace('TOC','总有机碳').replace('含气量','总含气量') if '总含气量' not in str(c) else str(c) for c in h]
  arr=[h]+valid;arr=[tuple((list(row)+[None]*40)[:40]) for row in arr]
  extras.append(dict(file=str(p),kind='geology',sheet='文档表'+ti,rows=arr,document=True,caption=caption))
  audit.append({'file':str(p),'table':ti,'rows':len(valid),'caption':caption})
rs.extend(extras);pickle.dump(rs,open(B/'source_tables.pkl','wb'));(B/'document_supplement_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print('DOCUMENT_TABLES',len(extras),'WELLS',len({wid(r['file']) for r in extras}))
