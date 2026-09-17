from pathlib import Path
import zipfile,lxml.etree as ET,json,re,sys
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');out=[]
for p in Path('E:/压裂窜扰项目/02 规范化').rglob('*.docx'):
 if '地质工程因素' not in str(p):continue
 try:
  with zipfile.ZipFile(p) as z:root=ET.fromstring(z.read('word/document.xml'))
  ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
  pars=[''.join(t.itertext()) for t in root.findall('.//w:t',ns)]
  tables=[]
  for tab in root.findall('.//w:tbl',ns):
   rows=[[''.join(cell.itertext()) for cell in row.findall('w:tc',ns)] for row in tab.findall('w:tr',ns)]
   tx=str(rows)
   if any(k in tx for k in ['坐标','井口','应力方位','最大水平主应力方向']):tables.append(rows)
  out.append(dict(path=str(p),tables=tables))
 except Exception as e:out.append(dict(path=str(p),error=str(e)))
(B/'docx_spatial_tables.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for r in out[:8]:print(json.dumps(r,ensure_ascii=False)[:3800])
print('docs',len(out),'tables',sum(len(r.get('tables',[])) for r in out))
