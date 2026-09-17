from pathlib import Path
import openpyxl,json,re
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');root=Path('E:/压裂窜扰项目/02 规范化')
files=[p for p in root.rglob('*.xlsx') if '地质工程因素' in str(p) and any(s in p.name for s in ['井斜','射孔','分段','井位'])]
seen=set();out=[]
for p in files:
 typ='survey' if '井斜' in p.name else 'stage' if '分段' in p.name else 'wellhead' if '井位' in p.name else 'perforation'
 key=(p.parts[-3],typ)
 if key in seen:continue
 seen.add(key)
 try:
  w=openpyxl.load_workbook(p,read_only=True,data_only=True);s=w.worksheets[0];rows=list(s.iter_rows(max_row=8,values_only=True));w.close()
  out.append(dict(path=str(p),kind=typ,rows=rows))
 except Exception as e:out.append(dict(path=str(p),error=str(e)))
(B/'geology_headers.json').write_text(json.dumps(out,ensure_ascii=False,default=str,indent=2),encoding='utf-8')
print(json.dumps(out[:10],ensure_ascii=False,default=str))
