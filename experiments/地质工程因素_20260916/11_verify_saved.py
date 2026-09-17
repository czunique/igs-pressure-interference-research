from pathlib import Path
import sys,json,math,openpyxl
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');P=Path('F:/论文库/IGS/实验/结论/地质工程因素_20260916/压裂井段邻井_地质工程因素与五类形态.xlsx');data=json.loads((B/'workbook_data.json').read_text(encoding='utf-8'));w=openpyxl.load_workbook(P,read_only=True,data_only=True);checked=0
assert w.sheetnames==list(data)
for name,d in data.items():
 s=w[name];rows=iter(s.iter_rows(max_row=len(d['rows'])+1,max_col=len(d['columns']),values_only=True));assert list(next(rows))==d['columns'],name
 for i,(row,expected) in enumerate(zip(rows,d['rows']),2):
  for j,(v,e) in enumerate(zip(row,expected)):
   if name=='加权计算示例' and j in [6,7]:continue
   if e is None or e=='':assert v is None or v=='',(name,i,j,v,e)
   elif isinstance(e,(int,float)) and not isinstance(e,bool):assert isinstance(v,(int,float)) and math.isclose(v,e,rel_tol=1e-10,abs_tol=1e-8),(name,i,j,v,e)
   else:assert v==e,(name,i,j,str(v)[:80],str(e)[:80])
   checked+=1
 print('CHECKED',name,len(d['rows']),flush=True)
ex=data['加权计算示例']['rows'];n=len(ex);calc=w['加权计算示例'].cell(n+4,7).value;target=json.loads((B/'data_checks.json').read_text(encoding='utf-8'))['example_young_GPa'];assert abs(calc-target)<1e-8;assert w['加权计算示例'].cell(n+5,7).value==1;w.close()
f=openpyxl.load_workbook(P,read_only=False,data_only=False);assert all(len(s.tables)==1 for s in f);assert f['井段邻井因素'].freeze_panes=='D2';assert f['井段邻井因素'].cell(1,3).value=='压窜分类_目标5类';assert f['加权计算示例'].cell(n+4,7).data_type=='f';f.close()
result={'passed':True,'checked_cells':checked,'sheet_count':len(data),'file_bytes':P.stat().st_size,'example_formula_cache':calc,'identity_and_blank_checks':True,'tables_and_frozen_identifier_columns':True}
(B/'saved_workbook_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
