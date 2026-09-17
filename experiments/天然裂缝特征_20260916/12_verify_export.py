from pathlib import Path
import json,sys,openpyxl,math
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916');O=Path('F:/论文库/IGS/实验/结论/天然裂缝特征_20260916');p=O/'天然裂缝特征_井段邻井汇总_阶段版.xlsx'
d=json.loads((B/'workbook_data.json').read_text(encoding='utf-8'));w=openpyxl.load_workbook(p,read_only=True,data_only=True);wf=openpyxl.load_workbook(p,read_only=True,data_only=False);mismatches=[];density_checks=0;unavailable=0
for name,item in d.items():
 s=w[name];assert s['A2'].value;rows=list(s.iter_rows(min_row=6,max_row=5+len(item['rows']),max_col=len(item['columns']),values_only=True));assert len(rows)==len(item['rows'])
 for i,(row,exp) in enumerate(zip(rows,item['rows']),6):
  for j,(a,b) in enumerate(zip(row,exp)):
   if name=='井段邻井特征' and j==6:
    if exp[5] is not None:
     assert isinstance(a,(int,float)) and abs(a-exp[5]/exp[10]*1000)<1e-8;density_checks+=1
    else:assert a in [None,''];unavailable+=1
    continue
   if b is None or b=='':ok=a is None or a==''
   elif isinstance(b,(int,float)) and not isinstance(b,bool):ok=isinstance(a,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
   else:ok=a==b
   if not ok:mismatches.append([name,i,j,a,b])
assert not mismatches,str(mismatches[:5]);f=[r[0] for r in wf['井段邻井特征'].iter_rows(min_row=6,max_row=17243,min_col=7,max_col=7,values_only=True)];assert len(f)==17238 and all(isinstance(x,str) and x.startswith('=IF(ISNUMBER(') for x in f)
checks={'saved_workbook':str(p),'bytes':p.stat().st_size,'sheets':w.sheetnames,'all_exported_values_match':True,'numeric_P21_cache_checks':density_checks,'unavailable_P21_cells_preserved':unavailable,'P21_formulas_present':len(f),'title_cells_present':True};w.close();wf.close();(B/'saved_workbook_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(checks,ensure_ascii=False))
