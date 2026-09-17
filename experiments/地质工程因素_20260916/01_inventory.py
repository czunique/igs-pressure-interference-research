from pathlib import Path
import sys,json,re,pickle,openpyxl
sys.stdout.reconfigure(encoding='utf-8')
B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');R=Path('E:/压裂窜扰项目/02 规范化')
records=[];audit=[]
paths=[(Path('E:/压裂窜扰项目/压窜数据统计—1.0.xlsx'),'legacy')]
for p in R.rglob('*.xlsx'):
 if p.name.startswith('~$') or 'xxx平台' in str(p):continue
 if '地质工程因素' in str(p) and any(t in p.name for t in ['解释','射孔','分段']):kind='geology' if '解释' in p.name else 'perforation' if '射孔' in p.name else 'stage'
 elif any(t in p.name for t in ['施工参数','压裂参数','平台参数表']):kind='engineering'
 else:continue
 paths.append((p,kind))
for p,kind in paths:
 try:
  w=openpyxl.load_workbook(p,read_only=True,data_only=True)
  for s in w:
   head=list(s.iter_rows(max_row=8,max_col=220 if kind=='engineering' else 40,values_only=True))
   if kind=='engineering' and not any('段' in str(row) for row in head):continue
   rows=list(s.iter_rows(max_row=min(s.max_row or 3000,3000),max_col=220 if kind=='engineering' else 40,values_only=True))
   while rows and all(c is None for c in rows[-1]):rows.pop()
   rec=dict(file=str(p),kind=kind,sheet=s.title,rows=rows);records.append(rec)
   audit.append(dict(file=str(p),kind=kind,sheet=s.title,nrows=len(rows),header=head[:4]))
  w.close()
 except Exception as e:audit.append(dict(file=str(p),kind=kind,error=str(e)))
with (B/'source_tables.pkl').open('wb') as f:pickle.dump(records,f)
(B/'source_inventory.json').write_text(json.dumps(audit,ensure_ascii=False,default=str,indent=2),encoding='utf-8')
print('TABLES',len(records),'FILES',len(paths),'KINDS',{k:sum(r['kind']==k for r in records) for k in set(r['kind'] for r in records)},'ERRORS',[r for r in audit if 'error' in r])
