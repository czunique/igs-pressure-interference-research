import subprocess,re,json,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/时序聚类_V2_20260915');O=Path('F:/论文库/IGS/实验/结论/时序聚类_V2_20260915');url=(O/'全量曲线查询.html').as_uri();chrome='C:/Program Files/Google/Chrome/Application/chrome.exe';out=[]
for name,fragment,expected in [('default','',3180),('class2','#class=2',598),('empty','#q=XYZ_NO_SUCH_WELL',0)]:
 args=[chrome,'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--allow-file-access-from-files','--user-data-dir='+str(B/'browser_qa_profile'),'--window-size=1440,1000','--virtual-time-budget=1800','--dump-dom']
 if name=='default':args+=['--screenshot='+str(B/'curve_catalogue_qa.png')]
 p=subprocess.run(args+[url+fragment],capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW);doc=p.stdout.decode('utf-8',errors='replace');m=re.search(r'<p id="count">([^<]+)',doc);message=m.group(1) if m else '';valid=bool(m and f'匹配 {expected} / 3180 条'==message);out.append(dict(test=name,expected=expected,observed=message,passed=valid,returncode=p.returncode));print(out[-1],flush=True)
(B/'browser_qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');assert all(z['passed'] for z in out)
