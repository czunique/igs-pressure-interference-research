from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import json,re,zipfile,sys
from lxml import etree as E
sys.stdout.reconfigure(encoding='utf-8');B=Path('F:/论文库/IGS/实验/过程/地质工程因素_20260916');O=Path('F:/论文库/IGS/实验/结论/地质工程因素_20260916');manifest=json.loads((B/'parts_manifest.json').read_text(encoding='utf-8'));data=json.loads((B/'workbook_data.json').read_text(encoding='utf-8'))
X='http://schemas.openxmlformats.org/spreadsheetml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';PR='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types'
ns={'x':X};xp=lambda name:'{'+X+'}'+name
xml=lambda node:E.tostring(node,encoding='utf-8',xml_declaration=True)
assert len(manifest)==len(data)+3
assert all(sum(p['n'] for p in manifest if p['name']==name)==len(d['rows']) for name,d in data.items())
with zipfile.ZipFile(B/manifest[0]['path']) as z:
 base={name:z.read(name) for name in z.namelist()};styles=E.fromstring(base['xl/styles.xml']);workbook=E.fromstring(base['xl/workbook.xml']);wr=E.fromstring(base['xl/_rels/workbook.xml.rels']);ct=E.fromstring(base['[Content_Types].xml'])
for tag in ['numFmts','fonts','fills','borders','cellStyleXfs','cellXfs','dxfs']:
 q=styles.find(xp(tag))
 if q is None:q=E.SubElement(styles,xp(tag))
 for child in list(q):q.remove(child)
 q.set('count','0')
styles.remove(styles.find(xp('cellStyles')))
# Restore schema order after collecting style components.
style_order=['numFmts','fonts','fills','borders','cellStyleXfs','cellXfs','cellStyles','dxfs','tableStyles','colors','extLst']
numformats={};shared=E.Element(xp('sst'),nsmap={'x':X});sstmap={};sheetroots=OrderedDict();extras={};groups={};nextfmt=200;stylemaps={}
for part in manifest:
 with zipfile.ZipFile(B/part['path']) as z:
  assert z.read('xl/theme/theme1.xml')==base['xl/theme/theme1.xml']
  ss=E.fromstring(z.read('xl/styles.xml'));maps={};fmtmap={}
  for item in ss.findall('x:numFmts/x:numFmt',ns):
   code=item.get('formatCode');old=int(item.get('numFmtId'))
   if code not in numformats:
    numformats[code]=nextfmt;new=deepcopy(item);new.set('numFmtId',str(nextfmt));styles.find(xp('numFmts')).append(new);nextfmt+=1
   fmtmap[old]=numformats[code]
  for tag in ['fonts','fills','borders','cellStyleXfs','cellXfs','dxfs']:
   dest=styles.find(xp(tag));origin=ss.find(xp(tag));maps[tag]={}
   if origin is None:continue
   for i,item in enumerate(origin):
    new=deepcopy(item)
    if tag in ['cellStyleXfs','cellXfs']:
     for attr,kind in [('fontId','fonts'),('fillId','fills'),('borderId','borders'),('xfId','cellStyleXfs')]:
      if attr in new.attrib:new.set(attr,str(maps[kind].get(int(new.get(attr)),int(new.get(attr)))))
     if 'numFmtId' in new.attrib:new.set('numFmtId',str(fmtmap.get(int(new.get('numFmtId')),int(new.get('numFmtId')))))
    for fmt in new.findall('.//x:numFmt',ns):fmt.set('numFmtId',str(fmtmap.get(int(fmt.get('numFmtId')),int(fmt.get('numFmtId')))))
    maps[tag][i]=len(dest);dest.append(new)
  if styles.find(xp('cellStyles')) is None:
   css=deepcopy(ss.find(xp('cellStyles')))
   if css is not None:styles.append(css)
  indices={}
  sourceS=E.fromstring(z.read('xl/sharedStrings.xml'))
  for i,item in enumerate(sourceS):
   key=E.tostring(item)
   if key not in sstmap:sstmap[key]=len(shared);shared.append(deepcopy(item))
   indices[i]=sstmap[key]
  sheet=E.fromstring(z.read('xl/worksheets/sheet1.xml'))
  for c in sheet.findall('.//x:c',ns):
   if 's' in c.attrib:c.set('s',str(maps['cellXfs'][int(c.get('s'))]))
   if c.get('t')=='s':v=c.find(xp('v'));v.text=str(indices[int(v.text)])
  for row in sheet.findall('x:sheetData/x:row',ns):
   if 's' in row.attrib:row.set('s',str(maps['cellXfs'][int(row.get('s'))]))
  for col in sheet.findall('x:cols/x:col',ns):
   if 'style' in col.attrib:col.set('style',str(maps['cellXfs'][int(col.get('style'))]))
  for rule in sheet.findall('.//x:cfRule',ns):
   if 'dxfId' in rule.attrib:rule.set('dxfId',str(maps['dxfs'][int(rule.get('dxfId'))]))
  name=part['name']
  if name not in sheetroots:
   assert part['offset']==0;idx=len(sheetroots)+1;sheetroots[name]=sheet;groups[name]=idx
   rel=E.fromstring(z.read('xl/worksheets/_rels/sheet1.xml.rels'))
   for relationship in rel:
    assert relationship.get('Type').endswith('/table'),relationship.attrib
    target=relationship.get('Target').lstrip('/');table=E.fromstring(z.read(target));table.set('id',str(idx));newtarget=f'xl/tables/table{idx}.xml';relationship.set('Target','/'+newtarget)
    ref=table.get('ref');letter=re.search(r':([A-Z]+)',ref)[1];fullref=f'A1:{letter}{len(data[name]["rows"])+1}';table.set('ref',fullref)
    af=table.find(xp('autoFilter'))
    if af is not None:af.set('ref',fullref)
    extras[newtarget]=xml(table)
   extras[f'xl/worksheets/_rels/sheet{idx}.xml.rels']=xml(rel)
  else:
   dest=sheetroots[name].find(xp('sheetData'))
   for row in sheet.findall('x:sheetData/x:row',ns):
    oldrow=int(row.get('r'))
    if oldrow==1:continue
    newrow=oldrow+part['offset'];row.set('r',str(newrow))
    for c in row.findall(xp('c')):c.set('r',re.sub(r'\d+$',str(newrow),c.get('r')))
    dest.append(row)
 print('ASSEMBLED',name,part['offset'],flush=True)
for name,sheet in sheetroots.items():
 n=len(data[name]['rows']);expected_rows=n+1
 if name=='井段邻井因素':
  assert len(sheet.find(xp('sheetData')))==expected_rows
  for cf in sheet.findall(xp('conditionalFormatting')):cf.set('sqref',re.sub(r':([A-Z]+)\d+',lambda m:':'+m[1]+str(n+1),cf.get('sqref')))
 extras[f'xl/worksheets/sheet{groups[name]}.xml']=xml(sheet)
for tag in ['numFmts','fonts','fills','borders','cellStyleXfs','cellXfs','dxfs']:
 q=styles.find(xp(tag));q.set('count',str(len(q)))
for child in sorted(list(styles),key=lambda q:style_order.index(E.QName(q).localname)):styles.remove(child);styles.append(child)
shared.set('count',str(sum(c.get('t')=='s' for sh in sheetroots.values() for c in sh.findall('.//x:c',ns))));shared.set('uniqueCount',str(len(shared)))
sheets=workbook.find(xp('sheets'))
for child in list(sheets):sheets.remove(child)
for child in list(wr):
 if child.get('Type').endswith('/worksheet'):wr.remove(child)
for name,idx in groups.items():
 rid=f'RSheet{idx}';E.SubElement(sheets,xp('sheet'),name=name,sheetId=str(idx),attrib={'{'+R+'}id':rid});E.SubElement(wr,'{'+PR+'}Relationship',Id=rid,Type=R+'/worksheet',Target=f'/xl/worksheets/sheet{idx}.xml')
for child in list(ct):
 if child.get('PartName','').startswith(('/xl/worksheets/','/xl/tables/')):ct.remove(child)
for name in extras:
 if name.endswith('.rels'):continue
 kind='worksheet' if name.startswith('xl/worksheets/') else 'table';E.SubElement(ct,'{'+CT+'}Override',PartName='/'+name,ContentType=f'application/vnd.openxmlformats-officedocument.spreadsheetml.{kind}+xml')
extras.update({'xl/styles.xml':xml(styles),'xl/sharedStrings.xml':xml(shared),'xl/workbook.xml':xml(workbook),'xl/_rels/workbook.xml.rels':xml(wr),'[Content_Types].xml':xml(ct)})
for name,blob in base.items():
 if name not in extras and not name.startswith(('xl/worksheets/','xl/tables/')):extras[name]=blob
final=O/'压裂井段邻井_地质工程因素与五类形态.xlsx';tmp=O/'工作簿合并中.xlsx'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,blob in extras.items():z.writestr(name,blob)
tmp.replace(final)
print('FINAL',final.stat().st_size,'SHEETS',len(sheetroots),'SHARED_STRINGS',len(shared))
(B/'assembly_checks.json').write_text(json.dumps({'parts':len(manifest),'sheets':list(sheetroots),'file_bytes':final.stat().st_size,'assembly':'Artifact Tool authored parts combined with preserved OOXML values, styles, tables, formulas and relationships'},ensure_ascii=False,indent=2),encoding='utf-8')
