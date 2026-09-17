import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const B='F:/论文库/IGS/实验/过程/地质工程因素_20260916',O='F:/论文库/IGS/实验/结论/地质工程因素_20260916';
const data=JSON.parse(await fs.readFile(B+'/workbook_data.json','utf8'));const selected=process.argv[2]?.split(',');const previous=selected?JSON.parse(await fs.readFile(B+'/parts_manifest.json','utf8')):[];const manifest=previous.filter(p=>!selected.includes(p.name));await fs.mkdir(B+"/workbook_parts",{recursive:true});
const col=(i)=>{let s='';for(i++;i>0;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
const renders={};
for(const [name,whole] of Object.entries(data)){
 if(selected&&!selected.includes(name))continue;
 for(let offset=0;offset<whole.rows.length;offset+=(name==='井段邻井因素'?7000:whole.rows.length)){
 const d={columns:whole.columns,rows:whole.rows.slice(offset,offset+(name==='井段邻井因素'?7000:whole.rows.length))};const wb=Workbook.create();
 const sh=wb.worksheets.add(name);sh.showGridLines=false;const n=d.rows.length,m=d.columns.length,end=col(m-1);const used=sh.getRange(`A1:${end}${n+1}`);
 for(let start=0;start<n;start+=1500)sh.getRangeByIndexes(start+1,0,Math.min(1500,n-start),m).values=d.rows.slice(start,start+1500);
 sh.getRange(`A1:${end}1`).values=[d.columns];used.format.font={name:'Microsoft YaHei',size:10,color:'#263445'};used.format.columnWidth=20;used.format.rowHeight=22;used.format.verticalAlignment='center';
 const head=sh.getRange(`A1:${end}1`);head.format.fill='#24445C';head.format.font={name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'};head.format.wrapText=true;head.format.rowHeight=58;head.format.horizontalAlignment='center';
 sh.freezePanes.freezeRows(1);sh.freezePanes.freezeColumns(name==='井段邻井因素'?3:1);sh.tables.add(`A1:${end}${n+1}`,true,'Data'+(Object.keys(data).indexOf(name)+1));
 for(let j=0;j<m;j++){
  const c=d.columns[j],letter=col(j),rng=sh.getRange(`${letter}2:${letter}${n+1}`);
  const numeric=d.rows.some(r=>typeof r[j]==='number');
  if(numeric)rng.setNumberFormat(/覆盖率/.test(c)?'0.0%':/段号|压裂段$|井号$|邻井号$|井数|井段数|组合数|曲线数|簇数|孔数|^well$/.test(c)?'0':'0.000');
  if(/井段键|组合键|压裂井-压裂段-压窜井/.test(c))sh.getRange(`${letter}1:${letter}${n+1}`).format.columnWidth=50;
  if(/分类|状态|说明|问题|原曲线分类|关联问题/.test(c))sh.getRange(`${letter}1:${letter}${n+1}`).format.columnWidth=30;
  if(/来源定位|候选值及来源|来源编号|来源$/.test(c))sh.getRange(`${letter}1:${letter}${n+1}`).format.columnWidth=c==='来源定位'?110:44;
 }
 if(name==='平台覆盖')sh.getRange(`B2:J${n+1}`).setNumberFormat('0');
 if(name==='曲线关联')sh.getRange(`A2:A${n+1}`).setNumberFormat('@');
 if(name==='质量核查'){sh.getRange(`C1:C${n+1}`).format.columnWidth=52;sh.getRange(`F1:F${n+1}`).format.columnWidth=100;sh.getRange(`C2:G${n+1}`).format.wrapText=true;for(let i=0;i<n;i++){const row=d.rows[i];const lines=Math.max(String(row[2]||'').length/24,String(row[5]||'').length/75,String(row[6]||'').length/75);sh.getRange(`A${i+2}:G${i+2}`).format.rowHeight=Math.min(400,Math.max(55,Math.ceil(lines)*18));}}
 if(name==='井段邻井因素'){
  sh.tabColor='#24445C';sh.getRange(`B1:B${n+1}`).format.columnWidth=16;sh.getRange(`C1:C${n+1}`).format.columnWidth=26;
  sh.getRange(`C2:C${n+1}`).conditionalFormats.addCustom('OR(C2="待归入5类",C2="多记录分类不一致")',{fill:'#FFF1D4',font:{color:'#865C13'}});
 }
 if(name==='方法说明'){
  sh.getRange(`A1:A${n+1}`).format.columnWidth=25;sh.getRange(`B1:B${n+1}`).format.columnWidth=112;sh.getRange(`B2:B${n+1}`).format.wrapText=true;sh.getRange(`A2:B${n+1}`).format.rowHeight=72;head.format.rowHeight=28;
 }
 if(name==='来源定位'){
  sh.getRange(`A1:A${n+1}`).format.columnWidth=15;sh.getRange(`B1:B${n+1}`).format.columnWidth=120;sh.getRange(`B2:B${n+1}`).format.wrapText=true;
  for(let i=0;i<n;i++)sh.getRange(`A${i+2}:B${i+2}`).format.rowHeight=Math.min(400,Math.max(50,Math.ceil(String(d.rows[i][1]).length/70)*17));
 }
 if(name==='加权计算示例'){
  sh.getRange(`G2:G${n+1}`).formulas=d.rows.map((_,i)=>[`=MAX(0,MIN(C${i+2},E${i+2})-MAX(B${i+2},D${i+2}))`]);
  sh.getRange(`H2:H${n+1}`).formulas=d.rows.map((_,i)=>[`=G${i+2}*F${i+2}`]);
  sh.getRange(`F${n+4}`).values=[['井段加权杨氏模量_GPa']];sh.getRange(`G${n+4}`).formulas=[[`=SUM(H2:H${n+1})/SUM(G2:G${n+1})`]];sh.getRange(`G${n+4}`).setNumberFormat('0.000');
  sh.getRange(`F${n+5}`).values=[['有效覆盖率']];sh.getRange(`G${n+5}`).formulas=[[`=SUM(G2:G${n+1})/(C2-B2)`]];sh.getRange(`G${n+5}`).setNumberFormat('0.0%');sh.getRange('F1:F8').format.columnWidth=30;
 }
 renders[name]=name==='方法说明'?'A1:B6':name==='来源定位'?'A1:B7':name==='加权计算示例'?'A1:H8':`A1:${col(Math.min(m-1,6))}8`;
 console.log('PREPARED',name,offset,n,m);
 wb.recalculate();
 if(name==='加权计算示例'){
  const calc=sh.getRange(`G${n+4}`).values[0][0];const check=JSON.parse(await fs.readFile(B+'/data_checks.json','utf8'));if(Math.abs(calc-check.example_young_GPa)>1e-8)throw new Error('Weighted example mismatch '+calc);
  const old=sh.getRange('F2').values[0][0];sh.getRange('F2').values=[[old+1]];const changed=sh.getRange(`G${n+4}`).values[0][0];if(!(changed>calc))throw new Error('Formula did not respond');sh.getRange('F2').values=[[old]];wb.recalculate();
  await fs.writeFile(B+'/artifact_inspection.ndjson',(await wb.inspect({kind:'table',range:'加权计算示例!A1:H8',include:'values,formulas',tableMaxRows:8,tableMaxCols:8,maxChars:3000})).ndjson);
 }
 const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1500});console.log('ERROR_SCAN',name,errors.ndjson);
 const part=previous.find(p=>p.name===name&&p.offset===offset)?.path||('workbook_parts/'+String(manifest.length+1).padStart(2,'0')+'.xlsx');
 const out=await SpreadsheetFile.exportXlsx(wb);await out.save(B+'/'+part);manifest.push({name,offset,n,m,path:part});console.log('EXPORTED_PART',part);
 if(offset===0){const image=await wb.render({sheetName:name,range:renders[name],scale:1.25,format:'png'});await fs.writeFile(B+'/预览_'+name+'.png',new Uint8Array(await image.arrayBuffer()));console.log('RENDERED',name);}
 manifest.sort((a,b)=>Object.keys(data).indexOf(a.name)-Object.keys(data).indexOf(b.name)||a.offset-b.offset);await fs.writeFile(B+'/parts_manifest.json',JSON.stringify(manifest,null,2));
 }
}
await fs.writeFile(B+'/build_checks.json',JSON.stringify({recalculated:true,input_change_recalculated:true,parts_exported:manifest.length,rendered:Object.keys(renders)},null,2));
