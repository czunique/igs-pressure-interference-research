import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const B='F:/论文库/IGS/实验/过程/天然裂缝特征_20260916';
const O='F:/论文库/IGS/实验/结论/天然裂缝特征_20260916';
const data=JSON.parse(await fs.readFile(B+'/workbook_data.json','utf8'));
const wb=Workbook.create();
const col=(i)=>{let s='';for(i++;i>0;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
const titles={'井段邻井特征':'天然裂缝特征：井段与邻井','平台覆盖':'各平台数据覆盖与待补项','井段定位':'分段区间与空间定位','地震数据':'SGY数据来源与格式核查','指标说明':'指标定义、计算假设与使用范围'};
const previews={'井段邻井特征':'A1:J14','平台覆盖':'A1:H14','井段定位':'A1:I13','地震数据':'A1:F11','指标说明':'A1:B12'};
for(const [name,d] of Object.entries(data)){
 const s=wb.worksheets.add(name);s.showGridLines=false;
 const n=d.rows.length,m=d.columns.length,end=col(m-1),body=s.getRange(`A5:${end}${n+5}`);
 body.values=[d.columns,...d.rows];body.format.font={name:'Microsoft YaHei',size:10,color:'#263445'};body.format.rowHeight=20;body.format.columnWidth=19;body.format.verticalAlignment='center';
 s.getRange(`A2:${col(Math.min(m-1,8))}2`).merge();s.getRange('A2').values=[[titles[name]]];s.getRange('A2').format.font={name:'Microsoft YaHei',size:16,bold:true,color:'#18344b'};
 const header=s.getRange(`A5:${end}5`);header.format.fill='#24445C';header.format.font={name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'};header.format.wrapText=true;header.format.rowHeight=48;header.format.horizontalAlignment='center';
 if(name!=='指标说明'){s.tables.add(`A5:${end}${n+5}`,true,'Table'+(Object.keys(data).indexOf(name)+1));s.freezePanes.freezeRows(5);}
 if(name==='井段邻井特征'){
  s.tabColor='#24445C';s.freezePanes.freezeColumns(4);s.getRange('A3:J3').merge();s.getRange('A3').values=[['泸201H5已计算294组；其余记录保留待补原因。角度含方向假设，详见“指标说明”。']];s.getRange('A3').format.font={name:'Microsoft YaHei',size:10,color:'#626b73'};
  s.getRange(`C6:C${n+5}`).setNumberFormat('0');s.getRange(`E6:M${n+5}`).setNumberFormat('#,##0.00');s.getRange(`J6:J${n+5}`).setNumberFormat('0.0%');s.getRange(`N6:N${n+5}`).setNumberFormat('0');s.getRange(`R6:U${n+5}`).setNumberFormat('#,##0.00');
  s.getRange(`A5:D${n+5}`).format.columnWidth=15;s.getRange(`C5:C${n+5}`).format.columnWidth=9;s.getRange(`E5:I${n+5}`).format.columnWidth=22;s.getRange(`J5:J${n+5}`).format.columnWidth=17;s.getRange(`O5:O${n+5}`).format.columnWidth=67;s.getRange(`Q5:Q${n+5}`).format.columnWidth=45;s.getRange(`W5:X${n+5}`).format.columnWidth=95;
  s.getRange(`G6:G${n+5}`).formulas=d.rows.map((_,i)=>[`=IF(ISNUMBER(F${i+6}),F${i+6}/K${i+6}*1000,"")`]);
 } else if(name==='平台覆盖'){
  s.getRange(`B5:B${n+5}`).format.columnWidth=43;s.getRange(`L5:L${n+5}`).format.columnWidth=70;s.getRange(`D6:K${n+5}`).setNumberFormat('#,##0');
 } else if(name==='井段定位'){
  s.getRange(`D6:H${n+5}`).setNumberFormat('#,##0.00');s.getRange(`I5:I${n+5}`).format.columnWidth=30;s.getRange(`J5:K${n+5}`).format.columnWidth=95;
 } else if(name==='地震数据'){
  s.getRange(`A5:A${n+5}`).format.columnWidth=42;s.getRange(`B5:B${n+5}`).format.columnWidth=80;s.getRange(`M5:M${n+5}`).format.columnWidth=110;
 } else {
  s.getRange(`A5:A${n+5}`).format.columnWidth=27;s.getRange(`B5:B${n+5}`).format.columnWidth=105;s.getRange(`B6:B${n+5}`).format.wrapText=true;s.getRange(`A6:B${n+5}`).format.rowHeight=44;
 }
 console.log('sheet prepared',name,n);
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'井段邻井特征!A6:K10',include:'values,formulas',tableMaxRows:5,tableMaxCols:11,maxChars:2200})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(O+'/天然裂缝特征_井段邻井汇总_阶段版.xlsx');console.log('EXPORTED');
for(const [name,range] of Object.entries(previews)){
 const p=await wb.render({sheetName:name,range,scale:1.4,format:'png'});await fs.writeFile(B+'/预览_'+name+'.png',new Uint8Array(await p.arrayBuffer()));console.log('rendered',name);
}
await fs.writeFile(B+'/workbook_build_checks.json',JSON.stringify({sheets:Object.fromEntries(Object.entries(data).map(([k,v])=>[k,v.rows.length])),recalculated:true,exported:true,rendered:true},null,2));
