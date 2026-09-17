import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const B='F:/论文库/IGS/实验/过程/分类预测与SHAP_20260916',O='F:/论文库/IGS/实验/结论/分类预测与SHAP_20260916';
const data=JSON.parse(await fs.readFile(B+'/model_workbook.json','utf8'));const wb=Workbook.create();
const col=i=>{let s='';for(i++;i>0;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
let k=0;for(const [name,d] of Object.entries(data)){
const sh=wb.worksheets.add(name),n=d.rows.length,m=d.columns.length,end=col(m-1);sh.showGridLines=false;
sh.getRange(`A1:${end}${n+1}`).values=[d.columns,...d.rows];const used=sh.getRange(`A1:${end}${n+1}`);used.format.font={name:'Microsoft YaHei',size:10,color:'#243746'};used.format.rowHeight=24;used.format.columnWidth=20;
const h=sh.getRange(`A1:${end}1`);h.format.fill='#24445C';h.format.font={name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'};h.format.wrapText=true;h.format.rowHeight=48;
sh.getRange(`A1:A${n+1}`).format.columnWidth=/预测|输入/.test(name)?52:30;sh.freezePanes.freezeRows(1);sh.tables.add(`A1:${end}${n+1}`,true,'Results'+(++k));
for(let j=0;j<m;j++)if(d.rows.some(r=>typeof r[j]==='number'))sh.getRange(`${col(j)}2:${col(j)}${n+1}`).setNumberFormat(/样本数|分组数|^n$|observed_n|outer_fold|present_classes/.test(d.columns[j])?'0':'0.000');
if(name==='方法与来源'){sh.getRange(`B1:B${n+1}`).format.columnWidth=105;sh.getRange(`B2:B${n+1}`).format.wrapText=true;sh.getRange(`A2:B${n+1}`).format.rowHeight=68;}
if(name==='折外预测')sh.getRange(`E2:E${n+1}`).setNumberFormat('@');
const view=await wb.render({sheetName:name,range:`A1:${col(Math.min(m-1,4))}${Math.min(n+1,7)}`,scale:1.5,format:'png'});await fs.writeFile(B+'/表格预览_'+name+'.png',new Uint8Array(await view.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(O+'/分类预测与SHAP分析结果.xlsx');console.log('EXPORTED',Object.keys(data));

