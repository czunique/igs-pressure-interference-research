$docInput='F:\论文库\IGS\实验\结论\IGS中文初稿_20260916\IGS中文初稿_邻井压力形态与可解释预测.docx'
$pdfOutput='F:\论文库\IGS\实验\过程\分类预测与SHAP_20260916\论文排版核查\manuscript.pdf'
$wordForIGS=New-Object -ComObject Word.Application
try {
$wordForIGS.Visible=$false
$wordForIGS.DisplayAlerts=0
$igsDoc=$wordForIGS.Documents.Open($docInput,$false,$true)
$igsDoc.ExportAsFixedFormat($pdfOutput,17)
$igsDoc.Close(0)
} finally {$wordForIGS.Quit()}
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 -c "import pypdfium2 as pdfium;from pathlib import Path;b=Path(r'F:/论文库/IGS/实验/过程/分类预测与SHAP_20260916/论文排版核查');p=pdfium.PdfDocument(str(b/'manuscript.pdf'));print('pages',len(p));[(page.render(scale=1.6).to_pil().save(b/f'page-{i+1:02d}.png')) for i,page in enumerate(p)]"
