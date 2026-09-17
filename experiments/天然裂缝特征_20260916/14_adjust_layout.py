from pathlib import Path
p=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916/14_plot_all_overlays.py');s=p.read_text(encoding='utf-8-sig')
s=s.replace("palette={p:colors(i%20) for i,p in enumerate(allplatforms)}", "palette={p:(colors(i) if i<20 else plt.get_cmap('tab20b')(i-20)) for i,p in enumerate(allplatforms)}")
a=s.index("  absent=[p for p in expected");b=s.index("  footer=",a)
s=s[:a]+'''  absent=[p for p in expected if p not in plotted_platforms]
  missingtext=f'缺坐标或有效轨迹：{len(unlocated)}口\\n目录所列但位于范围外：{len(outside)}口'
  if absent:missingtext+='\\n未展示平台：'+'、'.join(absent)
  missingtext+='\\n逐井原因见配套CSV清单'
  wrapped='\\n'.join(textwrap.fill(line,width=23) for line in missingtext.splitlines());side.text(0,min(yinfo-.24,.28),wrapped,va='top',fontsize=8,color='#6b4f41',linespacing=1.4)
''' +s[b:]
p.write_text(s,encoding='utf-8')
