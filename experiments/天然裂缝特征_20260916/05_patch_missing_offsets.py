from pathlib import Path
p=Path('F:/论文库/IGS/实验/过程/天然裂缝特征_20260916/05_geometry.py');s=p.read_text(encoding='utf-8-sig');s=s.replace("hh=h[(h.platform==key[0])&(h.well==key[1])]", "valid_offsets=q.dx.notna()&q.dy.notna()\n if valid_offsets.sum()>=3:q=q[valid_offsets].copy()\n hh=h[(h.platform==key[0])&(h.well==key[1])]");p.write_text(s,encoding='utf-8')
