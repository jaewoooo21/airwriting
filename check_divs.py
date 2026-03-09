import sys

with open('platform_app/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = open('div_depths.txt', 'w', encoding='utf-8')

depth = 0
for i, l in enumerate(lines):
    opens = l.count('<div')
    closes = l.count('</div')
    depth += opens - closes
    if 'id="panel-' in l or 'tab-content' in l:
        out.write(f'Line {i+1} [Depth: {depth}]: {l.strip()[:60]}\n')
    if depth < 0:
        out.write(f'NEGATIVE DEPTH AT LINE {i+1}\n')
        depth = 0

out.close()
