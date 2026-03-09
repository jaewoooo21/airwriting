import re

with open('platform_app/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

stack = []
for i, l in enumerate(lines):
    if 'id="panel-' in l or 'id="tab-' in l:
        print(f'{l.strip()[:60]} starts at {i+1} : stack depth={len(stack)}')
    
    for match in re.finditer(r'</?(div|section|nav|aside|header|footer|main)\b[^>]*>', l):
        tag = match.group(0)
        if tag.startswith('</'):
            if stack:
                stack.pop()
            else:
                print(f'UNBALANCED CLOSE AT LINE {i+1}: {l.strip()[:60]}')
        else:
            stack.append(f'{tag} at Line {i+1}')

if stack:
    print('Unclosed tags at end:', [(t) for t in stack])
else:
    print('PERFECTLY BALANCED HTML!')
