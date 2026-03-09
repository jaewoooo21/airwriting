import re

with open('platform_app/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

stack = []
for i, l in enumerate(lines):
    # Find all <div> and </div> tags exactly
    for match in re.finditer(r'</?(div)\b[^>]*>', l):
        tag = match.group(0)
        if tag.startswith('</'):
            if stack:
                stack.pop()
        else:
            stack.append(i + 1)
            
    if 'id="panel-hardware"' in l:
        print(f'Line {i+1}: panel-hardware')
        print(f'Current stack length (depth): {len(stack)}')
        print(f'Unclosed tags opened at lines: {stack[-5:]}')
        break
