import re

with open('platform_app/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

stack = []
for i, l in enumerate(lines[320:572]):
    for match in re.finditer(r'</?(div)\b[^>]*>', l):
        tag = match.group(0)
        if tag.startswith('</'):
            if stack:
                stack.pop()
        else:
            stack.append(i + 321)
    
    if len(stack) == 0 and i > 5:
        print(f"DEPTH DROPPED TO 0 AT LINE {i+321}")
        break

print(f"Final stack at line 572: {stack}")
