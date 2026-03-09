import sys

with open("platform_app/index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines[320:800]):
    opens = l.count("<div")
    closes = l.count("</div")
    if opens != closes:
        print(f"Line {i+321} [+str(opens) -str(closes)]: {l.strip()[:60]}")
