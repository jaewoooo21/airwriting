import os
file_path = "c:/Users/USER/airwriting_imu_only/platform_app/style.css"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    'background: #050505;': 'background: var(--bg-panel);',
    'background: #0a0a0a;': 'background: var(--bg-panel);',
    'background: #000;': 'background: var(--bg-panel);',
    'background: #111;': 'background: #F1F5F9;',
    'background: #111111;': 'background: #F1F5F9;',
    'background-color: #0f172a;': 'background-color: var(--bg-panel);',
    'border: 1px solid #222;': 'border: 1px solid var(--border-color);',
    'border: 1px solid #333;': 'border: 1px solid var(--border-color);',
    'border: 1px solid #111;': 'border: 1px solid var(--border-color);',
    'border-bottom: 1px solid #333;': 'border-bottom: 1px solid var(--border-color);',
    'color: #ccc;': 'color: var(--text-muted);',
    'color: #cccccc;': 'color: var(--text-muted);',
    'color: #888;': 'color: #94A3B8;',
    'color: #888888;': 'color: #94A3B8;',
    'color: #fff;': 'color: var(--text-main);',
    'color: #ffffff;': 'color: var(--text-main);',
    'rgba(255, 255, 255, 0.05)': 'rgba(15, 23, 42, 0.03)',
    'rgba(255, 255, 255, 0.1)': 'rgba(15, 23, 42, 0.06)',
    'rgba(255, 255, 255, 0.03)': 'rgba(15, 23, 42, 0.02)',
    'rgba(255, 255, 255, 0.8)': 'rgba(15, 23, 42, 0.1)',
    'rgba(255,255,255,0.05)': 'rgba(15, 23, 42, 0.03)',
    'rgba(0, 0, 0, 0.4)': 'rgba(255, 255, 255, 0.8)',
    '#050505': 'var(--bg-panel)'
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Theme colors replaced.")
