from pathlib import Path
text = Path('cogniforge/core/optimization.py').read_text()
for line in text.splitlines():
    if 'PolicyOptimizer' in line:
        print(line)
