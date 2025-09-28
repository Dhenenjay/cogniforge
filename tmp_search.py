from pathlib import Path
root = Path('.')
found = False
for path in root.rglob('*.py'):
    if 'PolicyOptimizer' in path.read_text(encoding='utf-8', errors='ignore'):
        print(path)
        found = True
if not found:
    print('not found')
