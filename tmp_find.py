from pathlib import Path
text = Path('cogniforge/core/optimization.py').read_text()
idx = text.find('PolicyOptimizer')
print('index', idx)
if idx != -1:
    start = max(0, idx - 200)
    end = idx + 400
    print(text[start:end])
