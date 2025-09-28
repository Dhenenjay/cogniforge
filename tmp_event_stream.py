from pathlib import Path
text = Path('cogniforge/api/execute_endpoint.py').read_text()
start = text.index('        # Event streaming')
print(text[start:start+200])
