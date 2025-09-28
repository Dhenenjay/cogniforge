from pathlib import Path
text = Path('cogniforge/api/execute_endpoint.py').read_text()
start = text.index('        result = ExecutionResult(')
print(text[start:start+400])
