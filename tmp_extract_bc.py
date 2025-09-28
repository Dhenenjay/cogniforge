from pathlib import Path
text = Path('cogniforge/api/execute_endpoint.py').read_text()
start = text.index('    async def _execute_behavior_cloning')
end = text.index('    async def _execute_optimization', start)
print(text[start:end])
