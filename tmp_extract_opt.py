from pathlib import Path
text = Path('cogniforge/api/execute_endpoint.py').read_text()
start = text.index('    async def _execute_optimization')
end = text.index('    async def _execute_vision_refinement', start)
print(text[start:end])
