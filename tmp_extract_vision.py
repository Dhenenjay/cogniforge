from pathlib import Path
text = Path('cogniforge/api/execute_endpoint.py').read_text()
start = text.index('    async def _execute_vision_refinement')
end = text.index('    async def _generate_execution_code', start)
print(text[start:end])
