import sys
import os
from pathlib import Path

# Ensure src is on path to import recovery_system
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / 'src'
sys.path.insert(0, str(src_dir))

from recovery_system import RecoverySystem

print("=== NON-INTERACTIVE RECOVERY SYSTEM DEMO ===")

recovery = RecoverySystem(checkpoint_dir=str(repo_root / 'generated' / 'test_recovery'))

# Create simple scripts
work_dir = Path.cwd()

good_script = work_dir / 'good_script_demo.py'
good_script.write_text(
    """
import sys
print('Good script running...')
sys.exit(0)
""",
    encoding='utf-8'
)

bad_script = work_dir / 'bad_script_demo.py'
bad_script.write_text(
    """
import sys
print('Bad script running...')
raise RuntimeError('Simulated failure')
""",
    encoding='utf-8'
)

final_script = work_dir / 'final_script_demo.py'
final_script.write_text(
    """
import sys
print('Final script executed successfully!')
sys.exit(0)
""",
    encoding='utf-8'
)

# Save a checkpoint first
recovery.save_checkpoint(str(good_script), {"param": True}, output="ok", execution_time=0.1)

# Monkeypatch UI to auto-skip to final
recovery.show_recovery_button = lambda error_info: 'skip'

# Execute bad script with recovery to final
result = recovery.execute_with_recovery(bad_script, {"param": "bad"}, final_script_path=final_script)
print("Result:", result)

# Cleanup
for p in [good_script, bad_script, final_script]:
    try:
        p.unlink()
    except Exception:
        pass
