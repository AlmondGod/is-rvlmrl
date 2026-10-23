"""Preserve verification evidence automatically, without selecting or tuning."""
import json,pathlib,subprocess,time
root=pathlib.Path('/opt/cap-assembly')
status=root/'assembly04-verification-status.json'
while True:
 state=json.loads(status.read_text())
 if state['state']=='failed':raise RuntimeError(state)
 if state['state']=='complete':break
 time.sleep(15)
subprocess.run([str(root/'venv/bin/python'),str(root/'verification-tools-v2/package_cap_assembly.py'),
 '--run',str(root/'assembly04'),'--replay',str(root/'assembly04-test-replay'),
 '--out',str(root/'assembly04-verified-evidence.tar.gz')],check=True)
