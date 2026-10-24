import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parent;folder=root/'online-backups25';stage=root/'varied-release-staging';repo='AlmondGod/is-rvlmrl';tag='cap-rl-varied-assembly-2026-10-24'
for step in range(3,9):
 local=folder/f'step-{step:04d}.tar.gz';verified=folder/f'step-{step:04d}.verified.json';name=f'online25-step-{step:04d}.tar.gz'
 while not verified.exists():time.sleep(10)
 expected=json.loads(verified.read_text())
 with local.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 assert sha==expected['archive_sha256']
 target=stage/name
 if not target.exists():os.link(local,target)
 while True:
  try:
   view=json.loads(subprocess.check_output(['gh','release','view',tag,'--repo',repo,'--json','assets'],text=True))
   asset=next((a for a in view['assets'] if a['name']==name),None)
   if asset:
    assert asset['size']==local.stat().st_size and asset['digest']=='sha256:'+sha,asset
    break
   subprocess.run(['gh','release','upload',tag,str(target),'--repo',repo],check=True)
  except subprocess.CalledProcessError as e:
   print('PUBLISH_RETRY',step,e.returncode,flush=True);time.sleep(30)
 report=dict(passes=True,step=step,local_payload_verification=expected,github_asset=asset)
 (folder/f'step-{step:04d}.github.json').write_text(json.dumps(report,indent=2)+'\n')
 print('GITHUB_BACKUP_VERIFIED',step,asset['size'],sha,flush=True)
