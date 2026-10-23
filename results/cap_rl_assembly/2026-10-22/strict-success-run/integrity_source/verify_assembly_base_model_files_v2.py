"""Verify actual cached bytes against official metadata for the pinned revision."""
import hashlib,json,pathlib,time
from huggingface_hub import model_info
run=pathlib.Path('/opt/cap-assembly/assembly04');metadata=json.loads((run/'metadata.json').read_text())
info=model_info(metadata['model'],revision=metadata['model_revision'],files_metadata=True)
assert info.sha==metadata['model_revision']
remote={f.rfilename:f for f in info.siblings}
root=pathlib.Path('/opt/cap-assembly/hf/hub')/('models--'+metadata['model'].replace('/','--'))/'snapshots'/info.sha
started=time.monotonic();records=[]
for path in sorted(root.iterdir()):
 assert path.is_file() and path.name in remote,str(path)
 record=remote[path.name];size=path.stat().st_size
 expected=record.lfs.sha256 if record.lfs else record.blob_id
 assert size==(record.lfs.size if record.lfs else record.size)
 sha256=hashlib.sha256();git=hashlib.sha1(f'blob {size}\0'.encode())
 with path.open('rb') as handle:
  while chunk:=handle.read(8*1024*1024):
   sha256.update(chunk)
   if not record.lfs:git.update(chunk)
 actual=sha256.hexdigest() if record.lfs else git.hexdigest()
 assert actual==expected, 'Official content identity mismatch: '+str(path)
 records.append(dict(file=path.name,bytes=size,sha256=sha256.hexdigest(),expected_identity=expected,
  identity_algorithm='sha256' if record.lfs else 'git_blob_sha1',
  cache_storage_identifier=path.resolve().name))
result=dict(passes=True,model=metadata['model'],revision=info.sha,files=records,
 wall_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 authority='Official Hugging Face model_info at the pinned revision, files_metadata=True',
 scope='Actual cached weight shards, tokenizer, template and configuration match official Git/LFS content identities. No files or weights changed.',
 prior_check_correction='The first helper incorrectly treated the final resolved cache filename as a SHA256. HF shared cache filenames can be Xet IDs; official revision metadata supplies content checksums. The failed helper/log are retained.')
out=run/'base_model_integrity.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
