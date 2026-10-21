"""Camera controller client. Reads only public packet; sends motor command JSON.

No simulator imports or evaluator-file access. Simulation pauses between requests.
"""
import argparse,json,time,uuid
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--action',help='JSON motor command');a=p.parse_args()
 latest=a.public/'latest.json'
 if a.action is None:print(latest.read_text());return
 action=json.loads(a.action)
 if not isinstance(action,dict):raise ValueError('Action must be a JSON object')
 token=uuid.uuid4().hex;request=a.public/'requests'/f'{token}.json';request.parent.mkdir(exist_ok=True)
 temporary=request.with_suffix('.tmp')
 temporary.write_text(json.dumps({'request_id':token,'action':action})+'\n');temporary.replace(request)
 start=time.monotonic()
 while time.monotonic()-start<55:
  reply=a.public/'replies'/f'{token}.json'
  if reply.exists():print(reply.read_text());return
  time.sleep(.1)
 print(json.dumps({'pending':token,'reply_path':str(a.public/'replies'/f'{token}.json')}))

if __name__=='__main__':main()
