"""Post-hoc reporting only; never feeds observations/actions back to a policy."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.stats import beta
p=argparse.ArgumentParser();p.add_argument('cohort');a=p.parse_args();root=Path(a.cohort)
results=[];groups={};pairs={};manifests=set()
for path in sorted(root.glob('*/evaluator/result.json')):
 r=json.loads(path.read_text());folder=path.parent.parent
 if r.get('infrastructure_only') or r.get('development_smoke'):raise ValueError('development result in cohort')
 c=json.loads((path.parent/'configuration.json').read_text());manifests.add(c['freeze']['manifest_sha256'])
 row={'trial':folder.name,'passes':r['passes'],'failure_class':r['failure_class'],'error':r['error'],
      'simulated_s':r['simulated_s'],'wall_s':r['wall_s'],'returned_calls':len(r['calls']),
      'submitted_observations':len(list((folder/'observations').glob('[0-9][0-9][0-9]'))),
      'completed_skills':sum(x.get('status')=='completed' for x in r['history']),
      'peak_penetration_mm':1000*r['peak_penetration_m'],'intact_fasteners':r['fasteners']['intact']}
 results.append(row);groups.setdefault(r['model'],[]).append((r,folder))
 initial=folder/'observations/000';seed=r['seed_evaluator_only']
 pairs.setdefault(str(seed),{})[r['model']]={'images':json.loads((initial/'image_hashes.json').read_text()),'packet_sha256':hashlib.sha256((initial/'packet.json').read_bytes()).hexdigest()}
if len(manifests)>1:raise ValueError('Refuse to pool different freezes')
def describe(x):return None if not x else {'n':len(x),'p50':float(np.median(x)),'p95':float(np.percentile(x,95)),'min':float(min(x)),'max':float(max(x))}
models={}
for name,items in groups.items():
 n=len(items);k=sum(x['passes'] for x,_ in items);valid=[];all_attempts=[];errors=[]
 for r,folder in items:
  for call in r['calls']:
   all_attempts.append(call['latency_wall_s']);response=json.loads((folder/'evaluator'/f"response_{call['call']:03d}.json").read_text())
   if not response.get('error') and isinstance(response.get('text'),str):valid.append(call['latency_wall_s'])
   if call.get('error'):errors.append({'trial':folder.name,'call':call['call'],'error':call['error']})
 models[name]={'successes':k,'started_trials':n,'exact_binomial_95pct':[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))],
 'all_attempt_latency_s':describe(all_attempts),'completed_response_latency_s':describe(valid),'inference_errors':errors}
checks={}
for seed,items in pairs.items():
 values=list(items.values());checks[seed]={'both_models_present':len(values)==2,'identical_initial_inputs':len(values)==2 and values[0]==values[1],'details':items}
out={'manifest_sha256':next(iter(manifests),None),'models':models,'trials':results,'paired_initial_inputs':checks,
     'scope':'Three-case supervisory pilot; not image dependence, code-vs-EE, learned motor control or physical validation.',
     'latency_note':'Completed response latency excludes provider errors but may include invalid action JSON. Attempt latencies include timeouts; these are not uncensored completion times. An in-flight request at episode deadline may lack a returned-call record; compare submitted_observations with returned_calls.',
     'confidence_note':'Exact intervals are descriptive. Three layouts do not establish high reliability or model superiority.'}
(root/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
