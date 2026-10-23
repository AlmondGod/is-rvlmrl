import ctypes,errno,hashlib,json,os,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
from experiments.cap_rl import task_assembly as task

BASE=Path('/opt/cap-assembly')
MAIN=BASE/'assembly02-evidence'
WARM=BASE/'assembly02-speculative-evidence-v2'

def initialize(expected):
    task.EXPECTED_PROVENANCE=expected
    os.environ['CAP_RL_ASSEMBLY_CACHE']=str(WARM)

def evaluate(seed,text,manifest):
    key=hashlib.sha256(json.dumps(dict(seed=seed,calls=task.parse_program(text),provenance=manifest),sort_keys=True).encode()).hexdigest()
    if (MAIN/key).exists():return dict(seed=seed,evidence_id=key,skipped='already started in main')
    result=task.execute(task.Layout(seed),text)
    return dict(result,seed=seed)

def publish(source,destination):
    temporary=BASE/('cache-publish-'+source.name)
    shutil.copytree(source,temporary)
    libc=ctypes.CDLL(None,use_errno=True)
    function=libc.renameat2
    function.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    status=function(-100,os.fsencode(temporary),-100,os.fsencode(destination),1)
    if status:
        error=ctypes.get_errno();shutil.rmtree(temporary)
        if error==errno.EEXIST:return False
        raise OSError(error,os.strerror(error))
    return True

def main():
    sources=list(MAIN.glob('2056f804*/program.py'));assert len(sources)==1
    source=sources[0];text=source.read_text();manifest=task.provenance()
    seeds=[41004,41006,41007,41008,41003,41005]
    record=dict(scope='Speculative physical evaluation of an already model-sampled program on training layouts only. The optimizer receives its reward only if its own sampled program matches this exact parsed program and seed. No demonstration targets.',source_model_program=str(source),program_sha256=hashlib.sha256(text.encode()).hexdigest(),seeds=seeds,started_unix=time.time(),records=[])
    report=BASE/'assembly02-speculative-v2.json';report.write_text(json.dumps(record,indent=2))
    with ProcessPoolExecutor(4,mp_context=get_context('spawn'),initializer=initialize,initargs=(manifest,)) as pool:
        futures=[pool.submit(evaluate,seed,text,manifest) for seed in seeds]
        for future in as_completed(futures):
            result=future.result()
            if not result.get('skipped'):
                result['published_to_main']=publish(Path(result['evidence']),MAIN/result['evidence_id'])
            record['records'].append(result);report.write_text(json.dumps(record,indent=2))
            print(json.dumps({k:result.get(k) for k in ('seed','success','reward','error','skipped','published_to_main')}),flush=True)
    record['finished_unix']=time.time();report.write_text(json.dumps(record,indent=2))

if __name__=='__main__':main()
