#!/usr/bin/env python3
"""Serialize a cold-provider regression/fix with the owned Torch CPU build."""
import datetime,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];w=Path('/var/tmp/hbfsim-yjl/llama2-layer15-full-v1');work=w/'cudart11-v1';e=ROOT/'artifacts/llama2-layer15-full-validation-v1/cuda11-compat';out=work/'logs/provider-fix-v2';out.mkdir()
runinfo=json.loads((ROOT/'artifacts/llama2-layer15-full-validation-v1/framework-cpu/build-v1.running.json').read_text());worker=int(runinfo['pid']);paused=False;rows=[];started=time.monotonic();result={}
def interrupted(*_):raise RuntimeError('provider verification interrupted')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
def ticks(pid):return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
worker_ticks=None
def run(cmd,label,expected=0):
 with (out/(label+'.stdout')).open('x') as stdout,(out/(label+'.stderr')).open('x') as stderr:
  r=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=600)
 rows.append({'label':label,'returncode':r.returncode,'command':cmd})
 (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
 if r.returncode!=expected:raise RuntimeError(label+' returned '+str(r.returncode))
try:
 if Path('/proc/'+str(worker)).exists() and not (ROOT/'artifacts/llama2-layer15-full-validation-v1/framework-cpu/build-v1.json').exists():
  assert os.getpgid(worker)==worker and Path('/proc/'+str(worker)+'/cwd').resolve()==w/'pytorch-src'
  assert b'setup.py' in Path('/proc/'+str(worker)+'/cmdline').read_bytes()
  worker_ticks=ticks(worker);os.killpg(worker,signal.SIGSTOP);paused=True
  (out/'TORCH_BUILD_PAUSED.json').write_text(json.dumps({'pid':worker,'start_ticks':worker_ticks,'reason':'serialize fixed CPU adapter verification with the owned source build','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
 run(['python3','-B',str(ROOT/'experiments/llama2_sm80_cuda11/test_cpu.py'),'registry','--cold-start','--work',str(work),'--attempt','cold-before-v1'],'cold-before',expected=1)
 before=json.loads((e/'registry-tests-cold-before-v1.json').read_text());assert 'CUDA11 registration missing after late application provider load' in before['error'],before.get('error')
 original=work/'build-agent-on/runtime/agent/libbpftime-agent.so';saved=work/'agent-on-before-provider-fix.so';assert not saved.exists();shutil.copyfile(original,saved)
 result['previous_agent']={'path':str(saved),'sha256':hashlib.sha256(saved.read_bytes()).hexdigest()}
 run(['python3','-B',str(ROOT/'experiments/llama2_sm80_cuda11/apply_provider_fix.py'),'--source',str(work/'agent-src'),'--patch-output',str(ROOT/'experiments/llama2_sm80_cuda11/patches/agent-cuda11-provider.patch')],'apply-isolated-fix')
 run(['python3','-B',str(ROOT/'experiments/llama2_sm80_cuda11/build_cpu.py'),'agent-on','--work',str(work),'--attempt','provider-v2'],'build-agent-on')
 run(['python3','-B',str(ROOT/'experiments/llama2_sm80_cuda11/test_cpu.py'),'registry','--cold-start','--work',str(work),'--attempt','cold-on-after-v2'],'cold-on-after')
 run(['python3','-B',str(ROOT/'experiments/llama2_sm80_cuda11/test_cpu.py'),'registry','--work',str(work),'--attempt','provider-after-v2'],'registry-after')
 run(['python3','-B',str(ROOT/'experiments/llama2_sm80_cuda11/build_cpu.py'),'agent-off','--work',str(work),'--attempt','provider-v2'],'build-agent-off')
 run(['python3','-B',str(ROOT/'experiments/llama2_sm80_cuda11/test_cpu.py'),'registry','--cold-start','--agent-profile','agent-off','--work',str(work),'--attempt','cold-off-after-v2'],'cold-off-after')
 # A renamed CUDA12 fixture must not satisfy the CUDA11 version-domain lookup.
 good=work/'tests-registry-cold-on-after-v2';bad=out/'wrong-provider';bad.mkdir()
 for name in ['libcuda.so.1','libcudart.so.12','libcudart.so.13']:(bad/name).symlink_to(good/name)
 (bad/'libcudart.so.11.0').symlink_to(good/'libcudart.so.12')
 env={k:v for k,v in os.environ.items() if not k.startswith(('HBFSIM_','BPFTIME_','CUDA_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']};env.update(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(bad)+':'+str(w/'agent-deps/lib'),CPU_FAKE_DIR=str(bad))
 rejected=subprocess.run([str(good/'registry_cpu')],env=env,cwd=out,capture_output=True,text=True,timeout=60)
 (out/'wrong-provider.stdout').write_text(rejected.stdout);(out/'wrong-provider.stderr').write_text(rejected.stderr)
 assert rejected.returncode==1 and 'Exact CUDA11 runtime provider unavailable' in rejected.stderr
 rows.append({'label':'renamed-CUDA12-provider-rejected','returncode':rejected.returncode,'expected_failure':True})
 def exports(path):
  text=subprocess.check_output(['readelf','--dyn-syms','--wide',str(path)],text=True)
  return {x.split()[7] for x in text.splitlines() if len(x.split())>=8 and x.split()[6]!='UND' and x.split()[4] in ['GLOBAL','WEAK']}
 assert exports(work/'build-agent-off/runtime/agent/libbpftime-agent.so')==exports(w/'agent-build-v2/runtime/agent/libbpftime-agent.so')
 assert exports(original)==exports(saved)
 result.update(status='PASS',scope='cold CUDA11 provider acquisition and exact-version registration; default-off and CUDA12 regression retained',renamed_provider_rejected=True,exports_preserved=True,steps=rows,GPU_tasks_started=0)
except Exception as exc:result.update(status='FAILED',error=str(exc),steps=rows,GPU_tasks_started=0)
finally:
 if paused:
  try:
   if ticks(worker)==worker_ticks and os.getpgid(worker)==worker:os.killpg(worker,signal.SIGCONT)
  except FileNotFoundError:pass
  result['torch_build_resumed']=True
 result['wall_seconds']=time.monotonic()-started
 path=e/'PROVIDER_FIX_V2.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');(out/('DONE' if result['status']=='PASS' else 'FAILED')).write_text(json.dumps(result)+'\n')
 print(json.dumps(result))
raise SystemExit(0 if result['status']=='PASS' else 1)
