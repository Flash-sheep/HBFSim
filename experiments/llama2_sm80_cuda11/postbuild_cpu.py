#!/usr/bin/env python3
"""Complete local CPU preparation after the framework build, without GPU work."""
import argparse,hashlib,json,os,signal,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--build-attempt',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
w=a.work.resolve();root=Path(__file__).resolve().parents[2];art=root/'artifacts/llama2-layer15-full-validation-v1';out=w/'logs'/('postbuild-'+a.attempt);out.mkdir()
e=art/'framework-cpu';prefix=w/'torch-cu117-env';python=prefix/'bin/python'
env={k:v for k,v in os.environ.items() if not k.startswith(('HBFSIM_','BPFTIME_','CUDA_')) and k not in ['PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH']}
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CCACHE_DISABLE='1',CCACHE_DIR=str(w/'private-ccache'),LD_LIBRARY_PATH='/usr/local/cuda-11.7/lib64:'+str(prefix/'lib'),PATH=str(w/'private-bin')+':'+str(prefix/'bin')+':/usr/bin:/bin')
record=e/('build-'+a.build_attempt+'.json');running=json.loads((e/('build-'+a.build_attempt+'.running.json')).read_text())
plan={'scope':'CPU software postprocessing only; no GPU execution or approval generation','waits_for':str(record),'steps':['install local wheel into the new prefix','verify source/CUDA/ABI and original model byte identities','fixed original-helper CPU forward','collect real embedded torch PTX','build remaining isolated HBF/agent artifacts without running them'],'postprocessing_wall_seconds':1800,'stop_available_memory_bytes':64*2**30,'stop_free_disk_bytes':50*2**30,'build_jobs':2,'GPU_tasks':0,'completion_notification':'NOT_CONFIGURED','DONE':str(out/'DONE'),'FAILED':str(out/'FAILED')}
(e/('postbuild-'+a.attempt+'.plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
(e/('postbuild-'+a.attempt+'.running.json')).write_text(json.dumps({'pid':os.getpid(),'waits_for_worker':running,'completion_notification':'NOT_CONFIGURED','DONE':str(out/'DONE'),'FAILED':str(out/'FAILED')},indent=2)+'\n')
rows=[];started=None;result={}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(cmd,label,cwd=None):
 if time.monotonic()-started>=1800:raise TimeoutError('postprocessing watchdog')
 available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 if available<64*2**30 or shutil.disk_usage(w).free<50*2**30:raise RuntimeError('resource reserve')
 with (out/(label+'.stdout')).open('x') as stdout,(out/(label+'.stderr')).open('x') as stderr:
  proc=subprocess.Popen(cmd,cwd=cwd or w,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
  while proc.poll() is None:
   elapsed=time.monotonic()-started
   available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
   if elapsed>1800 or available<64*2**30 or shutil.disk_usage(w).free<50*2**30:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise RuntimeError('postprocessing resource/watchdog stop')
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
 rows.append({'label':label,'command':cmd,'returncode':proc.returncode})
 if proc.returncode:raise RuntimeError(label+' failed; original stdout/stderr retained')
try:
 deadline=(e/('build-'+a.build_attempt+'.plan.json')).stat().st_mtime+14400+120
 while not record.exists():
  if time.time()>deadline:raise TimeoutError('framework build completion not received before its watchdog margin')
  if not Path('/proc/'+str(running['supervisor_pid'])).exists():raise RuntimeError('framework supervisor disappeared before a completion receipt')
  time.sleep(10)
 build=json.loads(record.read_text());assert build['status']=='PASS','framework build did not pass'
 started=time.monotonic()
 wheels=list((w/'pytorch-src/dist').glob('torch-2.4.0+cu117.sm80ptx-*.whl'));assert len(wheels)==1
 wheel=wheels[0];result['wheel']={'path':str(wheel),'sha256':sha(wheel),'bytes':wheel.stat().st_size}
 run([str(python),'-m','pip','install','--no-index','--no-deps',str(wheel)],'install-local-wheel')
 run([str(python),'-m','pip','check'],'pip-check')
 forward=out/'cpu-forward';forward.mkdir();(forward/'evidence').mkdir()
 shutil.copytree(art/'frozen',forward/'frozen')
 shutil.copyfile(art/'cpu_workload_check.py',forward/'cpu_workload_check.py')
 child=out/'verify_and_forward.py'
 child.write_text('''import hashlib,json,runpy,sys
from pathlib import Path
import torch
from safetensors import safe_open
art=Path(sys.argv[1]);forward=Path(sys.argv[2]);out=Path(sys.argv[3])
assert torch.version.git_version=='e4ee3be4063b7c430974252fdf7db42273388d86'
assert torch.version.cuda=='11.7' and not torch._C._GLIBCXX_USE_CXX11_ABI
assert not torch.cuda.is_initialized()
identity=json.loads((art/'evidence/INPUT_IDENTITY.json').read_text())
for row in identity['files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
for row in identity['tensors']:
 with safe_open(row['shard'],framework='pt',device='cpu') as f:
  value=f.get_tensor(row['name']);payload=value.contiguous().view(torch.uint8).numpy().tobytes()
  assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256'],row['name']
runpy.run_path(str(forward/'cpu_workload_check.py'),run_name='__main__')
assert not torch.cuda.is_initialized()
result={'status':'PASS','torch':torch.__version__,'torch_git':torch.version.git_version,'cuda_build':torch.version.cuda,'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),'CUDA_initialized':False,'GPU_tasks_started':0,'torch_cuda_library':str(Path(torch.__file__).parent/'lib/libtorch_cuda.so'),'CPU_forward_receipt':str(forward/'evidence/CPU_FULL_FORWARD.json')}
(out/'ENVIRONMENT_AND_FORWARD.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
''')
 run([str(python),'-B',str(child),str(art),str(forward),str(out)],'environment-and-forward',forward)
 info=json.loads((out/'ENVIRONMENT_AND_FORWARD.json').read_text());result['environment']=info
 cuda_lib=Path(info['torch_cuda_library']);assert prefix.resolve() in cuda_lib.resolve().parents
 collection=out/'torch-cuda-original-collection'
 run([str(python),'-B',str(root/'tools/native_ptx_reproduction/collector/collect_native_ptx.py'),'--container',str(cuda_lib),'--cuobjdump','/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/cuda/bin/cuobjdump','--output',str(collection)],'collect-torch-ptx')
 manifest=collection/'native-ptx-manifest.json';data=json.loads(manifest.read_text())
 assert data['collector_status']=='READY' and data['members'] and all(x['target']=='sm_80' for x in data['members'])
 result['torch_ptx']={'manifest':str(manifest),'sha256':sha(manifest),'members':len(data['members']),'targets':['sm_80'],'scope':'static source collection; runtime selected kernels and HBF coverage not validated'}
 cmake='/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/build-tools/cmake/data/bin/cmake';hbf=w/'cudart11-v1/build-gate-on'
 run([cmake,'--build',str(hbf),'--parallel','2','--target','hbfsim','hbfsimd','ptxpass_hbf','ptxpass_hbf_plugin','ptxpass_hbf_aggregate_plugin','hbfsim_bpftime_attach_loader','hbfsim_coverage_probe'],'remaining-hbf-artifacts')
 run(['/usr/bin/ninja','-C',str(w/'cudart11-v1/build-agent-on'),'-j1','bpftime-syscall-server'],'remaining-agent-server')
 result.update(status='CPU_PREPARATION_PASS_GPU_NOT_RUN',positive_real_PTX_binding='NOT_VALIDATED',GPU_tasks_started=0,steps=rows,wall_seconds=time.monotonic()-started,limits='GPU launcher, actual selected-entry staging/binding, GPU numerical and nine-weight HBF coverage still require preparation and a new approved preflight')
except Exception as exc:
 result.update(status='FAILED',error=str(exc),GPU_tasks_started=0,steps=rows)
path=e/('postbuild-'+a.attempt+'.json');assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');(out/('DONE' if result['status']=='CPU_PREPARATION_PASS_GPU_NOT_RUN' else 'FAILED')).write_text(json.dumps(result)+'\n');print(json.dumps(result));raise SystemExit(0 if result['status']=='CPU_PREPARATION_PASS_GPU_NOT_RUN' else 1)
