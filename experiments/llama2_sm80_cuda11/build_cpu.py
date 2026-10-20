#!/usr/bin/env python3
"""Build isolated adapters; never execute a CUDA workload."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('profile',choices=['gate-on','gate-off','agent-on','agent-off']);p.add_argument('--work',type=Path,required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
w=a.work.resolve();parent=w.parent;root=Path(__file__).resolve().parents[2]
e=root/'artifacts/llama2-layer15-full-validation-v1/cuda11-compat';e.mkdir(exist_ok=True)
log=w/'logs';log.mkdir(exist_ok=True);key=a.profile+'-'+a.attempt
receipt=e/(key+'.json');assert not receipt.exists()
build=w/('build-'+a.profile);cmake='/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/build-tools/cmake/data/bin/cmake'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CCACHE_DISABLE='1',CCACHE_DIR=str(w/'ccache'),PATH=str(parent/'private-bin')+':'+str(parent/'agent-deps/bin')+':/usr/bin:/bin')
feature='ON' if a.profile.endswith('-on') else 'OFF'
base=[cmake,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DHBFSIM_EXPERIMENTAL_CUDART11='+feature,'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
if a.profile.startswith('gate-'):
 cuda='/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/cuda'
 src=w/'hbfsim-src'
 config=base+['-S',str(src),'-B',str(build),'-DHBFSIM_ENABLE_CUDA=ON','-DHBFSIM_ENABLE_MQSIM=ON','-DHBFSIM_ENABLE_TIMING_FUTURES=OFF','-DHBFSIM_ENABLE_LLM_TESTS=OFF','-DHBFSIM_GATE_TEST_HOOKS=ON','-DHBFSIM_GATE_CUDART_VERSION_MAP='+str(src/'cmake/cudart-versions.map'),'-DCMAKE_CUDA_ARCHITECTURES=80','-DCUDAToolkit_ROOT='+cuda,'-DCMAKE_CUDA_COMPILER='+cuda+'/bin/nvcc','-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11','-DCMAKE_CXX_COMPILER=/usr/bin/g++-11']
 commands=[config,[cmake,'--build',str(build),'--parallel','2','--target','hbfsim_launch_gate','fake_cuda_lookup']]
else:
 src=w/'agent-src'
 old=json.loads((root/'artifacts/llama2-layer15-full-validation-v1/evidence/AGENT_CONFIGURE_V2.json').read_text())['command']
 config=[cmake]+[x for x in old[1:] if x!='-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
 config[config.index('-S')+1]=str(src);config[config.index('-B')+1]=str(build)
 config+=['-DHBFSIM_EXPERIMENTAL_CUDART11='+feature,'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
 commands=[config,['/usr/bin/ninja','-C',str(build),'-j1','copy_headers'],['/usr/bin/ninja','-C',str(build),'-j1','bpftime-agent']]
plan={'scope':'approved isolated build + fixed CPU validation; no GPU','commands':commands,'jobs':'gate2 OR agent Ninja1/nested make2, serial build profiles','watchdog_seconds':1800,'stop_free_disk_bytes':50*2**30,'stop_available_memory_bytes':32*2**30,'GPU_tasks':0,'DONE':str(log/(key+'.DONE')),'FAILED':str(log/(key+'.FAILED'))}
(e/(key+'.plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
started=time.monotonic();rows=[];reason=None
for i,cmd in enumerate(commands):
 with (log/(key+f'.{i}.stdout')).open('x') as out,(log/(key+f'.{i}.stderr')).open('x') as err:
  proc=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
  (e/(key+'.running.json')).write_text(json.dumps({'pid':proc.pid,'supervisor_pid':os.getpid(),'step':i,'command':cmd,'completion_notification':'NOT_CONFIGURED'},indent=2)+'\n')
  while proc.poll() is None:
   available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
   if time.monotonic()-started>1800:reason='wall watchdog'
   if shutil.disk_usage(w).free<50*2**30:reason='disk reserve'
   if available<32*2**30:reason='memory reserve'
   if reason:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    break
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
 rows.append({'step':i,'returncode':proc.returncode})
 if proc.returncode or reason:break
status='PASS' if not reason and len(rows)==len(commands) and not any(x['returncode'] for x in rows) else 'FAILED'
artifacts={}
for rel in ['libhbfsim_launch_gate.so','runtime/agent/libbpftime-agent.so']:
 path=build/rel
 if path.exists():artifacts[rel]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
r={'status':status,'profile':a.profile,'feature':feature,'steps':rows,'stop_reason':reason,'wall_seconds':time.monotonic()-started,'GPU_tasks_started':0,'artifacts':artifacts}
receipt.write_text(json.dumps(r,indent=2)+'\n');(log/(key+('.DONE' if status=='PASS' else '.FAILED'))).write_text(json.dumps(r)+'\n')
print(json.dumps(r));raise SystemExit(0 if status=='PASS' else 1)
