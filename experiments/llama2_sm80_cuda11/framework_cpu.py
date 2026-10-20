#!/usr/bin/env python3
"""Construct the exact Torch source with sm80 PTX, CPU processes only."""
import argparse,hashlib,json,os,re,signal,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['configure','build']);p.add_argument('--work',required=True,type=Path);p.add_argument('--attempt',required=True);a=p.parse_args()
w=a.work.resolve();root=Path(__file__).resolve().parents[2];src=w/'pytorch-src';prefix=w/'torch-cu117-env'
e=root/'artifacts/llama2-layer15-full-validation-v1/framework-cpu';e.mkdir(exist_ok=True);key=a.mode+'-'+a.attempt
assert not (e/(key+'.json')).exists()
git=lambda *args:subprocess.check_output(['/usr/bin/git','-C',str(src),*args],text=True).strip()
assert git('rev-parse','HEAD')=='e4ee3be4063b7c430974252fdf7db42273388d86'
submodules=subprocess.check_output(['/usr/bin/git','-C',str(src),'submodule','status','--recursive'],text=True)
assert all(x.startswith(' ') for x in submodules.splitlines()),'All submodules must be at their exact gitlinks before configure/build'
assert not git('diff','--name-only'), 'Framework source must remain unchanged'
env={k:v for k,v in os.environ.items() if not k.startswith(('CUDA_','HBFSIM_','BPFTIME_','CMAKE_','USE_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDACXX','CC','CXX']}
options=dict(CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-11.7',CUDA_PATH='/usr/local/cuda-11.7',CUDAToolkit_ROOT='/usr/local/cuda-11.7',CUDA_TOOLKIT_ROOT_DIR='/usr/local/cuda-11.7',CUDACXX='/usr/local/cuda-11.7/bin/nvcc',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',CUDAHOSTCXX='/usr/bin/g++-11',
 TORCH_CUDA_ARCH_LIST='8.0+PTX',MAX_JOBS='4',CMAKE_BUILD_PARALLEL_LEVEL='4',CMAKE_BUILD_TYPE='Release',DEBUG='0',REL_WITH_DEB_INFO='0',USE_CUDA='1',USE_CUDNN='0',USE_CUSPARSELT='0',USE_DISTRIBUTED='0',USE_NCCL='0',USE_GLOO='0',USE_MPI='0',USE_TENSORPIPE='0',USE_KINETO='0',USE_FLASH_ATTENTION='0',USE_MEM_EFF_ATTENTION='0',USE_MKLDNN='0',USE_MKL='0',USE_NNPACK='0',USE_QNNPACK='0',USE_PYTORCH_QNNPACK='0',USE_XNNPACK='0',USE_FBGEMM='0',BUILD_TEST='0',BUILD_CAFFE2='0',BLAS='OpenBLAS',USE_CCACHE='0',CCACHE_DISABLE='1',CCACHE_DIR=str(w/'private-ccache'),
 _GLIBCXX_USE_CXX11_ABI='0',PYTORCH_BUILD_VERSION='2.4.0+cu117.sm80ptx',PYTORCH_BUILD_NUMBER='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 PATH='/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/build-tools/cmake/data/bin:'+str(prefix/'bin')+':/usr/local/cuda-11.7/bin:/usr/bin:/bin',LD_LIBRARY_PATH='/usr/local/cuda-11.7/lib64:'+str(prefix/'lib'))
if a.mode=='configure':options.update(MAX_JOBS='1',CMAKE_BUILD_PARALLEL_LEVEL='1')
env.update(options)
cores=sorted(os.sched_getaffinity(0))[-int(options['MAX_JOBS']):]
plan={'cpu_affinity':cores,'scope':'isolated framework software construction; not a scientific experiment or GPU launch','source_git':git('rev-parse','HEAD'),'submodules':submodules,'environment':options,
 'reason':'The installed torch wheel has no embedded PTX; preserve original operator source and git identity while retaining compute80 PTX.',
 'feature_scope':'Original eager workload; optional distributed/profiler/CPU optimizers and fused attention omitted, no substitute operator implementation.',
 'jobs':int(options['MAX_JOBS']),'build_watchdog_seconds':14400,'configure_watchdog_seconds':1800,'stop_free_disk_bytes':50*2**30,'stop_host_available_bytes':64*2**30,'stop_process_group_rss_bytes':96*2**30,'stop_task_tree_bytes':75*2**30,'GPU_tasks':0,
 'build_start_condition':'configure succeeds; CMake cache and generated NVCC flags show CUDA11.7 and both sm80/compute80; other compilation finishes first',
 'limitations':'Custom build, not wheel-byte equivalence. Native GPU kernel identity, PTX coverage, numeric equivalence and HBF behavior remain unmeasured.'}
(e/(key+'.plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
def re_cache(text,name):
 match=re.search(r'^'+re.escape(name)+r':[^=]*=(.*)$',text,re.M)
 return match.group(1).strip() if match else None
if a.mode=='build':
 cache=(src/'build/CMakeCache.txt').read_text()
 assert re_cache(cache,'CUDA_VERSION')=='11.7'
 assert re_cache(cache,'TORCH_CUDA_ARCH_LIST')=='8.0+PTX'
 ninja=(src/'build/build.ninja').read_text()
 assert 'arch=compute_80,code=sm_80' in ninja and 'arch=compute_80,code=compute_80' in ninja
 # Keep compilation serial with the other adapter build profiles.
 for profile in ['agent-on','agent-off','gate-on','gate-off']:
  result=root/'artifacts/llama2-layer15-full-validation-v1/cuda11-compat'/(profile+'-v1.json')
  assert json.loads(result.read_text())['status']=='PASS'
# Avoid importing torch; setup.py --cmake-only does configuration only.
cmd=[str(prefix/'bin/python'),'-B','setup.py']+(['--cmake-only'] if a.mode=='configure' else [])+['bdist_wheel']
def restrict_child():
 os.sched_setaffinity(0,cores)
 os.nice(10)
start=time.monotonic();reason=None;peak=0;last_size_check=0;size=0
with (w/'logs'/('torch-'+key+'.stdout')).open('x') as out,(w/'logs'/('torch-'+key+'.stderr')).open('x') as err:
 proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=restrict_child)
 (e/(key+'.running.json')).write_text(json.dumps({'pid':proc.pid,'supervisor_pid':os.getpid(),'command':cmd,'completion_notification':'NOT_CONFIGURED','DONE':str(w/'logs'/('torch-'+key+'.DONE')),'FAILED':str(w/'logs'/('torch-'+key+'.FAILED'))},indent=2)+'\n')
 while proc.poll() is None:
  elapsed=time.monotonic()-start
  available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  rss=0
  for task in Path('/proc').iterdir():
   if not task.name.isdecimal():continue
   try:
    stat=(task/'stat').read_text();fields=stat[stat.rindex(')')+2:].split()
    if int(fields[2])==proc.pid:rss+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
   except (OSError,ValueError,IndexError):pass
  peak=max(peak,rss)
  if elapsed-last_size_check>60 or last_size_check==0:
   du=subprocess.run(['/usr/bin/du','-s','-B1',str(w)],capture_output=True,text=True)
   if du.returncode==0:size=int(du.stdout.split()[0])
   last_size_check=elapsed
  if available<64*2**30:reason='host memory reserve'
  if rss>96*2**30:reason='process-group RSS'
  if shutil.disk_usage(w).free<50*2**30 or size>75*2**30:reason='disk reserve/task size'
  if elapsed>(1800 if a.mode=='configure' else 14400):reason='wall watchdog'
  if reason:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   break
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:pass
status='PASS' if proc.returncode==0 and reason is None else 'FAILED'
r={'status':status,'mode':a.mode,'returncode':proc.returncode,'stop_reason':reason,'wall_seconds':time.monotonic()-start,'peak_process_group_rss_bytes':peak,'last_task_tree_bytes':size,'GPU_tasks_started':0,'command':cmd}
(e/(key+'.json')).write_text(json.dumps(r,indent=2)+'\n');(w/'logs'/('torch-'+key+('.DONE' if status=='PASS' else '.FAILED'))).write_text(json.dumps(r)+'\n');print(json.dumps(r));raise SystemExit(0 if status=='PASS' else 1)
