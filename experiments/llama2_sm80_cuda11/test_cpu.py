#!/usr/bin/env python3
"""Fixed CPU-only adapter checks; mock libcuda prevents GPU access."""
import argparse,ctypes,hashlib,json,os,re,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('mode',choices=['gate','registry']);p.add_argument('--work',type=Path,required=True);p.add_argument('--attempt',required=True);p.add_argument('--cold-start',action='store_true');p.add_argument('--agent-profile',choices=['agent-on','agent-off'],default='agent-on');a=p.parse_args()
w=a.work.resolve();out=w/('tests-'+a.mode+'-'+a.attempt);out.mkdir()
env={k:v for k,v in os.environ.items() if not k.startswith(('HBFSIM_','BPFTIME_','CUDA_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CCACHE_DISABLE='1',CCACHE_DIR=str(w/'ccache'))
commands=[]
def run(cmd,**kwargs):
 commands.append(cmd)
 kwargs.setdefault('cwd',out)
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=180,env=env,**kwargs)
 if r.returncode:raise RuntimeError(repr(cmd)+'\n'+r.stdout[-6000:]+'\n'+r.stderr[-6000:])
 return r

def symbols(path,pattern):
 r=run(['/usr/bin/nm','-D','--undefined-only',str(path)])
 return {line.split()[-1].split('@')[0] for line in r.stdout.splitlines() if line.split() and re.fullmatch(pattern,line.split()[-1].split('@')[0])}
# The driver fixture implements every dynamically required cu* symbol, all inert.
base_agent=w.parent/'agent-build-v2/runtime/agent/libbpftime-agent.so'
gate=w/'build-gate-on/libhbfsim_launch_gate.so'
imports=symbols(base_agent,r'cu[A-Z][A-Za-z0-9_]+')|symbols(gate,r'cu[A-Z][A-Za-z0-9_]+')
code='static int launches; int fake_driver_launches(void){return launches;}\n'
for name in sorted(imports|{'cuInit','cuKernelGetFunction'}):
 if name=='cuLaunchKernel':code+='int cuLaunchKernel(void){++launches;return 999;}\n'
 elif name=='cuInit':code+='int cuInit(unsigned flags){(void)flags;return 0;}\n'
 elif name=='cuKernelGetFunction':code+='int cuKernelGetFunction(void**out,void*k){if(out)*out=k;return 0;}\n'
 else:code+='int '+name+'(void){return 999;}\n'
(out/'fake_cuda.c').write_text(code)
run(['/usr/bin/gcc','-shared','-fPIC','-O0','-Wl,-soname,libcuda.so.1',str(out/'fake_cuda.c'),'-o',str(out/'libcuda.so.1')])
rt_imports=symbols(base_agent,r'cuda[A-Z][A-Za-z0-9_]+')|symbols(gate,r'cuda[A-Z][A-Za-z0-9_]+')
for major,version in [(11,'11.0'),(12,'12'),(13,'13')]:
 name='libcudart.so.'+version
 code='''#include <stddef.h>
#include <stdint.h>
typedef struct {unsigned x,y,z;} Dim;
static int pub,priv,ret;
void fake_reset(int r){pub=priv=0;ret=r;}
int fake_public_calls(void){return pub;}
int fake_private_calls(void){return priv;}
uint64_t bpftime_nv_strict_bridge_capabilities_v1(void){return 3;}
__attribute__((noinline)) int f_public(const void*f,Dim g,Dim b,void**a,size_t s,void*st){++pub;return ret;}
__attribute__((noinline)) int f_private(void*f,Dim g,Dim b,void**a,size_t s,void*st){++priv;return ret;}
int f_get(void**out,const void*f){if(out)*out=0;return 1;}
__attribute__((noinline)) void** f_register(void*f){volatile uintptr_t v=(uintptr_t)f;v+=1;return (void**)0x4100;}
__attribute__((noinline)) void f_unregister(void**h){volatile uintptr_t v=(uintptr_t)h;v+=1;}
__attribute__((noinline)) void f_function(void**h,const void*f,const void*d,const char*n){volatile uintptr_t v=(uintptr_t)h;v+=(uintptr_t)f;}
__attribute__((noinline)) void f_var(void**h,const void*f,const void*d,const char*n){volatile uintptr_t v=(uintptr_t)h;v+=(uintptr_t)f;}
__attribute__((noinline)) void f_end(void**h){volatile uintptr_t v=(uintptr_t)h;v+=1;}
'''
 alias={'f_public':'cudaLaunchKernel','f_get':'cudaGetFuncBySymbol','f_register':'__cudaRegisterFatBinary','f_unregister':'__cudaUnregisterFatBinary','f_function':'__cudaRegisterFunction','f_var':'__cudaRegisterVar','f_end':'__cudaRegisterFatBinaryEnd'}
 if major!=11:alias['f_private']='__cudaLaunchKernel'
 for source,target in alias.items():code+='__asm__(".symver '+source+','+target+'@@'+name+'");\n'
 for symbol in sorted(rt_imports-{'cudaGetFuncBySymbol','cudaLaunchKernel'}):
  if symbol=='cudaGetErrorString':code+='const char* cudaGetErrorString(int e){return "CPU fake cudart";}\n'
  else:code+='int '+symbol+'(void){return 999;}\n'
 (out/(name+'.c')).write_text(code);(out/(name+'.map')).write_text(name+' {};\n')
 run(['/usr/bin/gcc','-shared','-fPIC','-O0','-Wl,-Bsymbolic','-Wl,-soname,'+name,'-Wl,--version-script='+str(out/(name+'.map')),str(out/(name+'.c')),'-o',str(out/name)])
env['LD_LIBRARY_PATH']=str(out)+':'+str(w.parent/'agent-deps/lib')
env['CPU_FAKE_DIR']=str(out)
env['CPU_GATE']=str(gate)
started=time.monotonic();result={'scope':'CPU software fixtures with mock CUDA DSOs; no numerical GPU or HBF timing validation','GPU_tasks_started':0}
try:
 if a.mode=='gate':
  # Original 12/13 test unchanged against the default-off port.
  old=run([sys.executable,str(ROOT/'tests/integration/reproduction_gate_cpu/run.py'),'--gate',str(w/'build-gate-off/libhbfsim_launch_gate.so'),'--receipt',str(out/'legacy-default-off.json')]);result['legacy_default_off']=old.stdout
  # Same 12/13 cases on the enabled binary, adjusting only which newly supported domain is invalid.
  original=(ROOT/'tests/integration/reproduction_gate_cpu/run_gate_cpu.py').read_text()
  assert 'begin(function, 11, 1) == 0' in original
  modified=original.replace('begin(function, 11, 1) == 0','begin(function, 14, 1) == 0')
  # Reuse the pre-existing same-DSO fake12/private13 fixture for this unchanged body.
  legacy=out/'legacy-on';legacy.mkdir()
  old_fixture=ROOT/'tests/integration/reproduction_gate_cpu'
  for f in ['fake_cuda.c','fake_cudart.c','cudart.map','run.py']:(legacy/f).write_text((old_fixture/f).read_text())
  (legacy/'run_gate_cpu.py').write_text(modified)
  v=run([sys.executable,str(legacy/'run.py'),'--gate',str(gate),'--receipt',str(out/'legacy-enabled.json')]);result['legacy_enabled']=v.stdout
  script=out/'test11.py'
  script.write_text('''import ctypes as C,os,json,threading
from pathlib import Path
f=Path(os.environ['CPU_FAKE_DIR']);gatepath=Path(os.environ['CPU_GATE'])
libs={m:C.CDLL(str(f/n),mode=os.RTLD_GLOBAL|os.RTLD_NOW) for m,n in [(11,'libcudart.so.11.0'),(12,'libcudart.so.12'),(13,'libcudart.so.13')]}
g=C.CDLL(str(gatepath),mode=os.RTLD_GLOBAL|os.RTLD_NOW);libc=C.CDLL(None)
libc.dlvsym.argtypes=[C.c_void_p,C.c_char_p,C.c_char_p];libc.dlvsym.restype=C.c_void_p
class Dim(C.Structure):_fields_=[('x',C.c_uint),('y',C.c_uint),('z',C.c_uint)]
Launch=C.CFUNCTYPE(C.c_int,C.c_void_p,Dim,Dim,C.c_void_p,C.c_size_t,C.c_void_p)
addr=libc.dlvsym(g._handle,b'cudaLaunchKernel',b'libcudart.so.11.0');assert addr
launch=Launch(addr);one=Dim(1,1,1);fn=0x123456
begin=g.hbfsim_strict_original_begin_v1;begin.argtypes=[C.c_void_p,C.c_int,C.c_int];begin.restype=C.c_uint64
consume=g.hbfsim_strict_original_consume_v1;consume.argtypes=[C.c_uint64,C.c_void_p,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)];consume.restype=C.c_int
rows=[]
def done(t,domain=11,func=fn,result=0):
 d,r=C.c_int(-1),C.c_int(-1);return consume(t,func,domain,1,result,C.byref(d),C.byref(r)),d.value,r.value
os.environ['HBFSIM_INSTRUMENTATION_POLICY']='strict'
for value in [0,801]:
 for lib in libs.values():lib.fake_reset(value)
 t=begin(fn,11,1);assert t
 actual=launch(fn,one,one,None,0,None);assert actual==value
 assert done(t,result=value)==(1,1,value) and done(t,result=value)[0]==0
 assert libs[11].fake_public_calls()==1 and libs[12].fake_public_calls()==0 and libs[13].fake_public_calls()==0
 rows.append({'case':'exact11_forward_and_return','result':actual,'cross_domain_calls':0})
for mode in ['wrong_domain','wrong_function','cross_thread']:
 libs[11].fake_reset(0);t=begin(fn,11,1);assert launch(fn,one,one,None,0,None)==0
 if mode=='wrong_domain':assert done(t,domain=12)[0]==0 and done(t)[0]==0
 if mode=='wrong_function':assert done(t,func=fn+1)[0]==0 and done(t)[0]==0
 if mode=='cross_thread':
  out=[];th=threading.Thread(target=lambda:out.append(done(t)));th.start();th.join();assert out[0][0]==0 and done(t)==(1,1,0)
 rows.append({'case':mode,'status':'PASS'})
assert begin(fn,11,2)==0 and begin(fn,14,1)==0
for name in [b'cudaLaunchKernel_ptsz',b'cudaLaunchCooperativeKernel',b'cudaLaunchCooperativeKernel_ptsz']:
 addr=libc.dlvsym(g._handle,name,b'libcudart.so.11.0');assert addr
 libs[11].fake_reset(0);assert Launch(addr)(fn,one,one,None,0,None)==801 and libs[11].fake_public_calls()==0
 rows.append({'case':'unreviewed11_rejected','api':name.decode()})
assert not libc.dlvsym(g._handle,b'__cudaLaunchKernel',b'libcudart.so.11.0')
os.environ.pop('HBFSIM_INSTRUMENTATION_POLICY');libs[11].fake_reset(0)
assert launch(fn,one,one,None,0,None)==801 and libs[11].fake_public_calls()==0
maps=Path('/proc/self/maps').read_text();assert all(str(f) in row for row in maps.splitlines() if '/libcuda.so' in row)
assert not any('/dev/nvidia' in os.readlink(x) for x in Path('/proc/self/fd').iterdir() if x.exists())
print(json.dumps({'status':'PASS','rows':rows,'hardware_gpu_used':False}))
''')
  env.update(CPU_FAKE_DIR=str(out),CPU_GATE=str(gate))
  r=run([sys.executable,str(script)]);(out/'cuda11.stdout').write_text(r.stdout);(out/'cuda11.stderr').write_text(r.stderr);result['cuda11']=json.loads(r.stdout)
  # Default-off must expose no CUDA11 launch alias.
  r=run(['/usr/bin/readelf','--dyn-syms','--wide',str(w/'build-gate-off/libhbfsim_launch_gate.so')]);assert 'cudaLaunchKernel@libcudart.so.11.0' not in r.stdout
 else:
  b=w/('build-'+a.agent_profile);cc=json.loads((b/'compile_commands.json').read_text())
  c=next(x for x in cc if x['file'].endswith('/nv_attach_impl.cpp'))
  argv=shlex.split(c['command']);argv=[x for x in argv if x!='ccache'];obj=out/'registry_cpu.o'
  # Replace only the original input/output and dependency sidecars for this external fixture TU.
  for option in ['-MF','-MT']:
   if option in argv:i=argv.index(option);del argv[i:i+2]
  argv=[x for x in argv if x not in ['-MD','-DNDEBUG']]
  argv[argv.index('-o')+1]=str(obj);argv[argv.index('-c')+1]=str(HERE/('tests/cold_provider_cpu.cpp' if a.cold_start else 'tests/registry_cpu.cpp'))
  run(argv,cwd=b)
  cmd=run(['/usr/bin/ninja','-C',str(b),'-t','commands','bpftime-agent']).stdout.splitlines()[-1]
  assert cmd.startswith(': && ') and cmd.endswith(' && :')
  link=shlex.split(cmd[5:-5]);link=[x for x in link if x not in ['ccache','-shared'] and not x.startswith(('-Wl,--version-script=','-Wl,-soname,')) and not x.endswith('/agent.cpp.o')]
  link[link.index('-o')+1]=str(out/'registry_cpu');link.insert(link.index('-o'),str(obj))
  # The agent version script hides static NVPTXCompiler symbols. Match that visibility
  # in this external executable, so its compiler and the compiler plugin own separate state.
  link.append('-Wl,--exclude-libs,libnvptxcompiler_static.a')
  run(link,cwd=b)
  r=run([str(out/'registry_cpu'),str(out/'manifest.json')],cwd=b)
  (out/'registry.stdout').write_text(r.stdout);(out/'registry.stderr').write_text(r.stderr)
  result['registry']=json.loads(next(x for x in r.stdout.splitlines() if x.startswith('{"status"')))
 result['status']='PASS'
except Exception as exc:
 result.update(status='FAILED',error=str(exc))
result.update(wall_seconds=time.monotonic()-started,commands=commands,output=str(out))
receipt=ROOT/'artifacts/llama2-layer15-full-validation-v1/cuda11-compat'/(a.mode+'-tests-'+a.attempt+'.json');assert not receipt.exists();receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='commands'}));raise SystemExit(0 if result['status']=='PASS' else 1)
