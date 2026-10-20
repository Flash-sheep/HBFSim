#!/usr/bin/env python3
"""Build the standalone passive observer and test with a fake CUPTI provider."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from stage_observed_torch_cpu import ROOT, save, sha

parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--attempt', required=True); args=parser.parse_args()
assert args.attempt.replace('-', '').isalnum()
out = Path('/var/tmp/hbfsim-yjl/llama2-layer15-full-v1')/('cupti-observer-'+args.attempt); out.mkdir()
source = ROOT/'experiments/llama2_sm80_cuda11/cupti_observer.cpp'
include = ['/usr/local/cuda-11.7/extras/CUPTI/include', '/usr/local/cuda-11.7/include']
flags = ['-I'+p for p in include]
command = ['/usr/bin/g++-11','-shared','-fPIC','-O2','-std=c++17',*flags,str(source),
    '-L/usr/local/cuda-11.7/extras/CUPTI/lib64','-lcupti','-ldl','-pthread',
    '-Wl,-rpath,/usr/local/cuda-11.7/extras/CUPTI/lib64','-o',str(out/'libllama2_cupti_observer.so')]
run = subprocess.run(command, capture_output=True, text=True, timeout=60)
(out/'build.stdout').write_text(run.stdout); (out/'build.stderr').write_text(run.stderr)
assert run.returncode == 0, run.stderr[-3000:]
fake = out/'fake'; fake.mkdir()
(fake/'cupti.cpp').write_text('''#include <cupti.h>
static CUpti_CallbackFunc cb; static int failure;
extern "C" CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle*s,CUpti_CallbackFunc f,void*){if(failure)return (CUptiResult)failure;cb=f;*s=(CUpti_SubscriberHandle)0x123;return CUPTI_SUCCESS;}
extern "C" CUptiResult CUPTIAPI cuptiEnableDomain(uint32_t,CUpti_SubscriberHandle,CUpti_CallbackDomain){return CUPTI_SUCCESS;}
extern "C" CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle){cb=nullptr;return CUPTI_SUCCESS;}
extern "C" void fake_failure(int v){failure=v;}
extern "C" void fake_emit(int type,int site,int result){
 CUpti_CallbackData d{};d.callbackSite=(CUpti_ApiCallbackSite)site;
 d.functionName=type==0?"cuLaunchKernel":type==1?"cuGraphLaunch":"cuModuleGetFunction";
 d.symbolName="kernel_\\\"quoted";d.correlationId=42;d.contextUid=9;d.context=(CUcontext)0x789;
 CUresult r=(CUresult)result;d.functionReturnValue=&r;
 cuLaunchKernel_params launch{};launch.f=(CUfunction)0x456;
 CUfunction f=(CUfunction)0x456;cuModuleGetFunction_params lookup{};lookup.hfunc=&f;lookup.hmod=(CUmodule)0xabc;lookup.name="kernel_lookup";
 d.functionParams=type==2?(const void*)&lookup:(const void*)&launch;
 if(cb)cb(nullptr,CUPTI_CB_DOMAIN_DRIVER_API,type==0?CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:type==1?CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction,&d);
}
''')
mock_command = ['/usr/bin/g++-11','-shared','-fPIC','-O0',*flags,str(fake/'cupti.cpp'),
    '-Wl,-soname,libcupti.so.11.7','-o',str(fake/'libcupti.so.11.7')]
subprocess.run(mock_command,check=True,capture_output=True,timeout=60)
(out/'test.py').write_text('''import ctypes as C,json,os
from pathlib import Path
p=Path(os.environ['OBSERVER_TEST_DIR']);obs=C.CDLL(str(p/'libllama2_cupti_observer.so'));fake=C.CDLL(str(p/'fake/libcupti.so.11.7'))
start=obs.llama2_cupti_observer_start;start.argtypes=[C.c_char_p];stop=obs.llama2_cupti_observer_stop;stop.argtypes=[C.POINTER(C.c_ulonglong)]*3
assert start(None)==-1
path=p/'fixture.jsonl';assert start(str(path).encode())==0
assert start(str(path).encode())==-1
for typ in [0,1,2]:
 for site in [0,1]:fake.fake_emit(typ,site,0)
a,b,e=C.c_ulonglong(),C.c_ulonglong(),C.c_ulonglong();assert stop(C.byref(a),C.byref(b),C.byref(e))==0
assert (a.value,b.value,e.value)==(6,6,0)
rows=[json.loads(s) for s in path.read_text().splitlines()]
assert rows[0]['function']=='0x456' and rows[0]['kernel']=='kernel_"quoted'
assert rows[0]['result'] is None and rows[1]['result']==0
assert not rows[2]['supported_parameter_layout'] and rows[2]['event']=='driver_launch'
assert rows[-1]['module']=='0xabc' and rows[-1]['function']=='0x456' and rows[-1]['lookup_name']=='kernel_lookup'
fake.fake_failure(7);assert start(str(p/'failure.jsonl').encode())==7
maps=Path('/proc/self/maps').read_text();assert str(p/'fake/libcupti.so.11.7') in maps
assert 'libcuda.so' not in maps and 'libcudart.so' not in maps
assert not any('/dev/nvidia' in os.readlink(f) for f in Path('/proc/self/fd').iterdir() if f.exists())
print(json.dumps(dict(status='PASS',callbacks=a.value,recorded=b.value,write_errors=e.value,CUDA_driver_mapped=False,GPU_tasks_started=0)))
''')
env = {k:v for k,v in os.environ.items() if k not in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH')}
env.update(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(fake),OBSERVER_TEST_DIR=str(out))
run = subprocess.run(['/usr/bin/python3','-B',str(out/'test.py')],env=env,capture_output=True,text=True,timeout=30)
(out/'test.stdout').write_text(run.stdout); (out/'test.stderr').write_text(run.stderr)
assert run.returncode == 0, run.stderr[-3000:]
result = dict(status='CPU_BUILD_AND_MOCK_CALLBACK_PASS',GPU_tasks_started=0,
    observer=dict(path=str(out/'libllama2_cupti_observer.so'),sha256=sha(out/'libllama2_cupti_observer.so')),
    source=dict(path=str(source),sha256=sha(source)),command=command,
    headers=[dict(path=str(p),sha256=sha(p)) for p in [Path(include[0])/'cupti_callbacks.h',Path(include[0])/'generated_cuda_meta.h',Path(include[0])/'cupti_driver_cbid.h']],
    real_CUPTI=dict(path='/usr/local/cuda-11.7/extras/CUPTI/lib64/libcupti.so.11.7',sha256=sha('/usr/local/cuda-11.7/extras/CUPTI/lib64/libcupti.so.11.7')),
    fixture=json.loads(run.stdout),
    limits='Callback ABI fixtures only. No actual CUPTI initialization, GPU callback visibility or compatibility claim.')
save(out/'RESULT.json',result);print(json.dumps(result,indent=2))
