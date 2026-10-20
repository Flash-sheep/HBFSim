#!/usr/bin/env python3
"""Inspect real ELF launch relocation with the frozen gate and inert CUDA DSOs.

No agent constructor, real CUDA provider, device access, or GPU launch is used.
This is a diagnostic, not a compatibility or numerical acceptance test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--fake-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gate, fake, out = (p.resolve() for p in (args.gate, args.fake_dir, args.output))
    assert (fake / 'fake_cuda.c').is_file()
    assert (fake / 'libcudart.so.11.0.c').is_file()
    out.mkdir()
    source = out / 'consumer.c'
    source.write_text('''#include <stddef.h>
typedef struct {unsigned x,y,z;} Dim;
extern int imported_launch(const void*,Dim,Dim,void**,size_t,void*);
__asm__(".symver imported_launch,cudaLaunchKernel@libcudart.so.11.0");
void* consumer_address(void) {return (void*)&imported_launch;}
int consumer_launch(void) {
    Dim one={1,1,1};
    return imported_launch((const void*)0x123456,one,one,0,0,0);
}
''')
    command = ['/usr/bin/gcc', '-shared', '-fPIC', '-O0', str(source),
               '-L' + str(fake), '-Wl,--no-as-needed', '-l:libcudart.so.11.0',
               '-o', str(out / 'consumer.so')]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    child = out / 'inspect.py'
    child.write_text('''import ctypes as C,json,os
from pathlib import Path
fake=Path(os.environ['CPU_FAKE_DIR']); gate=Path(os.environ['CPU_GATE'])
libc=C.CDLL(None); libc.dlvsym.argtypes=[C.c_void_p,C.c_char_p,C.c_char_p];libc.dlvsym.restype=C.c_void_p
consumer=C.CDLL(os.environ['CPU_CONSUMER'],mode=os.RTLD_NOW|os.RTLD_LOCAL)
runtime=C.CDLL(str(fake/'libcudart.so.11.0'),mode=os.RTLD_NOW|os.RTLD_LOCAL)
runtime.fake_reset(0)
consumer.consumer_address.restype=C.c_void_p
actual=consumer.consumer_address()
provider=libc.dlvsym(runtime._handle,b'cudaLaunchKernel',b'libcudart.so.11.0')
exact=unversioned=None
if os.environ.get('LD_PRELOAD'):
    g=C.CDLL(str(gate),mode=os.RTLD_NOW|os.RTLD_NOLOAD|os.RTLD_LOCAL)
    exact=libc.dlvsym(g._handle,b'cudaLaunchKernel',b'libcudart.so.11.0')
    unversioned=C.cast(g.cudaLaunchKernel,C.c_void_p).value
result=consumer.consumer_launch()
maps=Path('/proc/self/maps').read_text()
cuda_paths=sorted({r.split()[-1] for r in maps.splitlines() if '/libcuda' in r})
assert cuda_paths and all(str(fake) in p for p in cuda_paths),cuda_paths
assert not any('/dev/nvidia' in os.readlink(p) for p in Path('/proc/self/fd').iterdir() if p.exists())
print(json.dumps(dict(relocated_address=hex(actual),provider_address=hex(provider),
    exact_gate_address=hex(exact) if exact else None,
    unversioned_gate_address=hex(unversioned) if unversioned else None,
    selects_exact_gate=actual==exact,selects_unversioned_gate=actual==unversioned,
    selects_provider=actual==provider,result=result,provider_calls=runtime.fake_public_calls(),
    mapped_cuda_paths=cuda_paths,hardware_GPU_used=False)))
''')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('HBFSIM_', 'BPFTIME_', 'CUDA_'))
           and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HBFSIM_INSTRUMENTATION_POLICY='strict', CPU_FAKE_DIR=str(fake),
               CPU_GATE=str(gate), CPU_CONSUMER=str(out / 'consumer.so'),
               LD_LIBRARY_PATH=str(fake) + ':' + str(fake.parent.parent / 'agent-deps/lib'))
    results = {}
    for name in ('provider-control', 'frozen-gate-preload'):
        child_env = dict(env)
        if name == 'frozen-gate-preload':
            child_env['LD_PRELOAD'] = str(gate)
        run = subprocess.run(['/usr/bin/python3', '-B', str(child)], env=child_env,
                             cwd=out, capture_output=True, text=True, timeout=30)
        (out / (name + '.stdout')).write_text(run.stdout)
        (out / (name + '.stderr')).write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(name + ': ' + run.stderr[-4000:])
        results[name] = json.loads(run.stdout)
    control, preloaded = results.values()
    assert control['selects_provider'] and control['result'] == 0 and control['provider_calls'] == 1
    result = dict(schema='llama2-sm80-elf-launch-lookup-cpu-v1',
                  evidence_class='CPU_REPRODUCED', GPU_tasks_started=0,
                  gate=str(gate), gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
                  compile_command=command, cases=results,
                  compatibility_validated=False)
    result['finding'] = ('VERSIONED_RELOCATION_SELECTS_UNVERSIONED_GATE'
                         if preloaded['selects_unversioned_gate']
                         else 'VERSIONED_RELOCATION_SELECTS_EXACT_GATE'
                         if preloaded['selects_exact_gate'] else 'OTHER_PROVIDER_SELECTED')
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
