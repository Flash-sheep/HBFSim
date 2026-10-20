#!/usr/bin/env python3
"""Finish only failed CPU preparations; preserve the first attempt unchanged."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_observed_torch_cpu import ROOT, load_module, save, sha

WORK = Path('/var/tmp/hbfsim-yjl/llama2-layer15-full-v1')


def child(path):
    spec = json.loads(path.read_text()); out = path.parent
    source = Path(spec['member']['saved_path']); assert sha(source) == spec['member']['member_sha256']
    proof = load_module('assertproof', ROOT/'experiments/llama2_sm80_cuda11/assertfail_proof_cpu.py')
    payload = source.read_bytes()
    save(out/'ENTRY_PROOF.json', [proof.prove(payload, entry) for entry in spec['entries']])
    auto = load_module('auto_prepare', ROOT/'adapters/vllm/auto_prepare_ptx.py')
    result = auto._transform_module_child(source, Path(spec['plugin']),
        out/'pass-manifest.raw.jsonl', spec['entries'], 'NO_UNRESOLVED_MEMORY_FUNCTION', [])
    staged = out/'transformed-original-version.ptx'
    staged.write_bytes(base64.b64decode(result.pop('output_ptx_base64'), validate=True))
    save(out/'TRANSFORM_RESULT.json', result)
    assert all(r['status'] == 'SUPPORTED_TRANSFORMED' for r in result['entry_results'])
    assert sha(staged) == result['staged_sha256']
    version = load_module('version_copy', ROOT/'experiments/llama2_sm80_cuda11/version_copy.py')
    copy = version.copy_version(staged, out/'version87', sha(staged))
    cmd = [spec['ptxas'], '--gpu-name=sm_80', '--compile-only', '--maxrregcount=64',
           '--entry='+','.join(spec['entries']), copy['output'], '-o', str(out/'version87.o')]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    (out/'assembly.stdout').write_text(run.stdout); (out/'assembly.stderr').write_text(run.stderr)
    save(out/'ASSEMBLY.json', dict(command=cmd, returncode=run.returncode,
        input_sha256=copy['output_sha256'], GPU_tasks_started=0))
    save(out/'RESULT.json', dict(status='CPU_TRANSFORMED_NOT_GPU_VALIDATED',
        assembly_returncode=run.returncode, GPU_tasks_started=0))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--child', type=Path); a = p.parse_args()
    if a.child: child(a.child); return
    previous = WORK/'observed-torch-stage-v1'
    out = WORK/'observed-torch-finish-v1'; out.mkdir()
    identity = json.loads((ROOT/'artifacts/llama2-sm80-native-discovery-v3/runtime-identity.json').read_text())
    env = {k:v for k,v in os.environ.items() if not k.startswith(('HBFSIM_', 'BPFTIME_', 'CUDA_'))
           and k not in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH')}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', LD_LIBRARY_PATH=identity['library_path'])
    start = time.monotonic(); results = []
    save(out/'PLAN.json', dict(scope='Fixed CPU repair verification of two call proofs and two register-limit failures',
        GPU_tasks=0, serial=True, child_wall_seconds=300, memory_budget_GiB=16,
        source_preserved=str(previous),
        register_option='--maxrregcount=64; existing agent HBFSIM_NVPTX_MAX_REGISTERS option, no helper or source PTX rewrite'))
    summary = json.loads((previous/'RESULT.json').read_text())
    for row in summary['modules']:
        module = previous/row['raw_sha256']
        if row['returncode']:
            current = out/row['raw_sha256']; current.mkdir()
            spec = json.loads((module/'SPEC.json').read_text()); save(current/'SPEC.json', spec)
            with (current/'stdout').open('xb') as stdout, (current/'stderr').open('xb') as stderr:
                proc = subprocess.run([sys.executable,'-B',__file__,'--child',str(current/'SPEC.json')],
                    env=env, stdout=stdout, stderr=stderr, timeout=300)
            save(current/'PROCESS.json', dict(returncode=proc.returncode))
            results.append(dict(module=row['raw_sha256'], kind='static_diagnostic_call_proof_and_transform', returncode=proc.returncode))
        elif row['result']['assembly_version87_returncode']:
            current = out/row['raw_sha256']; current.mkdir()
            spec = json.loads((module/'SPEC.json').read_text()); source = module/'version87/version-adapted.ptx'
            cmd = [spec['ptxas'],'--gpu-name=sm_80','--compile-only','--maxrregcount=64',
                '--entry='+','.join(spec['entries']),str(source),'-o',str(current/'version87.o')]
            proc = subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
            (current/'assembly.stdout').write_text(proc.stdout); (current/'assembly.stderr').write_text(proc.stderr)
            save(current/'ASSEMBLY.json',dict(command=cmd,returncode=proc.returncode,input_sha256=sha(source),GPU_tasks_started=0))
            results.append(dict(module=row['raw_sha256'],kind='existing_max_register_compile_option',returncode=proc.returncode))
    save(out/'RESULT.json',dict(status='CPU_FOLLOWUP_FINISHED',results=results,
        wall_seconds=time.monotonic()-start,GPU_tasks_started=0,
        limitation='Selected-entry relocatable assembly, not runtime full-module linking or GPU/HBF correctness'))
    print(json.dumps(results))


if __name__ == '__main__': main()
