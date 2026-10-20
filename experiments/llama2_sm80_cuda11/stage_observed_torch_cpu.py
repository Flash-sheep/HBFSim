#!/usr/bin/env python3
"""CPU-only, exact-entry preparation from the successful v3 observations.

This prepares only the observed Torch subset. It never infers missing cuBLAS
launches, binds a GPU function, or claims whole-layer HBF readiness.
"""
import argparse
import base64
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2); f.write('\n')


def child(spec_path):
    spec = json.loads(spec_path.read_text())
    out = spec_path.parent
    source = Path(spec['member']['saved_path'])
    assert sha(source) == spec['member']['member_sha256']
    proof = load_module('entry_proof', ROOT / 'tools/native_ptx_reproduction/execution/entry_proof.py')
    auto = load_module('auto_prepare', ROOT / 'adapters/vllm/auto_prepare_ptx.py')
    raw = source.read_bytes()
    evidence = []
    for entry in spec['entries']:
        _, body, span = proof.entry_body(raw, entry)
        calls = proof.CALL_RE.findall(body)
        evidence.append(dict(kernel=entry, call_count=len(calls), **span))
    save(out / 'ENTRY_PROOF.json', evidence)
    if any(row['call_count'] for row in evidence):
        raise RuntimeError('Selected entry has calls; no no-call proof; do not assert a closed call graph')
    result = auto._transform_module_child(source, Path(spec['plugin']),
        out / 'pass-manifest.raw.jsonl', spec['entries'], 'NO_UNRESOLVED_MEMORY_FUNCTION', [])
    payload = base64.b64decode(result.pop('output_ptx_base64'), validate=True)
    staged = out / 'transformed-original-version.ptx'
    with staged.open('xb') as f: f.write(payload)
    save(out / 'TRANSFORM_RESULT.json', result)
    assert all(r['status'] == 'SUPPORTED_TRANSFORMED' for r in result['entry_results'])
    assert sha(staged) == result['staged_sha256']
    version = load_module('version_copy', ROOT / 'experiments/llama2_sm80_cuda11/version_copy.py')
    copy = version.copy_version(staged, out / 'version87', sha(staged))
    # Assemble only the selected entries, retaining the full PTX member bytes.
    # Both failures and successes are immutable receipts; no architecture rewrite.
    for label, path in [('original-version', staged), ('version87', Path(copy['output']))]:
        cmd = [spec['ptxas'], '--gpu-name=sm_80', '--compile-only',
               '--entry=' + ','.join(spec['entries']), str(path), '-o', str(out / (label + '.o'))]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        (out / (label + '.ptxas.stdout')).write_text(proc.stdout)
        (out / (label + '.ptxas.stderr')).write_text(proc.stderr)
        save(out / (label + '.ASSEMBLY.json'), dict(command=cmd, returncode=proc.returncode,
            input_sha256=sha(path), GPU_tasks_started=0,
            output_sha256=sha(out / (label + '.o')) if proc.returncode == 0 else None))
    maps = Path('/proc/self/maps').read_text()
    assert 'libbpftime-agent.so' not in maps
    assert not any('/dev/nvidia' in os.readlink(p) for p in Path('/proc/self/fd').iterdir() if p.exists())
    save(out / 'RESULT.json', dict(status='CPU_PREPARED_NOT_GPU_VALIDATED',
         selected_entries=len(spec['entries']), original_member_sha256=sha(source),
         derived_version_copy=copy, GPU_tasks_started=0,
         assembly_version87_returncode=proc.returncode,
         limitation='Assembly is selected-entry relocatable-object validation, not runtime link/load or whole-layer coverage'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--child', type=Path)
    args = p.parse_args()
    if args.child:
        child(args.child); return
    assert args.output is not None
    out = args.output.resolve(); out.mkdir()
    started = time.monotonic()
    bundle = ROOT / 'artifacts/llama2-sm80-native-discovery-v3'
    capture = bundle / 'execution-v1/capture'
    assert json.loads((bundle / 'execution-v1/RESULT.json').read_text())['status'] == 'NATIVE_DISCOVERY_PASS_HBF_NOT_RUN'
    windows = json.loads((capture / 'FORWARD_CAPTURE.json').read_text())
    verifier = load_module('discovery', ROOT / 'experiments/llama2_sm80_cuda11/native_discovery_v3.py')
    verifier.validate_forward_records(windows)
    identity = json.loads((bundle / 'runtime-identity.json').read_text())
    collector = Path('/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/logs/postbuild-v1/torch-cuda-original-collection/native-ptx-manifest.json')
    collection = json.loads(collector.read_text())
    assert sha(collection['container']['path']) == collection['container']['sha256']
    join = load_module('scoped_join', ROOT / 'tools/native_ptx_reproduction/scoped_join/join_scoped_staging_provenance.py')
    members, memberships = join.verify_collector(collection)
    registrations = [json.loads(s) for s in (capture / 'native-registration.jsonl').read_text().splitlines()]
    functions = collections.defaultdict(list)
    for row in registrations:
        if row['event'] == 'register_function': functions[row['object']].append(row)
    assert len(windows['coverage']) == len(windows['strict-runtime'])
    observed, groups = [], {}
    for gate, launch in zip(windows['coverage'], windows['strict-runtime']):
        candidates = functions[launch['host_stub']]
        assert len(candidates) == 1
        reg = candidates[0]
        assert reg['symbol'] == gate['kernel'] and reg['runtime_domain'] == 'cudart11'
        assert Path(reg['object_dso']).resolve() == Path(collection['container']['path']).resolve()
        matches = memberships[gate['kernel']]
        assert len(matches) == 1 and matches[0]['target'] == 'sm_80'
        member = members[matches[0]['member_id']]
        group = groups.setdefault(member['member_sha256'], dict(member=member, entries=[]))
        if gate['kernel'] not in group['entries']: group['entries'].append(gate['kernel'])
        observed.append(dict(call_id=launch['call_id'], kernel=gate['kernel'],
                             native_registration=reg, member_membership=matches[0]))
    save(out / 'OBSERVED_SOURCE_JOIN.json', dict(status='OBSERVED_TORCH_SUBSET_SOURCE_JOIN',
        collector_path=str(collector), collector_sha256=sha(collector),
        capture_sha256=sha(capture / 'FORWARD_CAPTURE.json'), launches=observed,
        unique_entries=sum(len(g['entries']) for g in groups.values()), member_count=len(groups),
        whole_layer_kernel_inventory_complete=False, limitation='cuBLAS launch/identity not observed in this channel'))
    save(out / 'PLAN.json', dict(scope='Fixed software preparation of observed Torch subset only',
        modules=len(groups), selected_entries=sum(len(g['entries']) for g in groups.values()),
        source_bytes=sum(g['member']['size_bytes'] for g in groups.values()),
        serial=True, GPU_tasks=0, threads=1, child_memory_bytes=16*2**30,
        child_CPU_seconds=240, child_wall_seconds=300, aggregate_wall_seconds=1200,
        disk_reserve_bytes=2*2**30, host_memory_reserve_bytes=64*2**30,
        tool_hashes={str(path):sha(path) for path in [Path(__file__),
            ROOT/'tools/native_ptx_reproduction/execution/entry_proof.py',
            ROOT/'adapters/vllm/auto_prepare_ptx.py', Path(identity['plugin']),
            ROOT/'experiments/llama2_sm80_cuda11/version_copy.py']}))
    env = {k:v for k,v in os.environ.items() if not k.startswith(('HBFSIM_', 'BPFTIME_', 'CUDA_'))
           and k not in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH')}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               LD_LIBRARY_PATH=identity['library_path'])
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30))
        resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    results = []
    for raw, group in groups.items():
        assert time.monotonic()-started < 1200
        assert shutil.disk_usage(out).free > 2*2**30
        available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        assert available > 64*2**30
        module = out / raw; module.mkdir()
        spec = dict(group, plugin=identity['plugin'], ptxas=identity['cuda12']+'/bin/ptxas')
        save(module / 'SPEC.json', spec)
        with (module / 'stdout').open('xb') as stdout, (module / 'stderr').open('xb') as stderr:
            proc = subprocess.run([sys.executable, '-B', __file__, '--child', str(module/'SPEC.json')],
                env=env, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=300, preexec_fn=limits)
        row = dict(raw_sha256=raw, returncode=proc.returncode, entries=group['entries'])
        if (module/'RESULT.json').is_file(): row['result']=json.loads((module/'RESULT.json').read_text())
        save(module / 'PROCESS.json', row)
        results.append(row)
    save(out / 'RESULT.json', dict(status='CPU_PREPARATION_FINISHED', modules=results,
        wall_seconds=time.monotonic()-started, GPU_tasks_started=0,
        HBF_timing_validated=False, whole_layer_kernel_inventory_complete=False))
    print(json.dumps(dict(modules=len(results), child_failures=sum(r['returncode'] != 0 for r in results),
                         output=str(out), GPU_tasks_started=0)))


if __name__ == '__main__':
    main()
