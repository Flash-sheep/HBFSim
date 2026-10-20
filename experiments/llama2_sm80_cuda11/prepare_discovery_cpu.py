#!/usr/bin/env python3
"""Freeze native discovery inputs/preflight. Never write user approval or run GPU."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gate', ROOT / 'tools/eq3_experiment_gate.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--bundle', type=Path, required=True)
a = p.parse_args(); w = a.work.resolve(); bundle = a.bundle.resolve()
assert ROOT in bundle.parents
if bundle.exists():
    assert not list(bundle.iterdir()), 'Preparation outputs are immutable; use a new bundle after any files were written'
else:
    bundle.mkdir()
art = ROOT / 'artifacts/llama2-layer15-full-validation-v1'
post = json.loads((art / 'framework-cpu/postbuild-v1.json').read_text())
build = json.loads((art / 'framework-cpu/finish-build-v1.json').read_text())
assert build['status'] == 'PASS' and post['environment']['status'] == 'PASS'
prefix = w / 'torch-cu117-env'; site = prefix / 'lib/python3.11/site-packages'
hbf = w / 'cudart11-v1/build-gate-on'; agent = w / 'cudart11-v1/build-agent-on'
identity = dict(schema='llama2-native-discovery-runtime-v1',
    python=str(prefix / 'bin/python'), driver='/usr/lib/x86_64-linux-gnu/libcuda.so.565.57.01',
    input_identity='artifacts/llama2-layer15-full-validation-v1/evidence/INPUT_IDENTITY.json',
    cuda12=str(w / 'cuda-12.8'), gate_build=str(hbf),
    gate=str(hbf / 'libhbfsim_launch_gate.so'), plugin=str(hbf / 'libptxpass_hbf.so'),
    loader=str(hbf / 'hbfsim_bpftime_attach_loader'), probe=str(hbf / 'coverage_probe.bpf.o'),
    agent=str(agent / 'runtime/agent/libbpftime-agent.so'),
    server=str(agent / 'runtime/syscall-server/libbpftime-syscall-server.so'))
identity['library_path'] = ':'.join(map(str, [prefix / 'lib', site / 'torch/lib',
    Path('/usr/local/cuda-11.7/lib64'), w / 'agent-deps/lib', w / 'cuda-12.8/targets/x86_64-linux/lib',
    w / 'cuda-12.8/extras/CUPTI/lib64', agent / 'libbpf']))
files = {Path(x) for x in [identity[key] for key in ['python', 'driver', 'gate', 'plugin', 'loader', 'probe', 'agent', 'server']]}
files.add(agent / 'attach/nv_attach_impl/ptx_compiler/libnv_attach_impl_ptx_compiler.so')
for package in ['torch', 'transformers', 'safetensors', 'numpy']:
    files.update(f for f in (site / package).rglob('*') if f.is_file() and
                 (f.suffix == '.py' or '.so' in f.name or f.suffix in ['.json', '.txt']))
files.update((ROOT / 'configs/ptxpass').glob('*.json'))
files.update((art / 'frozen').glob('*.py'))
files.add(art / 'frozen/original_weight_manifest.json')
env = {k: v for k, v in os.environ.items() if k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH']}
env.update(CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH=identity['library_path'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ldd_rows = []
def dynamic_elf(path):
    with path.open('rb') as stream:
        header = stream.read(18)
    return header[:4] == b'\x7fELF' and int.from_bytes(header[16:18], 'little') in [2, 3]
for source in sorted(f for f in files if dynamic_elf(f)):
    cp = subprocess.run(['/usr/bin/ldd', str(source)], env=env, capture_output=True, text=True, timeout=30)
    assert cp.returncode == 0 and 'not found' not in cp.stdout, (source, cp.stdout, cp.stderr)
    ldd_rows.append({'path': str(source), 'stdout': cp.stdout, 'stderr': cp.stderr})
    for line in cp.stdout.splitlines():
        match = re.search(r'(?:=>\s+)?(/[^\s]+)\s+\(', line)
        if match:
            files.add(Path(match[1]))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 2**20), b''): h.update(chunk)
    return h.hexdigest()
identity['files'] = [{'path': str(f), 'resolved_path': str(f.resolve()),
                      'sha256': sha(f), 'bytes': f.stat().st_size} for f in sorted(files)]
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
identity['CPU_source_build'] = post['environment']
(bundle / 'runtime-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
(bundle / 'ldd-closure.json').write_text(json.dumps({'status': 'PASS', 'GPU_tasks_started': 0, 'rows': ldd_rows}, indent=2) + '\n')
cmd = [identity['python'], '-B', str(ROOT / 'experiments/llama2_sm80_cuda11/native_discovery.py'), '--bundle', str(bundle)]
t = time.monotonic(); cp = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=180)
(bundle / 'cpu-identity.stdout').write_text(cp.stdout); (bundle / 'cpu-identity.stderr').write_text(cp.stderr)
assert cp.returncode == 0, cp.stderr
(bundle / 'CPU_IDENTITY.json').write_text(json.dumps({'status': 'PASS', 'returncode': cp.returncode,
    'wall_seconds': time.monotonic() - t, 'files': len(files), 'GPU_tasks_started': 0,
    'CUDA_initialized': False, 'command': cmd}, indent=2) + '\n')
test_cmd = [identity['python'], '-B', '-m', 'unittest', 'discover', '-s', 'experiments/llama2_sm80_cuda11/tests',
            '-p', 'test_native_discovery.py', '-v']
cp = subprocess.run(test_cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=60)
(bundle / 'cpu-tests.stdout').write_text(cp.stdout); (bundle / 'cpu-tests.stderr').write_text(cp.stderr)
assert cp.returncode == 0, cp.stderr
(bundle / 'CPU_BOUNDARY_FIXTURES.json').write_text(json.dumps({'status': 'PASS', 'returncode': 0,
    'scope': 'mocked refusal-before-action, byte window and environment boundaries; no GPU',
    'GPU_tasks_started': 0, 'command': test_cmd}, indent=2) + '\n')

def binding(path, logical_id, role):
    path = Path(path)
    return dict(logical_id=logical_id, semantic_role=role, sha256=sha(path), path=str(path.relative_to(ROOT)))
code = [binding(ROOT / name, Path(name).name, 'Frozen discovery code or original helper') for name in [
    'experiments/llama2_sm80_cuda11/native_discovery.py', 'tools/eq3_experiment_gate.py',
    'artifacts/llama2-layer15-full-validation-v1/frozen/trace_decoder_layer.py',
    'artifacts/llama2-layer15-full-validation-v1/frozen/trace_flint_decoder_layer.py']]
scientific = {
 'research_question_and_hypothesis': {'question': 'Can the exact-source CUDA11.7/sm80 build execute the original complete layer15 and preserve its output under the actual strict bridge?', 'hypothesis': 'The original GPU forward succeeds; unbound strict observation selects original functions and preserves exact FP16 bytes.', 'evidence_identity': 'Software engineering verification only'},
 'evidence_type_and_limits': {'type': 'Bounded native/control and runtime-discovery software validation', 'limits': ['No HBF ranges or Timing requests', 'No instrumented PTX is loaded', 'Actual selected kernels may differ from old trace55', 'Discovery is a dependency of full HBF validation, not its PASS']},
 'device_topology': {'GPU_UUID': 'GPU-5302a497-ea8c-bb7b-7235-f7464792093e', 'GPU': 'A100-PCIE-40GB sm80', 'driver': '565.57.01', 'selection': 'Both loader and target see only this UUID', 'GPU0': 'Occupied, excluded; no changes or reset'},
 'geometry_materials_boundaries': {'status': 'NOT_USED', 'reason': 'No package/thermal/physical HBF experiment'},
 'workload_and_initial_state': {'model': '/mnt/llama2-7b', 'workload': '/mnt/hbfsim-workload', 'layer_index': 15, 'batch_query_context': [1,1,32], 'dtype': 'FP16', 'seed': 1, 'warmup': 0, 'forwards_per_arm': 1, 'fresh_processes': True, 'registered_parameters': 9, 'weight_bytes': 404766720, 'HBF_registration_count': 0, 'arena': 'Original helper, new per-process addresses,256B alignment', 'input_kind': 'Checkpoint weights; original fixed-seed synthetic hidden/KV', 'software_control_INFERRED': 'CUDA_MODULE_LOADING=EAGER; eager original operators; TF32 disabled'},
 'power_reliability_control': {'physical_power_reliability': 'NOT_USED', 'safety': 'Temperature and occupancy monitoring; no clock/power limit, driver or global environment changes'},
 'scan_matrix_and_repetitions': {'arms': ['native without agent/gate', 'strict agent+gate capture, unbound original path'], 'repeats': 1, 'total_full_forwards': 2, 'automatic_retry': False, 'loader': 'One private namespace server+loader; may initialize its own context on the same selected GPU'},
 'time_and_numerics': {'simulated_time': 'NOT_USED', 'expected_wall_seconds': '30–90 after input validation; no earlier measured full GPU time', 'execution_watchdog_seconds': 180, 'cleanup_grace_seconds_per_actor': 5, 'GPU_budget_minutes_including_cleanup': 4, 'comparison': 'Exact8192 FP16 output bytes; all finite; no tolerance relaxation', 'timing_limits': 'Host forward+sync interval includes instrumentation and synchronization overhead; not performance benchmarking'},
 'controls_ablations_observations': {'control': 'Same new build/input, two fresh processes; observation only in second arm', 'observations': ['GPU UUID/capability and resources', 'fresh9-weight arena addresses/layout', 'forward-scoped coverage and strict runtime/driver records', 'CUDA11 native registrations', 'loaded file paths', '8192-byte output and SHA per arm', 'PID/exit/stop reason'], 'AI_polling': 'No per-kernel AI polling; parent safety monitoring every1s'},
 'acceptance_abort_and_outputs': {'pass': 'Two single forwards finish, shape1x1x4096 FP16 all finite and byte-identical; capture has allowed/unmodeled original-path bridge records and successful CUDA11 runtime records', 'abort': ['Any CUDA,output,identity,bridge or child failure stops subsequent actors', 'Target busy/foreign process; startup memory>=1024MiB or utilization!=0 or temperature>=75C', 'Runtime GPU memory>=12288MiB or temperature>=80C or foreign process', 'Actor-group RSS>16GiB or host available RAM<64GiB', 'Output>2GiB or filesystem free<2GiB', 'Work execution>180s; terminate only owned actors; parent-death kill guard', 'No automatic retry or fallback'], 'output': str(bundle / 'execution-v1'), 'completion': ['START.json', '*PID.json', 'RESULT.json', 'DONE or FAILED'], 'state': 'PENDING_USER_APPROVAL; no job submitted', 'exact_command': cmd + ['--execute', '--approval', str(bundle / 'USER_CONFIRMATION.json')]}
}
resources = dict(cpu_configurations=2, executions_per_configuration=1, cpu_hours=0.2,
                 build_threads=0, ram_gib=16, disk_gib=2, gpu_compute_minutes=4)
prereq_paths = [('CPU identity/model tensor closure', bundle / 'CPU_IDENTITY.json'),
    ('CPU safety fixtures', bundle / 'CPU_BOUNDARY_FIXTURES.json'),
    ('Private runtime artifact build', art / 'framework-cpu/finish-build-v1.json'),
    ('CUDA11 provider and regression tests', art / 'cuda11-compat/PROVIDER_FIX_V2.json')]
manifest = dict(schema_version='eq3-experiment-manifest-v1', experiment_id='LLAMA2-SM80-NATIVE-DISCOVERY',
    version='1', approval_status='PENDING_USER_APPROVAL', scientific_config=scientific,
    code=dict(repository_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              dirty_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=ROOT)).hexdigest(), artifacts=code),
    dependencies=[binding(bundle / 'runtime-identity.json', 'runtime-identity', 'Actual external runtime/source files, rehashed before each spawn'),
                  binding(bundle / 'ldd-closure.json', 'ldd-closure', 'Read-only dynamic dependency closure')],
    inputs=[binding(art / 'evidence/INPUT_IDENTITY.json', 'original-inputs', 'Original scripts/config and ten checkpoint tensor identities'),
            binding(art / 'frozen/original_weight_manifest.json', 'original-layout', 'Original9 weights; old addresses never reused')],
    resource_budget=dict(requested=resources, limits=resources),
    prerequisites=[dict(id=str(i), status='PASSED', evidence=binding(path, 'prerequisite-' + str(i), role)) for i,(role,path) in enumerate(prereq_paths)],
    canonical_manifest_hash='0' * 64,
    execution_context={'created_at_utc': identity['captured_at_utc'], 'launch_performed': False,
                       'GPU_forward_calls_started': 0, 'scope': 'Native discovery only; not HBF Timing'})
manifest['canonical_manifest_hash'] = gate.canonical_manifest_hash(manifest)
(bundle / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PENDING_USER_APPROVAL', 'bundle': str(bundle), 'canonical_manifest_hash': manifest['canonical_manifest_hash'], 'GPU_tasks_started': 0}))
