#!/usr/bin/env python3
"""Prepare capture-only v2 after the stopped v1; never launch a GPU task."""
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import math
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
old = ROOT / 'artifacts/llama2-sm80-native-discovery-v1'
out = ROOT / 'artifacts/llama2-sm80-native-discovery-v2'
out.mkdir()
source = ROOT / 'experiments/llama2_sm80_cuda11/native_discovery_v2.py'
s = importlib.util.spec_from_file_location('gate', ROOT / 'tools/eq3_experiment_gate.py')
g = importlib.util.module_from_spec(s); s.loader.exec_module(g)
m = g.load_json(old / 'manifest.json')
g.validate_gate(m, g.load_json(old / 'USER_CONFIRMATION.json'), ROOT, old / 'USER_CONFIRMATION.json')
native = old / 'execution-v1/native'
baseline = g.load_json(native / 'RESULT.json')
assert baseline['status'] == 'NATIVE_FORWARD_PASS_HBF_NOT_RUN'
assert baseline['forward_calls'] == 1 and baseline['warmup_calls'] == 0
assert baseline['gpu_uuid'] == 'GPU-5302a497-ea8c-bb7b-7235-f7464792093e'
assert baseline['shape'] == [1, 1, 4096] and baseline['registered_HBF_ranges'] == 0
raw = (native / 'hidden-output.bin').read_bytes()
assert len(raw) == 8192 and hashlib.sha256(raw).hexdigest() == baseline['output_sha256']
assert all(math.isfinite(x) for x in struct.unpack('<4096e', raw))
expected = g.load_json(ROOT / 'artifacts/llama2-layer15-full-validation-v1/frozen/original_weight_manifest.json')
arena = g.load_json(native / 'arena.json')
assert arena['tensor_count'] == 9 and arena['weight_payload_bytes'] == 404766720
for actual, ref in zip(arena['tensors'], expected['tensors']):
    assert all(actual[key] == ref[key] for key in ['name', 'shape', 'offset_bytes', 'size_bytes'])
    assert actual['address'] == arena['hbf_base_addr'] + actual['offset_bytes']
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 2**20), b''): h.update(chunk)
    return h.hexdigest()
identity = g.load_json(old / 'runtime-identity.json')
for row in identity['files']:
    assert sha(row['path']) == row['sha256'], row['path']
identity['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
identity['baseline'] = {'output': str((native / 'hidden-output.bin').relative_to(ROOT)),
                        'output_sha256': baseline['output_sha256'],
                        'result': str((native / 'RESULT.json').relative_to(ROOT)),
                        'reused_scope': 'Same original input, helper, GPU UUID, driver and frozen environment; software output control only'}
for path in [native / 'RESULT.json', native / 'hidden-output.bin', native / 'arena.json']:
    identity['files'].append({'path': str(path), 'resolved_path': str(path.resolve()),
                             'sha256': sha(path), 'bytes': path.stat().st_size})
(out / 'runtime-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
reuse = {'status': 'PASS', 'scope': 'Software baseline applicability audit; not physical/calibrated evidence',
         'original_native_result': baseline, 'output_hash_verified': True,
         'nine_parameter_layout_verified': True, 'old_runtime_file_count_verified': len(identity['files']) - 3,
         'v2_new_gpu_forward_calls_requested': 1, 'native_baseline_reexecution': False,
         'source_change': 'Only worker child environment, logging and capture-only orchestration; original forward helpers unchanged',
         'GPU_tasks_started': 0}
(out / 'BASELINE_REUSE_CPU.json').write_text(json.dumps(reuse, indent=2) + '\n')
env = {k: v for k, v in os.environ.items() if k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']}
env.update(CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH=identity['library_path'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
cmd = [identity['python'], '-B', str(source), '--bundle', str(out)]
t = time.monotonic(); cp = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=180)
(out / 'cpu-identity.stdout').write_text(cp.stdout); (out / 'cpu-identity.stderr').write_text(cp.stderr)
assert cp.returncode == 0, cp.stderr
(out / 'CPU_IDENTITY.json').write_text(json.dumps({'status': 'PASS', 'command': cmd, 'returncode': 0,
     'wall_seconds': time.monotonic() - t, 'GPU_tasks_started': 0, 'CUDA_initialized': False}, indent=2) + '\n')
cmd = [identity['python'], '-B', '-m', 'unittest', 'discover', '-s', 'experiments/llama2_sm80_cuda11/tests',
       '-p', 'test_native_discovery_v2.py', '-v']
cp = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=60)
(out / 'cpu-tests.stdout').write_text(cp.stdout); (out / 'cpu-tests.stderr').write_text(cp.stderr)
assert cp.returncode == 0, cp.stderr
(out / 'CPU_SAFETY_TESTS.json').write_text(json.dumps({'status': 'PASS', 'command': cmd, 'returncode': 0,
    'scope': 'Four CPU tests: preload cleanup before approval, no approval no launch, capture-only scope, unchanged resource thresholds',
    'GPU_tasks_started': 0}, indent=2) + '\n')
def binding(path, logical, role):
    path = Path(path)
    return dict(logical_id=logical, semantic_role=role, path=str(path.relative_to(ROOT)), sha256=sha(path))
new = copy.deepcopy(m)
new['version'] = '2'
new['approval_status'] = 'PENDING_USER_APPROVAL'
for row in new['code']['artifacts']:
    if row['logical_id'] == 'native_discovery.py':
        row.update(binding(source, 'native_discovery_v2.py', 'Capture-only wrapper with CPU-proven preload isolation and explicit safety telemetry'))
new['dependencies'][0] = binding(out / 'runtime-identity.json', 'runtime-identity', 'Same frozen runtime plus immutable successful v1 native control')
new['inputs'].append(binding(native / 'hidden-output.bin', 'native-output-v1', 'Immutable successful native GPU output; not an old CPU or other-device surrogate'))
science = new['scientific_config']
science['research_question_and_hypothesis']['question'] = 'Does the corrected strict capture complete the original layer15 and preserve the successful immutable v1 native output?'
science['scan_matrix_and_repetitions'] = {'arms': ['strict agent+gate capture only'], 'repeats': 1,
    'total_new_full_forwards': 1, 'native_control': 'Reuse immutable successful v1 native, after identity/layout audit',
    'automatic_retry': False, 'loader': 'One fresh private-namespace server+loader on the same selected GPU'}
science['workload_and_initial_state']['native_control_reused'] = identity['baseline']
science['controls_ablations_observations']['control'] = 'Successful v1 native on the exact same frozen runtime/input/GPU; no native reexecution'
science['controls_ablations_observations']['observations'] += ['Per-check GPU resource and allowlist samples', 'Explicit safety-stop reason and sample', 'Final actor exit codes', 'Task-local bpftime runtime log']
science['time_and_numerics']['expected_wall_seconds'] = '20–60 after CPU validation; v1 total actual20.573s stopped before capture forward, so no measured capture duration'
science['acceptance_abort_and_outputs']['pass'] = 'One capture forward succeeds; FP16 finite1x1x4096 output matches immutable v1 native8192 bytes exactly; actual strict/runtime records select successful original path'
science['acceptance_abort_and_outputs']['output'] = str(out / 'execution-v1')
science['acceptance_abort_and_outputs']['exact_command'] = [identity['python'], '-B', str(source), '--bundle', str(out), '--execute', '--approval', str(out / 'USER_CONFIRMATION.json')]
science['acceptance_abort_and_outputs']['repair_scope'] = 'Drop LD_PRELOAD from the worker child environment before its Git check; keep the already loaded agent/gate active. Keep main-PID allowlist and all thresholds; no core/ABI/ownership rewrites.'
resources = dict(new['resource_budget']['requested']); resources['cpu_configurations'] = 1
new['resource_budget'] = dict(requested=resources, limits=dict(resources))
paths = [('baseline-reuse', out / 'BASELINE_REUSE_CPU.json', 'Current identity and layout audit of the immutable successful native GPU control'),
         ('CPU-identity', out / 'CPU_IDENTITY.json', 'Actual new runtime/input CPU identity verification'),
         ('CPU-safety', out / 'CPU_SAFETY_TESTS.json', 'Four capture-only wrapper CPU fixtures'),
         ('preload-defect-fixed', old / 'analysis-v1/PRELOAD_INHERITANCE_CPU.json', 'Actual Git subprocess CPU-marker reproducer before/after; no GPU'),
         ('runtime-build', ROOT / 'artifacts/llama2-layer15-full-validation-v1/framework-cpu/finish-build-v1.json', 'Unchanged built runtime artifacts')]
new['prerequisites'] = [dict(id=key, status='PASSED', evidence=binding(path, key, role)) for key, path, role in paths]
new['execution_context'] = {'created_at_utc': identity['captured_at_utc'], 'launch_performed': False,
                           'GPU_forward_calls_started': 0, 'scope': 'Capture only; native control reused; HBF not run'}
new['canonical_manifest_hash'] = g.canonical_manifest_hash(new)
(out / 'manifest.pre-entry-check.json').write_text(json.dumps(new, ensure_ascii=False, indent=2) + '\n')
(out / 'manifest.json').write_text(json.dumps(new, ensure_ascii=False, indent=2) + '\n')
rows = []
for flags in [['--execute'], ['--worker', 'capture']]:
    cmd = [identity['python'], '-B', str(source), '--bundle', str(out), *flags]
    cp = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert cp.returncode == 2 and 'no independently supplied user confirmation' in cp.stderr, cp.stderr
    rows.append({'command': cmd, 'returncode': cp.returncode, 'stderr': cp.stderr})
assert not (out / 'execution-v1').exists()
(out / 'CLI_APPROVAL_BOUNDARY.json').write_text(json.dumps({'status': 'PASS', 'rows': rows,
     'scope': 'Both actual allowed GPU entries refuse missing user approval before execution directory creation',
     'GPU_tasks_started': 0}, indent=2) + '\n')
new['prerequisites'].append(dict(id='actual-entry-refusal', status='PASSED', evidence=binding(
    out / 'CLI_APPROVAL_BOUNDARY.json', 'actual-entry-refusal', 'Actual final source GPU CLI refusal check')))
new['canonical_manifest_hash'] = g.canonical_manifest_hash(new)
(out / 'manifest.json').write_text(json.dumps(new, ensure_ascii=False, indent=2) + '\n')
g.verify_artifacts(new, ROOT)
try:
    g.validate_gate(new, None, ROOT)
except g.GateError as error:
    assert error.code == 'PENDING_USER_APPROVAL'
else:
    raise AssertionError('missing approval must refuse')
print(json.dumps({'status': 'PENDING_USER_APPROVAL', 'version': '2', 'canonical_manifest_hash': new['canonical_manifest_hash'],
                  'GPU_tasks_started': 0, 'new_forward_calls_requested': 1}))
