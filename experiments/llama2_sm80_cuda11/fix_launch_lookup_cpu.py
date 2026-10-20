#!/usr/bin/env python3
"""Relink a new experimental gate; preserve every frozen source/build/output.

The approved CUDA11-only version map gives the rejected unversioned launch a
distinct version, so it cannot shadow an explicit libcudart.so.11.0 relocation.
No function body, HBF ABI, CUDA library, original build, or OFF map is changed.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('/var/tmp/hbfsim-yjl/llama2-layer15-full-v1')
BASE = WORK / 'cudart11-v1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--attempt', required=True)
args = parser.parse_args()
assert args.attempt.replace('-', '').isalnum()
OUT = WORK / ('cudart11-launch-lookup-' + args.attempt)
OUT.mkdir()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


old_gate = BASE / 'build-gate-on/libhbfsim_launch_gate.so'
assert sha(old_gate) == 'e6e7e5645fe714fc09a92288596cb3ad288961ddc095824e8a1c827d67b714e2'
old_map = BASE / 'hbfsim-src/cmake/cudart11-experiment.map'
original = old_map.read_text()
assert original == 'libcudart.so.11.0 {};\nlibcudart.so.12 {} libcudart.so.11.0;\nlibcudart.so.13 {} libcudart.so.12;\n'
changed = ('HBFSIM_CUDART11_UNVERSIONED {\n'
           '    global: cudaLaunchKernel;\n'
           '};\n' + original)
new_map = OUT / 'cudart11-experiment.map'
new_map.write_text(changed)
patch = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
    fromfile='a/cmake/cudart11-experiment.map', tofile='b/cmake/cudart11-experiment.map'))
(OUT / 'gate-cuda11-elf-lookup.patch').write_text(patch)
build = BASE / 'build-gate-on'
raw = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands',
                               'hbfsim_launch_gate'], text=True).splitlines()[-1]
assert raw.startswith(': && ') and raw.endswith(' && :')
command = shlex.split(raw[5:-5])
assert command.count('-Wl,--version-script=' + str(old_map)) == 1
command[command.index('-Wl,--version-script=' + str(old_map))] = '-Wl,--version-script=' + str(new_map)
new_gate = OUT / 'libhbfsim_launch_gate.so'
command[command.index('-o') + 1] = str(new_gate)
inputs = []
for item in command:
    if not item.startswith('-') and item.endswith(('.o', '.a', '.so')) and item != str(new_gate):
        path = Path(item) if item.startswith('/') else build / item
        inputs.append(dict(path=str(path), sha256=sha(path)))
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
env.pop('LD_PRELOAD', None)
run = subprocess.run(command, cwd=build, env=env, capture_output=True, text=True, timeout=120)
(OUT / 'link.stdout').write_text(run.stdout)
(OUT / 'link.stderr').write_text(run.stderr)
assert run.returncode == 0, run.stderr


def exports(path):
    text = subprocess.check_output(['readelf', '--dyn-syms', '--wide', str(path)], text=True)
    return {p[7] for row in text.splitlines() if len(p := row.split()) >= 8
            and p[6] != 'UND' and p[4] in ('GLOBAL', 'WEAK')}


before, after = exports(old_gate), exports(new_gate)
assert before - after == {'cudaLaunchKernel'}, before - after
assert after - before == {'cudaLaunchKernel@@HBFSIM_CUDART11_UNVERSIONED',
                          'HBFSIM_CUDART11_UNVERSIONED'}, after - before
assert {s for s in before if s.startswith('hbfsim')} == {s for s in after if s.startswith('hbfsim')}
assert {s for s in before if '@libcudart' in s} == {s for s in after if '@libcudart' in s}
assert sha(old_gate) == 'e6e7e5645fe714fc09a92288596cb3ad288961ddc095824e8a1c827d67b714e2'
assert old_map.read_text() == original
receipt = dict(status='BUILD_PASS_CPU_CHECKS_PENDING', GPU_tasks_started=0,
    scope='Approved isolated CUDA11 version-symbol adaptation; ON map only; relink frozen objects',
    old_gate=dict(path=str(old_gate), sha256=sha(old_gate)),
    new_gate=dict(path=str(new_gate), sha256=sha(new_gate)),
    map_before_sha256=sha(old_map), map_after_sha256=sha(new_map),
    command=command, immutable_link_inputs=inputs,
    HBF_exports_unchanged=True, CUDA_versioned_11_12_13_exports_unchanged=True,
    function_bodies_unchanged=True, default_OFF_build_unchanged=True,
    scope_limit='ELF routing/build proof only; no GPU or full HBF validation')
(OUT / 'BUILD.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
