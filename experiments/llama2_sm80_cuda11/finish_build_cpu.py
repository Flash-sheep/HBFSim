#!/usr/bin/env python3
"""Resume remaining isolated artifacts using the pinned private libbpf."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--attempt', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
w = a.work.resolve()
assert root not in w.parents and w != root
e = root / 'artifacts/llama2-layer15-full-validation-v1/framework-cpu'
old = json.loads((e / 'postbuild-v1.json').read_text())
assert old['error'].startswith('remaining-hbf-artifacts failed')
assert old['environment']['status'] == 'PASS' and old['torch_ptx']['members'] == 239
assert json.loads((e / 'build-v1.json').read_text())['status'] == 'PASS'
out = w / 'logs' / ('finish-build-' + a.attempt)
out.mkdir()
libbpf = w / 'cudart11-v1/build-agent-on/libbpf'
assert (libbpf / 'libbpf.so').resolve().is_file()
pkg = out / 'pkgconfig'
pkg.mkdir()
pc = pkg / 'libbpf.pc'
pc.write_text(f'''prefix={libbpf}
libdir=${{prefix}}
includedir=${{prefix}}
Name: libbpf
Description: Private pinned bpftime libbpf, no system installation
Version: 1.7.0
Libs: -L${{libdir}} -lbpf
Requires.private: libelf zlib
Cflags: -I${{includedir}}
''')
env = {k: v for k, v in os.environ.items() if not k.startswith(('HBFSIM_', 'BPFTIME_', 'CUDA_'))
       and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PKG_CONFIG_PATH']}
env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', CCACHE_DISABLE='1', CCACHE_DIR=str(w / 'private-ccache'),
           PATH=str(w / 'private-bin') + ':/usr/bin:/bin', PKG_CONFIG_PATH=str(pkg))
cmake = '/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/build-tools/cmake/data/bin/cmake'
hbf = w / 'cudart11-v1/build-gate-on'
commands = [
    [cmake, '-S', str(w / 'cudart11-v1/hbfsim-src'), '-B', str(hbf),
     '-DHBFSIM_LLVM_OBJDUMP_EXECUTABLE=/usr/lib/llvm-17/bin/llvm-objdump'],
    [cmake, '--build', str(hbf), '--parallel', '2', '--target', 'hbfsim', 'hbfsimd',
     'ptxpass_hbf', 'ptxpass_hbf_plugin', 'ptxpass_hbf_aggregate_plugin',
     'hbfsim_bpftime_attach_loader', 'hbfsim_coverage_probe'],
    ['/usr/bin/ninja', '-C', str(w / 'cudart11-v1/build-agent-on'), '-j1', 'bpftime-syscall-server'],
]
plan = {'scope': 'CPU isolated build recovery only; no loader/daemon/GPU execution',
        'reason': 'Earlier CMake configure had no libbpf pkg-config package; attach targets were absent',
        'commands': commands, 'watchdog_seconds': 1800, 'host_available_reserve_bytes': 64 * 2**30,
        'disk_free_reserve_bytes': 50 * 2**30, 'GPU_tasks_started': 0,
        'private_pc': str(pc), 'original_private_pc_unchanged': True,
        'completion_notification': 'NOT_CONFIGURED', 'DONE': str(out / 'DONE'), 'FAILED': str(out / 'FAILED')}
(e / ('finish-build-' + a.attempt + '.plan.json')).write_text(json.dumps(plan, indent=2) + '\n')
started = time.monotonic()
rows = []
result = {}
try:
    for i, cmd in enumerate(commands):
        with (out / (str(i) + '.stdout')).open('x') as stdout, (out / (str(i) + '.stderr')).open('x') as stderr:
            proc = subprocess.Popen(cmd, cwd=w, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            (e / ('finish-build-' + a.attempt + '.running.json')).write_text(json.dumps({
                'supervisor_pid': os.getpid(), 'pid': proc.pid, 'step': i, 'command': cmd,
                'DONE': str(out / 'DONE'), 'FAILED': str(out / 'FAILED')}, indent=2) + '\n')
            while proc.poll() is None:
                available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                     if x.startswith('MemAvailable:'))) * 1024
                if time.monotonic() - started > 1800 or available < 64 * 2**30 or shutil.disk_usage(w).free < 50 * 2**30:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise RuntimeError('resource/watchdog stop')
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        rows.append({'step': i, 'returncode': proc.returncode})
        if proc.returncode:
            raise RuntimeError('build step failed: ' + str(i))
    files = [hbf / name for name in ['libhbfsim.so', 'hbfsimd', 'libhbfsim_launch_gate.so',
             'libptxpass_hbf.so', 'libptxpass_hbf_aggregate.so', 'hbfsim_bpftime_attach_loader', 'coverage_probe.bpf.o']]
    files += [w / 'cudart11-v1/build-agent-on/runtime' / suffix for suffix in
              ['agent/libbpftime-agent.so', 'syscall-server/libbpftime-syscall-server.so']]
    files.append(libbpf / 'libbpf.so')
    artifacts = []
    for path in files:
        assert path.is_file(), path
        artifacts.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'bytes': path.stat().st_size})
    result.update(status='PASS', artifacts=artifacts)
except Exception as exc:
    result.update(status='FAILED', error=str(exc))
result.update(steps=rows, wall_seconds=time.monotonic() - started, GPU_tasks_started=0)
with (e / ('finish-build-' + a.attempt + '.json')).open('x') as f:
    json.dump(result, f, indent=2); f.write('\n')
(out / ('DONE' if result['status'] == 'PASS' else 'FAILED')).write_text(json.dumps(result) + '\n')
print(json.dumps(result))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
