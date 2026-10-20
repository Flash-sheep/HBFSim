#!/usr/bin/env python3
"""Reproduce the inherited-preload defect with an inert CPU-only marker DSO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--attempt', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
out = a.work.resolve() / ('logs/preload-inheritance-cpu-' + a.attempt)
out.mkdir()
source = out / 'marker.c'
source.write_text(r'''#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
__attribute__((constructor)) static void cpu_marker(void) {
    const char *trace = getenv("HBFSIM_CPU_PRELOAD_TRACE");
    if (!trace) return;
    char exe[4096]; ssize_t n=readlink("/proc/self/exe",exe,sizeof(exe)-1);
    if(n<0) return; exe[n]=0;
    FILE *f=fopen(trace,"a"); if(!f) return;
    fprintf(f,"%ld\t%ld\t%s\n",(long)getpid(),(long)getppid(),exe); fclose(f);
}
''')
marker = out / 'cpu-marker.so'
cmd = ['/usr/bin/gcc-11', '-O2', '-shared', '-fPIC', str(source), '-o', str(marker)]
cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
(out / 'compile.stdout').write_text(cp.stdout); (out / 'compile.stderr').write_text(cp.stderr)
assert cp.returncode == 0
program = out / 'gate_only.py'
program.write_text('''import importlib.util,sys
from pathlib import Path
root=Path(sys.argv[1]);kind=sys.argv[2]
filename='native_discovery_v2.py' if kind=='fixed' else 'native_discovery.py'
spec=importlib.util.spec_from_file_location('discovery',root/'experiments/llama2_sm80_cuda11'/filename)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if kind=='fixed':module.isolate_worker_preload()
bundle=root/'artifacts/llama2-sm80-native-discovery-v1'
print(module.approval_gate(bundle,bundle/'USER_CONFIRMATION.json'))
''')
python = str(a.work.resolve() / 'torch-cu117-env/bin/python')
rows = []
for label in ['before', 'fixed']:
    trace = out / (label + '.tsv')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HBFSIM_', 'BPFTIME_', 'CUDA_'))
           and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']}
    env.update(CUDA_VISIBLE_DEVICES='', LD_PRELOAD=str(marker), HBFSIM_CPU_PRELOAD_TRACE=str(trace))
    cmd = [python, '-B', str(program), str(root), label]
    cp = subprocess.run(cmd, env=env, cwd=root, capture_output=True, text=True, timeout=60)
    (out / (label + '.stdout')).write_text(cp.stdout); (out / (label + '.stderr')).write_text(cp.stderr)
    assert cp.returncode == 0, cp.stderr
    events = [dict(zip(['pid', 'ppid', 'exe'], line.split('\t'))) for line in trace.read_text().splitlines()]
    git_events = [row for row in events if Path(row['exe']).name == 'git']
    if label == 'before':
        assert len(git_events) >= 2, events
    else:
        assert len(events) == 1 and not git_events, events
    rows.append({'case': label, 'returncode': cp.returncode, 'events': events,
                 'git_children_loading_preload': len(git_events)})
result = {'status': 'PASS', 'scope': 'Real Git subprocesses with inert CPU constructor; no CUDA library or GPU workload',
          'defect_reproduced': True, 'fixed_git_preload_count': 0, 'rows': rows,
          'GPU_tasks_started': 0, 'production_core_or_ABI_changed': False,
          'marker_sha256': hashlib.sha256(marker.read_bytes()).hexdigest(),
          'v2_source_sha256': hashlib.sha256((root / 'experiments/llama2_sm80_cuda11/native_discovery_v2.py').read_bytes()).hexdigest()}
receipt = root / 'artifacts/llama2-sm80-native-discovery-v1/analysis-v1/PRELOAD_INHERITANCE_CPU.json'
with receipt.open('x') as f:
    json.dump(result, f, indent=2); f.write('\n')
print(json.dumps(result))
