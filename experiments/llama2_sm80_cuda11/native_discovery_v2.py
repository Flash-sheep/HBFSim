#!/usr/bin/env python3
"""Capture-only v2: isolate child preloads; reuse the immutable v1 native control."""
import argparse
import ctypes as C
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import struct
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
GPU = 'GPU-5302a497-ea8c-bb7b-7235-f7464792093e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(16 * 2**20), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def approval_gate(bundle, approval):
    spec = importlib.util.spec_from_file_location('experiment_gate', ROOT / 'tools/eq3_experiment_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate.validate_gate(gate.load_json(bundle / 'manifest.json'),
                              gate.load_json(approval) if approval else None,
                              ROOT, approval_path=approval)


def verify(bundle, tensors=False):
    identity = json.loads((bundle / 'runtime-identity.json').read_text())
    for row in identity['files']:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('runtime/input file changed: ' + row['path'])
    inputs = json.loads((ROOT / identity['input_identity']).read_text())
    for row in inputs['files']:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('original input changed: ' + row['path'])
    if tensors:
        import torch
        from safetensors import safe_open
        assert not torch.cuda.is_initialized()
        assert torch.version.git_version == 'e4ee3be4063b7c430974252fdf7db42273388d86'
        assert torch.version.cuda == '11.7' and not torch._C._GLIBCXX_USE_CXX11_ABI
        for row in inputs['tensors']:
            with safe_open(row['shard'], framework='pt', device='cpu') as f:
                raw = f.get_tensor(row['name']).contiguous().view(torch.uint8).numpy().tobytes()
                assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        assert not torch.cuda.is_initialized()
    return identity


def base_env(identity):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HBFSIM_', 'BPFTIME_', 'CUDA_', 'NVIDIA_'))
           and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']}
    env.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               LD_LIBRARY_PATH=identity['library_path'],
               PATH=str(Path(identity['python']).parent) + ':/usr/bin:/bin',
               CUDA_MODULE_LOADING='EAGER')
    return env


def snapshot():
    values = subprocess.check_output(['/usr/bin/nvidia-smi', '--id=' + GPU,
        '--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip().split(',')
    v = [x.strip() for x in values]
    return dict(uuid=v[0], name=v[1], driver=v[2], memory_mib=int(v[3]),
                utilization_percent=int(v[4]), temperature_c=int(v[5]))


def foreign(owned):
    rows = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                  '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
    return [row for row in rows if row.startswith(GPU + ',') and int(row.split(',')[1]) not in owned]


def gpu_stop_reasons(snap, unexpected):
    reasons = []
    if snap['temperature_c'] >= 80:
        reasons.append('temperature>=80C')
    if snap['memory_mib'] >= 12288:
        reasons.append('GPU memory>=12288MiB')
    if unexpected:
        reasons.append('unexpected GPU process')
    return reasons


def log_window(path, start, end):
    if not path.exists():
        assert start == end == 0
        return []
    with path.open('rb') as f:
        f.seek(start); raw = f.read(end - start)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def isolate_worker_preload():
    os.environ.pop('LD_PRELOAD', None)


def worker(bundle, approval, arm):
    if arm != 'capture':
        raise RuntimeError('v2 permits only the capture arm')
    # Already loaded DSOs stay active in this process. Control-plane children
    # (notably Git in the approval gate) must not bootstrap another CUDA agent.
    isolate_worker_preload()
    approval_gate(bundle, approval)
    identity = verify(bundle)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    out = bundle / 'execution-v1' / arm
    assert out.is_dir() and not (out / 'RESULT.json').exists()
    # All CUDA initialization, including preloaded agent constructors in capture,
    # is protected by the parent's gate before process creation.
    import torch
    sys.path.insert(0, str(ROOT / 'artifacts/llama2-layer15-full-validation-v1/frozen'))
    import trace_decoder_layer as decoder
    import trace_flint_decoder_layer as flint
    assert torch.version.cuda == '11.7'
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (8, 0)
    driver = C.CDLL(identity['driver'], mode=os.RTLD_LOCAL)
    get_uuid = driver.cuDeviceGetUuid_v2
    get_uuid.argtypes = [C.c_void_p, C.c_int]; get_uuid.restype = C.c_int
    device_uuid = (C.c_ubyte * 16)()
    assert get_uuid(C.byref(device_uuid), 0) == 0
    assert 'GPU-' + str(uuid.UUID(bytes=bytes(device_uuid))) == GPU
    args = argparse.Namespace(model_dir=Path('/mnt/llama2-7b'), layer_index=15,
                              context_length=32, warmup_steps=0, repeats=1)
    config, locations = decoder.validate_inputs(args)
    torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    layer = decoder.load_layer(config, locations, 15).to(device=device, dtype=torch.float16).eval()
    arena, tensors, payload_bytes = flint.place_weights_in_arena(layer, 256)
    expected = json.loads((ROOT / 'artifacts/llama2-layer15-full-validation-v1/frozen/original_weight_manifest.json').read_text())
    assert len(tensors) == 9 and payload_bytes == arena.numel() * 2 == 404766720
    for actual, ref in zip(tensors, expected['tensors']):
        assert all(actual[key] == ref[key] for key in ['name', 'shape', 'offset_bytes', 'size_bytes'])
    manifest = dict(schema='flint-contiguous-weight-arena-v1', layer_index=15,
                    batch_size=1, query_length=1, context_length=32, dtype='float16',
                    hbf_base_addr=arena.data_ptr(), hbf_size=arena.numel() * 2,
                    weight_payload_bytes=payload_bytes, tensor_count=9, tensors=tensors)
    save(out / 'arena.json', manifest)
    gen = torch.Generator(device='cpu').manual_seed(1)
    hidden = torch.randn(1, 1, 4096, dtype=torch.float16, generator=gen).to(device)
    positions = torch.tensor([[31]], dtype=torch.long).to(device)
    key = torch.randn(1, 32, 31, 128, dtype=torch.float16, generator=gen).to(device)
    value = torch.randn(key.shape, dtype=torch.float16, generator=gen).to(device)
    rotary = None
    if 'position_embeddings' in decoder.inspect.signature(layer.forward).parameters:
        if getattr(config, '_attn_implementation', None) is None:
            config._attn_implementation = 'eager'
        rotary = decoder._make_rotary_embedding(config, device)
    paths = {name: out / (name + '.jsonl') for name in ['coverage', 'strict-runtime', 'strict-bridge']}
    torch.cuda.synchronize()
    begin = {name: path.stat().st_size if path.exists() else 0 for name, path in paths.items()}
    save(out / 'FORWARD_BEGIN.json', {'log_offsets': begin, 'forward_calls': 1, 'warmup_calls': 0})
    started = time.monotonic()
    with torch.inference_mode():
        kwargs = decoder._prepare_decoder_call(layer, config, 15, hidden, positions, key, value, rotary)
        result = layer(hidden, **kwargs)
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    end = {name: path.stat().st_size if path.exists() else 0 for name, path in paths.items()}
    save(out / 'FORWARD_END.json', {'log_offsets': end, 'host_forward_wall_seconds': elapsed})
    tensor = result[0] if isinstance(result, tuple) else result
    assert tuple(tensor.shape) == (1, 1, 4096) and tensor.dtype == torch.float16
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    assert len(raw) == 8192 and all(math.isfinite(x) for x in struct.unpack('<4096e', raw))
    with (out / 'hidden-output.bin').open('xb') as f:
        f.write(raw)
    windows = {name: log_window(path, begin[name], end[name]) for name, path in paths.items()}
    if arm == 'capture':
        coverage, bridge = windows['coverage'], windows['strict-bridge']
        assert coverage and bridge, 'No actual forward coverage/strict bridge records'
        assert all(row['allowed'] and not row['modeled'] and not row['requires_instrumented_execution'] for row in coverage)
        assert all(row['gate_decision'] == 1 and row['selected_path'] == 'ORIGINAL' and row['cuda_result'] == 0
                   and row['selected_function'] == row['original_function'] for row in bridge)
        runtime = windows['strict-runtime']
        assert runtime and any(row['runtime_domain'] == 'cudart11' for row in runtime), 'No CUDA11 runtime observation'
        assert all(row['runtime_result'] == 0 for row in runtime), 'Failed runtime launch'
        save(out / 'FORWARD_CAPTURE.json', windows)
    mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if line.split()[-1].startswith('/')})
    save(out / 'loaded-files.json', {'paths': mapped})
    save(out / 'RESULT.json', dict(status='NATIVE_FORWARD_PASS_HBF_NOT_RUN', arm=arm,
        gpu_uuid=GPU, dtype='float16', shape=[1, 1, 4096], output_sha256=hashlib.sha256(raw).hexdigest(),
        forward_calls=1, warmup_calls=0, original_helper=True, registered_HBF_ranges=0,
        HBF_timing_validated=False, host_forward_wall_seconds=elapsed,
        limits='Host time includes observation overhead; no GPU performance or physical calibration claim'))


def execute(bundle, approval):
    approval_gate(bundle, approval)
    if os.environ.get('LD_PRELOAD'):
        raise RuntimeError('unexpected caller preload')
    identity = verify(bundle, tensors=True)
    lock = open('/var/tmp/hbfsim-yjl/' + GPU + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = snapshot()
    assert initial['uuid'] == GPU and initial['driver'] == '565.57.01'
    assert initial['memory_mib'] < 1024 and initial['temperature_c'] < 75
    assert initial['utilization_percent'] == 0 and not foreign(set()), 'target GPU busy'
    out = bundle / 'execution-v1'; out.mkdir()
    started = time.monotonic()
    actors = []
    files = []
    shm = None
    shm_stat = None
    result = {}
    def interrupted(signum, _frame):
        raise RuntimeError('launcher interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save(out / 'START.json', {'parent_pid': os.getpid(), 'snapshot': initial, 'GPU': GPU, 'time': time.time()})

    def monitor():
        if time.monotonic() - started > 180:
            raise RuntimeError('180s execution watchdog')
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
        if available < 64 * 2**30 or shutil.disk_usage(out).free < 2 * 2**30:
            raise RuntimeError('host resource reserve')
        rss = 0
        cpu = 0.0
        for task in Path('/proc').iterdir():
            if not task.name.isdecimal():
                continue
            try:
                text = (task / 'stat').read_text(); fields = text[text.rindex(')') + 2:].split()
                if int(fields[2]) in {p.pid for p in actors}:
                    rss += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
                    cpu += (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
            except (OSError, ValueError, IndexError):
                pass
        if rss > 16 * 2**30:
            raise RuntimeError('16GiB process group RSS stop')
        for whom in [resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN]:
            usage = resource.getrusage(whom)
            cpu += usage.ru_utime + usage.ru_stime
        if cpu > 720:
            raise RuntimeError('0.2 CPU-hour budget stop')
        snap = snapshot()
        owned = {p.pid for p in actors if p.poll() is None}
        unexpected = foreign(owned)
        sample = {'wall_seconds': time.monotonic() - started, 'gpu': snap,
                  'allowed_main_pids': sorted(owned), 'unexpected_GPU_rows': unexpected,
                  'process_group_RSS_bytes': rss, 'CPU_seconds': cpu,
                  'actor_returncodes': {str(p.pid): p.returncode for p in actors}}
        with (out / 'resource-samples.jsonl').open('a') as log:
            log.write(json.dumps(sample) + '\n')
        reasons = gpu_stop_reasons(snap, unexpected)
        if reasons:
            save(out / 'SAFETY_STOP.json', dict(sample, reasons=reasons))
            raise RuntimeError('GPU safety stop: ' + ', '.join(reasons))
        if sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > 2 * 2**30:
            raise RuntimeError('2GiB output budget stop')

    def spawn(cmd, env, label):
        approval_gate(bundle, approval)
        verify(bundle)
        monitor()
        stdout = (out / (label + '.stdout')).open('x')
        stderr = (out / (label + '.stderr')).open('x')
        files.extend([stdout, stderr])
        parent_pid = os.getpid()
        libc = C.CDLL(None)
        def child_guard():
            # A killed supervisor must not leave a detached GPU actor behind.
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
                os._exit(125)
            os.nice(10)
        proc = subprocess.Popen(cmd, cwd=out, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=True, preexec_fn=child_guard)
        actors.append(proc)
        save(out / (label + '.PID.json'), {'pid': proc.pid, 'command': cmd})
        return proc

    def finish(proc):
        while proc.poll() is None:
            monitor()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if proc.returncode:
            raise RuntimeError('child failed pid=' + str(proc.pid) + ', rc=' + str(proc.returncode))

    try:
        env = base_env(identity)
        worker_cmd = [identity['python'], '-B', str(Path(__file__).resolve()), '--bundle', str(bundle),
                      '--approval', str(approval), '--worker']
        (out / 'tmp').mkdir()
        env['TMPDIR'] = str(out / 'tmp')
        capture = out / 'capture'; capture.mkdir()
        (capture / 'cuda-cache').mkdir()
        env['CUDA_CACHE_PATH'] = str(capture / 'cuda-cache')
        stage = out / 'empty-ptx'; stage.mkdir()
        ns = 'hbfsim_llama15_' + uuid.uuid4().hex
        shm = Path('/dev/shm') / ns
        assert not shm.exists()
        ready = out / 'attach-ready.txt'
        env.update(BPFTIME_GLOBAL_SHM_NAME=ns, BPFTIME_SHM_MEMORY_MB='64',
            BPFTIME_LOG_OUTPUT=str(out / 'bpftime-runtime.log'),
            BPFTIME_CUDA_DEVICE='0', BPFTIME_CUDA_DISABLE_CUOBJDUMP='1',
            BPFTIME_CUDA_LATE_PTX_DIR=str(stage), BPFTIME_CUDA_LATE_PTX_PREPATCHED='1',
            BPFTIME_CUDA_ROOT=identity['cuda12'], BPFTIME_PTXPASS_LIBRARIES=identity['plugin'],
            BPFTIME_PTXPASS_DIR=str(ROOT / 'configs/ptxpass'), HBFSIM_BUILD_DIR=identity['gate_build'],
            HBFSIM_INSTRUMENTATION_POLICY='strict', HBFSIM_COVERAGE_PATH=str(capture / 'coverage.jsonl'),
            HBFSIM_PASS_MANIFEST_PATH=str(capture / 'pass-manifests.jsonl'),
            HBFSIM_NATIVE_REGISTRATION_LOG_PATH=str(capture / 'native-registration.jsonl'),
            HBFSIM_STRICT_RUNTIME_LOG_PATH=str(capture / 'strict-runtime.jsonl'),
            HBFSIM_STRICT_BRIDGE_LOG_PATH=str(capture / 'strict-bridge.jsonl'))
        loader = spawn([identity['loader'], identity['probe'], str(ready)],
                       dict(env, LD_PRELOAD=identity['server']), 'loader')
        deadline = time.monotonic() + 20
        while not ready.exists():
            monitor()
            if loader.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('attach loader unavailable/timeout')
            time.sleep(0.1)
        lines = ready.read_text().splitlines()
        assert lines[:3] == ['HBFSIM_BPFTIME_ATTACH_READY v1', 'shm=bpftime', 'attach_type=8']
        assert any(re.fullmatch(r'entries=[1-9][0-9]*', line) for line in lines)
        assert shm.is_file() and shm.stat().st_uid == os.getuid(), 'private SHM missing'
        shm_stat = shm.stat()
        save(out / 'SHM_IDENTITY.json', {'name': ns, 'device': shm_stat.st_dev, 'inode': shm_stat.st_ino,
                 'owner_uid': shm_stat.st_uid, 'legacy_ready_shm_line_is_not_namespace_identity': True})
        finish(spawn(worker_cmd + ['capture'], dict(env, LD_PRELOAD=identity['agent'] + ':' + identity['gate']), 'capture-worker'))
        baseline = ROOT / identity['baseline']['output']
        native = baseline.read_bytes()
        assert sha(baseline) == identity['baseline']['output_sha256']
        captured = (capture / 'hidden-output.bin').read_bytes()
        assert native == captured, 'native/captured FP16 output bytes differ'
        monitor()
        verify(bundle)
        result.update(status='NATIVE_DISCOVERY_PASS_HBF_NOT_RUN', output_comparison='EXACT_FP16_BYTES',
                      output_sha256=hashlib.sha256(native).hexdigest(), GPU_forward_calls=1,
                      native_baseline=str(baseline), native_baseline_reused=True,
                      HBF_timing_validated=False, registered_HBF_ranges=0,
                      limits='Kernel/container staging and HBF binding/coverage remain separate unresolved checks')
    except Exception as exc:
        result.update(status='FAILED', error=str(exc), HBF_timing_validated=False)
    finally:
        # Cleanup is finite and cannot itself be interrupted halfway through.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for proc in reversed(actors):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for f in files:
            f.close()
        if shm_stat is not None and shm.exists():
            current = shm.stat()
            if (current.st_dev, current.st_ino, current.st_uid) == (shm_stat.st_dev, shm_stat.st_ino, os.getuid()):
                shm.unlink()
        result['wall_seconds'] = time.monotonic() - started
        result['actor_exit_codes'] = {str(p.pid): p.returncode for p in actors}
        try:
            result['final_snapshot'] = snapshot()
        except Exception as exc:
            result.update(status='FAILED', final_snapshot_error=str(exc))
        save(out / 'RESULT.json', result)
        (out / ('DONE' if result['status'] != 'FAILED' else 'FAILED')).write_text(json.dumps(result) + '\n')
    print(json.dumps(result))
    if result['status'] == 'FAILED':
        raise RuntimeError('discovery failed; raw logs retained')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--approval', type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', choices=['capture'], help=argparse.SUPPRESS)
    a = p.parse_args(); bundle = a.bundle.resolve()
    if a.approval is not None:
        a.approval = a.approval.resolve()
    if a.execute:
        execute(bundle, a.approval)
    elif a.worker:
        worker(bundle, a.approval, a.worker)
    else:
        verify(bundle, tensors=True)
        print(json.dumps({'status': 'CPU_IDENTITY_PASS', 'CUDA_initialized': False, 'GPU_tasks_started': 0}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
