#!/usr/bin/env python3
"""One approval-gated CUDA driver probe. Default mode performs CPU checks only."""
import argparse
import ctypes as C
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GPU = "GPU-5302a497-ea8c-bb7b-7235-f7464792093e"
KERNEL = "_Z18scal_64addr_kernelIffLi1ELb1ELi6ELi5ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_"
GUARD, ELEMENTS = 256, 64 * 32
INITIAL = b"\xa5" * GUARD + struct.pack("<f", 1.0) * ELEMENTS + b"\xa5" * GUARD
EXPECTED = b"\xa5" * GUARD + b"\0" * (4 * ELEMENTS) + b"\xa5" * GUARD


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def gate_module():
    spec = importlib.util.spec_from_file_location("eq3_experiment_gate", REPO / "tools/eq3_experiment_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_approval(path):
    gate = gate_module()
    manifest = gate.load_json(HERE / "manifest.json")
    approval = gate.load_json(path) if path else None
    return gate.validate_gate(manifest, approval, REPO, approval_path=path)


def verify_runtime():
    identity = json.loads((HERE / "evidence/runtime_identity.json").read_text())
    for item in identity["files"]:
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError("runtime identity changed: " + item["path"])
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("unexpected LD_PRELOAD")
    import shutil
    if shutil.disk_usage(HERE).free < 1024**3:
        raise RuntimeError("less than 1 GiB filesystem reserve")
    available = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
    if not available or int(available[1]) < 4 * 1024**2:
        raise RuntimeError("less than 4 GiB host memory reserve")
    return identity


def parameters():
    p = (C.c_uint32 * 12)()
    p[2], p[3], p[5], p[10] = 64, 32, 64, 0
    return p


def cpu_check():
    # Derived from the selected raw PTX, not from the kernel's suggestive name.
    raw = (HERE / "inputs/raw77.ptx").read_text()
    body = raw.split(".entry " + KERNEL + "(", 1)[1].split("\n.entry ", 1)[0]
    stores = re.findall(r"(?m)^st\.global\.u32\b", body)
    assert len(stores) == 12 and not re.search(r"\bld\.global\b", body)
    addresses = []
    for tid in range(256):
        lane, row = tid & 31, tid >> 5
        for row_step in (0, 8, 16, 24):
            for col_step in (0, 32, 64):
                col = lane + col_step
                if col <= 63:
                    addresses.append((row + row_step) * 64 + col)
    assert len(addresses) == ELEMENTS and sorted(addresses) == list(range(ELEMENTS))
    p = parameters()
    assert C.sizeof(p) == 48 and C.alignment(p) == 4
    assert bytes(p) == struct.pack("<12I", 0, 0, 64, 32, 0, 64, 0, 0, 0, 0, 0, 0)
    assert len(INITIAL) == len(EXPECTED) == 8704
    for arm in ("raw77", "raw87", "staged87"):
        assert (HERE / "inputs" / (arm + ".cubin")).read_bytes().startswith(b"\x7fELF")
    return {"status": "PASS", "gpu_initialized": False, "kernel": KERNEL,
            "parameter_bytes": 48, "elements": ELEMENTS, "allocation_bytes": len(INITIAL),
            "address_min": min(addresses), "address_max": max(addresses),
            "address_count": len(addresses), "each_element_written_once": True,
            "expected_output_sha256": hashlib.sha256(EXPECTED).hexdigest(),
            "limitation": "CPU checks do not establish GPU correctness or symbol lookup success"}


def gpu_snapshot():
    args = ["/usr/bin/nvidia-smi", "--id=" + GPU,
            "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    values = [x.strip() for x in subprocess.check_output(args, text=True, timeout=5).strip().split(",")]
    return {"uuid": values[0], "name": values[1], "driver": values[2],
            "memory_mib": int(values[3]), "utilization_percent": int(values[4]), "temperature_c": int(values[5])}


def foreign_processes(own_pid=None):
    text = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=5)
    return [line for line in text.splitlines() if line.startswith(GPU + ",") and int(line.split(",")[1]) != own_pid]


def worker(approval, output):
    check_approval(approval)
    identity = verify_runtime()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError("GPU UUID selection mismatch")
    # First CUDA operation occurs only here, after approval and identities pass.
    driver = C.CDLL(identity["driver_path"], mode=os.RTLD_LOCAL)

    def bind(name, types):
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = types, C.c_int
        return fn

    def call(fn, *args):
        status = fn(*args)
        if status:
            raise RuntimeError(fn.__name__ + " failed, CUresult=" + str(status))

    ptr = C.c_void_p
    init = bind("cuInit", [C.c_uint])
    getcount = bind("cuDeviceGetCount", [C.POINTER(C.c_int)])
    getdev = bind("cuDeviceGet", [C.POINTER(C.c_int), C.c_int])
    getuuid = bind("cuDeviceGetUuid_v2", [ptr, C.c_int])
    attribute = bind("cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int])
    create = bind("cuCtxCreate_v2", [C.POINTER(ptr), C.c_uint, C.c_int])
    destroy = bind("cuCtxDestroy_v2", [ptr])
    load = bind("cuModuleLoad", [C.POINTER(ptr), C.c_char_p])
    unload = bind("cuModuleUnload", [ptr])
    getfn = bind("cuModuleGetFunction", [C.POINTER(ptr), ptr, C.c_char_p])
    getglobal = bind("cuModuleGetGlobal_v2", [C.POINTER(C.c_uint64), C.POINTER(C.c_size_t), ptr, C.c_char_p])
    alloc = bind("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t])
    free = bind("cuMemFree_v2", [C.c_uint64])
    htod = bind("cuMemcpyHtoD_v2", [C.c_uint64, ptr, C.c_size_t])
    dtoh = bind("cuMemcpyDtoH_v2", [ptr, C.c_uint64, C.c_size_t])
    launch = bind("cuLaunchKernel", [ptr, *([C.c_uint] * 7), ptr, C.POINTER(ptr), C.POINTER(ptr)])
    sync = bind("cuCtxSynchronize", [])
    call(init, 0)
    count, dev = C.c_int(), C.c_int()
    call(getcount, C.byref(count))
    if count.value != 1:
        raise RuntimeError("expected exactly one visible GPU")
    call(getdev, C.byref(dev), 0)
    device_uuid = (C.c_ubyte * 16)()
    call(getuuid, C.byref(device_uuid), dev)
    if "GPU-" + str(uuid.UUID(bytes=bytes(device_uuid))) != GPU:
        raise RuntimeError("driver device UUID mismatch")
    major, minor = C.c_int(), C.c_int()
    call(attribute, C.byref(major), 75, dev)
    call(attribute, C.byref(minor), 76, dev)
    if (major.value, minor.value) != (8, 0):
        raise RuntimeError("expected sm_80")
    ctx = ptr()
    rows = []
    call(create, C.byref(ctx), 2, dev)
    try:
        for arm in ("raw77", "raw87", "staged87"):
            module, fn, memory = ptr(), ptr(), C.c_uint64()
            call(load, C.byref(module), str(HERE / "inputs" / (arm + ".cubin")).encode())
            try:
                if arm == "staged87":
                    for name in ("__hbfsim_control", "__hbfsim_control_generation"):
                        addr, size, value = C.c_uint64(), C.c_size_t(), C.c_uint64()
                        call(getglobal, C.byref(addr), C.byref(size), module, name.encode())
                        if size.value != 8:
                            raise RuntimeError("unexpected control symbol size")
                        call(dtoh, C.byref(value), addr, 8)
                        if value.value != 0:
                            raise RuntimeError("control must be unbound in this probe")
                call(getfn, C.byref(fn), module, KERNEL.encode())
                call(alloc, C.byref(memory), len(INITIAL))
                try:
                    initial = C.create_string_buffer(INITIAL, len(INITIAL))
                    call(htod, memory, initial, len(INITIAL))
                    p = parameters()
                    a, b, c = [C.c_uint64(memory.value + GUARD) for _ in range(3)]
                    args = (ptr * 4)(C.addressof(p), C.addressof(a), C.addressof(b), C.addressof(c))
                    t0 = time.monotonic()
                    call(launch, fn, 1, 1, 1, 256, 1, 1, 0, None, args, None)
                    call(sync)
                    elapsed = time.monotonic() - t0
                    result = C.create_string_buffer(len(INITIAL))
                    call(dtoh, result, memory, len(INITIAL))
                    data = result.raw
                    (output / (arm + ".output.bin")).write_bytes(data)
                    row = {"arm": arm, "launches": 1, "launch_and_sync_wall_s": elapsed,
                           "output_sha256": hashlib.sha256(data).hexdigest(),
                           "zero_output": data[GUARD:-GUARD] == b"\0" * (4 * ELEMENTS),
                           "guards_unchanged": data[:GUARD] == data[-GUARD:] == b"\xa5" * GUARD,
                           "matches_expected": data == EXPECTED}
                    rows.append(row)
                    save(output / "progress.json", {"completed_arms": rows})
                    if not row["matches_expected"]:
                        raise RuntimeError("output or guard mismatch: " + arm)
                finally:
                    if memory.value:
                        call(free, memory)
            finally:
                call(unload, module)
    finally:
        call(destroy, ctx)
    verify_runtime()
    save(output / "RESULT.json", {"status": "PASS", "gpu_uuid": GPU, "arms": rows,
         "evidence_type": "GPU_ENGINEERING_FIXTURE", "hbf_control": "UNBOUND_PASSTHROUGH",
         "hbf_timing_validated": False, "llama_workload_validated": False})


def execute(approval):
    check_approval(approval)
    verify_runtime()
    cpu_check()
    lock = open("/var/tmp/hbfsim-yjl/" + GPU + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = gpu_snapshot()
    if before["driver"] != "565.57.01" or before["memory_mib"] >= 1024 or before["utilization_percent"] != 0 or before["temperature_c"] >= 75 or foreign_processes():
        raise RuntimeError("GPU startup conditions not satisfied")
    output = HERE / "execution-v1"
    output.mkdir(exist_ok=False)
    save(output / "START.json", {"parent_pid": os.getpid(), "gpu_uuid": GPU, "snapshot": before, "time": time.time()})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    started = time.monotonic()
    reason = None
    with (output / "worker.stdout").open("w") as stdout, (output / "worker.stderr").open("w") as stderr:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--approval", str(approval)], env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        save(output / "PID.json", {"pid": child.pid, "parent_pid": os.getpid(), "done": str(output / "DONE"), "failed": str(output / "FAILED")})
        try:
            while child.poll() is None:
                if time.monotonic() - started > 60:
                    raise RuntimeError("60s watchdog exceeded")
                status = Path("/proc") / str(child.pid) / "status"
                if status.exists():
                    match = re.search(r"VmRSS:\s+(\d+)", status.read_text())
                    if match and int(match[1]) > 1024**2:
                        raise RuntimeError("worker RSS exceeded 1 GiB")
                snap = gpu_snapshot()
                if snap["temperature_c"] >= 80 or snap["memory_mib"] >= 1024 or foreign_processes(child.pid):
                    raise RuntimeError("GPU resource/occupancy stop condition")
                if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 64 * 1024**2:
                    raise RuntimeError("64 MiB output budget exceeded")
                time.sleep(1)
        except Exception as exc:
            reason = str(exc)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
        finally:
            child.wait()
    passed = child.returncode == 0 and reason is None and (output / "RESULT.json").exists()
    save(output / "FINISH.json", {"status": "PASS" if passed else "FAILED", "returncode": child.returncode, "reason": reason, "wall_s": time.monotonic() - started})
    (output / ("DONE" if passed else "FAILED")).write_text("PASS\n" if passed else "FAILED\n")
    if not passed:
        raise RuntimeError("GPU probe failed; inspect " + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu-check", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.approval)
    elif args.worker:
        worker(args.approval, HERE / "execution-v1")
    else:
        print(json.dumps(cpu_check(), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        sys.exit(2)
