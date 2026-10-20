# Isolated CUDA11 adapter for the Llama2 layer15 validation

The user requested a local branch split on2026-10-20. `original` preserves the
unmodified repository at `874a20f289b7bd587fa4048a788e12df2618444b`.
`sm_80支持` adds this isolated adapter package, patches, CPU preparation and
validation records; `main` stays at the original commit. CUDA/model binaries,
private environments and raw run artifacts remain in their existing external or
ignored directories. This branch records the current partial support: the
original full layer's native/strict output comparison passed, while full HBF
Timing validation remains incomplete.

Historical experiment manifests and confirmations retain their original commit,
hash and approval status. In particular, the pending v4 manifest binds `874a20f`;
before executing from a new support-branch commit, freeze a new execution version
against that commit and obtain its required approval. Do not rewrite old receipts
or reuse an old confirmation for the new revision. Branch separation starts no
GPU task and supplies no experiment approval.

The user approved `docs/llama2_layer15_live/MINIMAL_CUDART11_PLAN.md` with
“批准”. The approved text SHA256 is
`7ceb358d6e3e64a0642a630fb3efb55363d16c850ae593f8d4b61228e6d2df99`;
the original text and actual confirmation are preserved in
`artifacts/llama2-layer15-full-validation-v1/cuda11-plan-approval-v1/`.
This is approval for isolated compatibility implementation and fixed CPU tests.
It does not authorize GPU execution.

`apply_compat.py` only accepts external source copies. `source_identity.json`
binds the original HBF source and the ec26 agent plus its existing 0002 patch.
The two patches are the reviewable implementation; production source, default
builds, CUDA libraries, public HBF ABI, helper PTX, futures, TMA, cache, scheduling
and ownership code are unchanged. Application output uses new attempt directories.

Both source copies add `HBFSIM_EXPERIMENTAL_CUDART11`, default OFF. ON adds the
true `libcudart.so.11.0` domain, its public host-stub launch, version-specific
registration, and the same strict native binding guards. CUDA11 PTSZ/cooperative
launches explicitly reject; no CUDA11 private kernel entry is fabricated. The
CUDA11 route requires strict policy. Existing CUDA12/13 conditions are preserved.

Apply `agent-cuda11-provider.patch` after the base agent patch. Its separate
identity file binds the base-patched inputs and final source hashes. The
incremental cold-start fix acquires the exact CUDA11 provider before collecting
registration hooks; only the new domain uses it. It leaves the existing 12/13
lookups and cleanup code unchanged. `apply_provider_fix.py` reproduces this step
on an external base-patched copy and writes a new patch/identity output pair.

Actual copies/builds are below
`/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/cudart11-v1/`. Reproduce CPU builds with:

```bash
python3 -B experiments/llama2_sm80_cuda11/build_cpu.py gate-on --work /path/to/isolated/work --attempt NEW
python3 -B experiments/llama2_sm80_cuda11/build_cpu.py gate-off --work /path/to/isolated/work --attempt NEW
python3 -B experiments/llama2_sm80_cuda11/build_cpu.py agent-on --work /path/to/isolated/work --attempt NEW
python3 -B experiments/llama2_sm80_cuda11/build_cpu.py agent-off --work /path/to/isolated/work --attempt NEW
python3 -B experiments/llama2_sm80_cuda11/test_cpu.py gate --work /path/to/isolated/work --attempt NEW
python3 -B experiments/llama2_sm80_cuda11/test_cpu.py registry --work /path/to/isolated/work --attempt NEW
python3 -B experiments/llama2_sm80_cuda11/test_cpu.py registry --cold-start --work /path/to/isolated/work --attempt NEW
python3 -B experiments/llama2_sm80_cuda11/test_cpu.py registry --cold-start --agent-profile agent-off --work /path/to/isolated/work --attempt NEW
```

The current-host scripts explicitly name the already isolated Python/CUDA,
Boost1.88 and LLVM17 toolchains; they are not portable environment locks or a
claim that the old host's build lock passed. Each build profile runs serially,
with gate jobs2 or agent Ninja1/nested make2 and an internal resource watchdog.
The agent executes the existing `copy_headers` target before compilation.

The gate tests run real OFF/ON gate DSOs with inert CUDA fixtures, retain 12
legacy cases per build, and exercise eight CUDA11 cases plus export/policy
boundaries. The registry fixture links the real agent implementation and uses
actual Gum registration callbacks against fake11/12/13 DSOs. It tests same-value
handle separation, reuse/unregister, synthetic stale generation, incorrect
container/entry/domain rejection, and unready/non-strict launch rejection. All
CUDA driver calls remain mocked; no kernel is launched. Positive real PTX binding
and GPU numerical/HBF coverage remain unvalidated.

`layer_acceptance.py` is a read-only, CPU receipt checker prepared for a future
GPU wrapper; that wrapper is not integrated yet. It requires nine separate
weight scopes, actual modeled gate addresses, complete access/byte counters,
ordinary request conservation and exact FP16 output bytes. Fixed-delay traces
replace the service path and are explicitly refused as ordinary Timing proof.
Six synthetic CPU rejection fixtures pass; they are not GPU evidence. The CLI
refuses fixture receipts. Supplied receipts still require independently checked
input, staging/binding and log provenance; this checker does not authenticate
their authorship or establish whole-weight byte coverage.

The first registry test failed during normal exit: dropping the agent version
script exposed static NVPTXCompiler symbols, conflicting with its compiler DSO.
The fixed external fixture hides those archive symbols, matching production
visibility. Normal destruction/exit now passes; no agent lifecycle change or
exit bypass was used. The failure and GDB trace remain preserved.

`sync_torch_sources.py` resumes original framework gitlinks with bounded network
retries. `framework_cpu.py` refuses wrong/unpopulated submodules, performs only
source configuration/construction, uses explicit sm80+PTX and a task watchdog,
and refuses a framework build before the adapter builds and CUDA11/architecture
configuration checks pass. Its optional backend exclusions retain the original
eager operator source; this is a custom build, not wheel-byte equivalence.
These framework preparation scripts are not a GPU execution entry point.
`postbuild_cpu.py` waits for the framework receipt, installs only the resulting
private wheel, verifies the original input bytes and fixed CPU forward, collects
actual embedded PTX, and builds the remaining private runtime artifacts. It does
not execute the attach loader, daemon or GPU workload. Its DONE/FAILED markers
and process identity are persisted; external completion notifications are not
configured.

Persistent evidence and the current limits are in
[`CUDA11_COMPATIBILITY_PROGRESS.md`](../../docs/llama2_layer15_live/CUDA11_COMPATIBILITY_PROGRESS.md).

`finish_build_cpu.py` recovers the missing attach targets using a task-local
pkg-config file for the already pinned private libbpf; it does not install
system packages or run the loader. The framework and remaining artifacts now
build successfully. The original postbuild failure is retained.

`native_discovery.py` is a separate approval-gated native baseline/capture
workflow. Default execution only verifies CPU identities. GPU mode runs one
original layer forward in each of two fresh processes, records actual strict
CUDA11/Driver observations, and requires exact output bytes. It registers no
HBF ranges and loads no staged PTX; no instrumentation activation or HBF service
PASS is implied. `prepare_discovery_cpu.py` freezes external file identities
and prepares a PENDING manifest without creating approval. The original
production wrapper's older provenance checks remain untouched. This separate
workflow binds the actual current-host build receipts and exact binaries.

The approved v1 was executed: native original layer15 passed, then capture was
stopped by the resource/foreign-process guard before its forward. v1 code and
raw evidence remain unchanged. A CPU-only marker reproduced inherited agent
preloads in Git control subprocesses. `native_discovery_v2.py` removes the child
preload environment while retaining the current process mappings, keeps all
resource thresholds/main-PID checks, and records explicit samples/exit codes.
It requests only one capture forward against the audited immutable successful
v1 native output. v2 was explicitly approved and executed once, then failed with45 unversioned rejected launches and a nonmatching output. No automatic retry occurred. See the v2 result linked below.

`version_copy.py` produces a new immutable PTX copy and a DERIVED receipt. It
accepts only original sm80/PTX7.4 or7.7 and changes exactly the version token to
8.7. It rejects sm90 and refuses overwriting an attempt. Use it after the ordinary
transform so the original raw member/module identity is retained; preserve the
transform attempt and generate a new provenance chain for the copied staged
hash. It does not rewrite an old pass/compile receipt, waive unsupported
instructions, establish assembly success, or prove GPU equivalence.


`reproduce_launch_lookup_cpu.py` demonstrates the actual ELF consumer relocation:
with the frozen gate, a versioned CUDA11 import selects its unversioned rejection
stub. `fix_launch_lookup_cpu.py` creates a separate ON-only version map and relinks
unchanged original objects into a new gate, preserving the frozen binaries/source.
The rejected unversioned launch receives a distinct version node; HBF exports,
explicit CUDA11/12/13 aliases and all function bodies are unchanged. The fixed
CPU consumer selects the exact11 gate and calls only its inert11 provider.
Twelve legacy cases, eight CUDA11 cases and the real-agent registration/unready
rejection fixture pass with mock CUDA providers. This is within the approved
isolated CUDA11 version-symbol extension; no production/default build changes.

`native_discovery_v3.py` retains the approved resource/numerical thresholds and
requires actual successful gate coverage plus strict runtime receipts. Existing
strict ORIGINAL completion can use the gate's authenticated one-use handshake
without a Driver interception; observed Driver calls still require successful
matching bridge records. Raw forward windows are saved before validation. The
v3 bundle was explicitly approved and executed once: exact native FP16 bytes,45 successful cudart11 ORIGINAL-completion records and no HBF ranges. Its raw evidence is immutable.

[v2 observed result](../../docs/llama2_layer15_live/NATIVE_DISCOVERY_V2_RESULT.md),
[v3 preflight](../../docs/llama2_layer15_live/NATIVE_DISCOVERY_PREFLIGHT_V3.md).


The v3 result and post-capture CPU preparation are recorded in
[v3 actual result](../../docs/llama2_layer15_live/NATIVE_DISCOVERY_V3_RESULT.md).
`stage_observed_torch_cpu.py` joins actual successful launch/registration names to
unique original Torch PTX members, retains failed transforms/assembly, and changes
only the version token in new transformed copies. `assertfail_proof_cpu.py` proves
the exact static diagnostic setup for existing assertfail calls, without removing
them or assuming the branch unreachable. `finish_observed_torch_cpu.py` uses the
existing compiler register option for fixed failed cases. `compose_observed_torch_stage_cpu.py`
publishes a new scoped join for19 entries/11 members; the reduction entry remains
unsupported (`atom.global.add.u32`). Selected-entry relocatable objects are not
runtime full-module link/load or GPU validation. All original failures remain.

`cupti_observer.cpp` is an independent optional DSO with no constructor. Explicit
start subscribes to CUPTI11.7 Driver callbacks; callbacks copy launch/lookup
metadata without CUDA API calls or changing arguments/results. Unknown launch
layouts remain visible. `build_cupti_observer_cpu.py` tests against inert CUPTI,
without mapping a real CUDA driver. This is a passive diagnostic, not a replacement
for the HBF gate or a claim that callbacks prove instrumentation interception.
`native_discovery_v4.py` adds scoped observer records and requires paired successful
actual Driver launches covering the known Torch set and at least one extra entry,
plus all existing strict/numerical/safety checks. Its ten CPU tests and actual
missing-approval CLI refusals pass. The v4 bundle remains PENDING_USER_APPROVAL;
no v4 GPU run has been started. See [v4 preflight](../../docs/llama2_layer15_live/NATIVE_DISCOVERY_PREFLIGHT_V4.md).
