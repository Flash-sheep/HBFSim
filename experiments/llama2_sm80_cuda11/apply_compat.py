#!/usr/bin/env python3
"""Apply the approved default-off CUDA11 port to isolated source copies only."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--hbf-source',type=Path,required=True)
p.add_argument('--agent-source',type=Path,required=True)
p.add_argument('--patch-output',type=Path,required=True)
a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2]
for src in (a.hbf_source,a.agent_source):
    assert src.resolve()!=ROOT and ROOT not in src.resolve().parents, 'Only external isolated source copies are allowed'
identity=json.loads((Path(__file__).parent/'source_identity.json').read_text())
expected={}
for kind,source in [('gate',a.hbf_source),('agent',a.agent_source)]:
    for item in identity['files'][kind]:
        expected[(source,item['path'])]=item['before_sha256']
before={}
def confined(root,name):
    path=root/name
    assert root.resolve() in path.resolve().parents, ('Source file escapes isolated copy',path)
    return path
def read(root,name):
    path=confined(root,name)
    s=path.read_text()
    if (root,name) in expected:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[(root,name)], ('Source identity drift',path)
    before.setdefault((root,name),s)
    return s
def save(root,name,s):
    path=confined(root,name)
    if expected.get((root,name),'existing') is None:
        assert not path.exists(), ('New adapter file already exists',path)
    path.write_text(s)
def replace(s,old,new,count=1):
    assert s.count(old)==count,(old[:100],s.count(old),count)
    return s.replace(old,new)
H=a.hbf_source; A=a.agent_source
flag='HBFSIM_EXPERIMENTAL_CUDART11'
s=read(H,'CMakeLists.txt')
s=replace(s,'option(HBFSIM_GATE_TEST_HOOKS',f'option({flag} "Isolated experimental CUDA11 launch identity adapter" OFF)\noption(HBFSIM_GATE_TEST_HOOKS')
s=replace(s,'    if(HBFSIM_GATE_CUDART_VERSION_MAP)\n',f'''    if({flag})
        target_compile_definitions(hbfsim_launch_gate PRIVATE {flag}=1)
        target_link_options(hbfsim_launch_gate PRIVATE "LINKER:--version-script=${{CMAKE_CURRENT_SOURCE_DIR}}/cmake/cudart11-experiment.map")
        set_property(TARGET hbfsim_launch_gate APPEND PROPERTY LINK_DEPENDS "${{CMAKE_CURRENT_SOURCE_DIR}}/cmake/cudart11-experiment.map")
    elseif(HBFSIM_GATE_CUDART_VERSION_MAP)
''')
save(H,'CMakeLists.txt',s)
before[(H,'cmake/cudart11-experiment.map')]=''
save(H,'cmake/cudart11-experiment.map','libcudart.so.11.0 {};\nlibcudart.so.12 {} libcudart.so.11.0;\nlibcudart.so.13 {} libcudart.so.12;\n')
s=read(H,'src/cuda_runtime/launch_gate.cpp')
s=replace(s,'enum class RuntimeDomain { Cudart12, Cudart13 };',f'''enum class RuntimeDomain {{ Cudart12, Cudart13
#if {flag}
    , Cudart11
#endif
}};''')
s=replace(s,'const char* runtime_version(RuntimeDomain domain) noexcept\n{',f'''const char* runtime_version(RuntimeDomain domain) noexcept
{{
#if {flag}
    if (domain == RuntimeDomain::Cudart11) return "libcudart.so.11.0";
#endif''')
s=replace(s,'const char* runtime_soname(RuntimeDomain domain) noexcept',f'''const char* runtime_domain_name(RuntimeDomain domain) noexcept
{{
#if {flag}
    if (domain == RuntimeDomain::Cudart11) return "cudart11";
#endif
    return domain == RuntimeDomain::Cudart12 ? "cudart12" : "cudart13";
}}
int runtime_domain_id(RuntimeDomain domain) noexcept
{{
#if {flag}
    if (domain == RuntimeDomain::Cudart11) return 11;
#endif
    return domain == RuntimeDomain::Cudart12 ? 12 : 13;
}}

const char* runtime_soname(RuntimeDomain domain) noexcept''')
s=replace(s,'const RuntimeLibrary& runtime_library(RuntimeDomain domain) noexcept\n{',f'''const RuntimeLibrary& runtime_library(RuntimeDomain domain) noexcept
{{
#if {flag}
    if (domain == RuntimeDomain::Cudart11) {{
        static const RuntimeLibrary cudart11{{
            dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL)}};
        return cudart11;
    }}
#endif''')
s=replace(s,'(domain != 12 && domain != 13) ||',f'''(domain != 12 && domain != 13
#if {flag}
        && domain != 11
#endif
        ) ||
#if {flag}
        (domain == 11 && api_kind != 1) ||
#endif''')
# Diagnostics and the existing original-completion receipt must carry the true domain.
# Keep the M1-specific domain check and the private CUDA13 dispatch unchanged.
start=s.index('cudaError_t runtime_launch(RuntimeDomain domain,')
pre=s[:start];post=s[start:]
post=post.replace('domain == RuntimeDomain::Cudart12 ? "cudart12" : "cudart13"','runtime_domain_name(domain)')
post=post.replace('domain == RuntimeDomain::Cudart12 ? 12 : 13','runtime_domain_id(domain)')
post=replace(post,'if (domain != RuntimeDomain::Cudart12)\n',f'''if (domain != RuntimeDomain::Cudart12
#if {flag}
            && domain != RuntimeDomain::Cudart11
#endif
            )
''')
s=pre+post
anchor='HBFSIM_RUNTIME_DOMAIN_ENTRY(cudaLaunchKernel, hbfsim_cudaLaunchKernel_12,'
new=f'''#if {flag}
// CUDA11 supports only the reviewed public host-stub launch route here.
extern "C" cudaError_t hbfsim_cudaLaunchKernel_11(
    const void* function, dim3 grid, dim3 block, void** arguments,
    std::size_t shared_memory, cudaStream_t stream)
{{
    if (!strict_instrumentation_requested()) return cudaErrorNotSupported;
    return runtime_launch(RuntimeDomain::Cudart11, "cudaLaunchKernel",
                          function, grid, block, arguments, shared_memory, stream);
}}
extern "C" cudaError_t hbfsim_cudaGetFuncBySymbol_11(
    cudaFunction_t* function, const void* symbol)
{{
    auto original = runtime_get_function_symbol(RuntimeDomain::Cudart11);
    if (original == nullptr) return cudaErrorInitializationError;
    return original(function, symbol);
}}
// Existing but unreviewed CUDA11 entry points reject, without calling cudart.
#define HBFSIM_REJECT_CUDART11(name) \\
extern "C" cudaError_t hbfsim_##name##_11( \\
    const void*, dim3, dim3, void**, std::size_t, cudaStream_t) \\
{{ return cudaErrorNotSupported; }} \\
__asm__(".symver hbfsim_" #name "_11," #name "@libcudart.so.11.0");
HBFSIM_REJECT_CUDART11(cudaLaunchKernel_ptsz)
HBFSIM_REJECT_CUDART11(cudaLaunchCooperativeKernel)
HBFSIM_REJECT_CUDART11(cudaLaunchCooperativeKernel_ptsz)
#undef HBFSIM_REJECT_CUDART11
__asm__(".symver hbfsim_cudaLaunchKernel_11,cudaLaunchKernel@libcudart.so.11.0");
__asm__(".symver hbfsim_cudaGetFuncBySymbol_11,cudaGetFuncBySymbol@libcudart.so.11.0");
#endif

'''+anchor
s=replace(s,anchor,new);save(H,'src/cuda_runtime/launch_gate.cpp',s)

s=read(A,'CMakeLists.txt')
s=replace(s,'include(cmake/CompilerWarnings.cmake)',f'''option({flag} "Isolated experimental CUDA11 launch identity adapter" OFF)
if({flag})
  add_compile_definitions({flag}=1)
endif()
include(cmake/CompilerWarnings.cmake)''');save(A,'CMakeLists.txt',s)
s=read(A,'attach/nv_attach_impl/nv_attach_impl.hpp')
s=replace(s,'\tCudart12 = 12,',f'''#if {flag}
\tCudart11 = 11,
#endif
\tCudart12 = 12,''')
s=replace(s,'\tstd::atomic<void *> orig_public_launch_kernel_12{ nullptr };',f'''#if {flag}
\tstd::atomic<void *> orig_public_launch_kernel_11{{ nullptr }};
#endif
\tstd::atomic<void *> orig_public_launch_kernel_12{{ nullptr }};''')
save(A,'attach/nv_attach_impl/nv_attach_impl.hpp',s)
s=read(A,'attach/nv_attach_impl/nv_attach_impl.cpp')
s=replace(s,'cudaError_t cuda_runtime_function__cudaLaunchKernel_12(',f'''#if {flag}
cudaError_t cuda_runtime_function__cudaLaunchKernel_11(
    const void*, dim3, dim3, void**, size_t, cudaStream_t);
#endif
cudaError_t cuda_runtime_function__cudaLaunchKernel_12(''')
s=replace(s,'\tcollect_runtime_domain("libcudart.so.12",',f'''#if {flag}
\tcollect_runtime_domain("libcudart.so.11.0", "libcudart.so.11.0",
                              RuntimeRegistrationDomain::Cudart11);
#endif
\tcollect_runtime_domain("libcudart.so.12",''')
s=replace(s,'\tvoid *public_launch_12 =',f'''#if {flag}
    void *public_launch_11 =
        gate_versioned_symbol("cudaLaunchKernel", "libcudart.so.11.0");
#endif
\tvoid *public_launch_12 =''')
s=replace(s,'\tif (global_public_launch != public_launch_12) {',f'''#if {flag}
    if (public_launch_11 != nullptr && global_public_launch == public_launch_11) {{
        SPDLOG_INFO("Skipping unversioned launch alias of exact CUDA11 entry");
    }} else
#endif
\tif (global_public_launch != public_launch_12) {{''')
s=replace(s,'\treplace_hook_address_once(\n\t\t"cudaLaunchKernel@libcudart.so.12",',f'''#if {flag}
    replace_hook_address_once(
        "cudaLaunchKernel@libcudart.so.11.0", public_launch_11,
        (gpointer)&cuda_runtime_function__cudaLaunchKernel_11,
        hook_state.orig_public_launch_kernel_11);
#endif
\treplace_hook_address_once(
\t\t"cudaLaunchKernel@libcudart.so.12",''')
save(A,'attach/nv_attach_impl/nv_attach_impl.cpp',s)
s=read(A,'attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp')
s=replace(s,'\tcase RuntimeRegistrationDomain::Cudart12:',f'''#if {flag}
\tcase RuntimeRegistrationDomain::Cudart11: return "cudart11";
#endif
\tcase RuntimeRegistrationDomain::Cudart12:''')
# Preserve the CUDA12 functions exactly and add bounded CUDA11 counterparts.
start=s.index('static void try_native_auto_bind_public12(')
end=s.index('\nstatic nv_attach_hook_state *current_hook_state()',start)
helper=s[start:end].replace('public12','public11').replace('libcudart.so.12','libcudart.so.11.0').replace('\n\t\t12,','\n\t\t11,')
s=s[:end]+f'\n#if {flag}\n'+helper+'\n#endif\n'+s[end:]
start=s.index('static cudaError_t cuda_public_launch_kernel_12_common(')
end=s.index('\n#define DEFINE_PRIVATE_RUNTIME_WRAPPER',start)
wrapper=s[start:end].replace('_12_common','_11_common').replace('public12','public11').replace('"cudart12"','"cudart11"').replace('func, 12, 1','func, 11, 1').replace('  12, 1, result','  11, 1, result').replace('__cudaLaunchKernel_12','__cudaLaunchKernel_11').replace('orig_public_launch_kernel_12','orig_public_launch_kernel_11')
wrapper=replace(wrapper,'''\tif (!strict_launch_bridge_enabled())
\t\treturn cuda_launch_kernel_common(impl, original_fn_ptr, func,
\t\t\tgrid_dim, block_dim, args, shared_mem, stream, true);''','''\tif (!strict_launch_bridge_enabled())
        return cudaErrorNotSupported;''')
assert '12, 1' not in wrapper and 'cudart12' not in wrapper
s=s[:end]+f'\n#if {flag}\n'+wrapper+'\n#endif\n'+s[end:]
save(A,'attach/nv_attach_impl/nv_attach_impl_frida_setup.cpp',s)
s=read(A,'runtime/agent/agent.cpp')
s=replace(s,'static std::atomic<cuda_register_fatbinary_fn> original_register_fatbinary_12{};',f'''#if {flag}
static std::atomic<cuda_register_fatbinary_fn> original_register_fatbinary_11{{}};
#endif
static std::atomic<cuda_register_fatbinary_fn> original_register_fatbinary_12{{}};''')
s=replace(s,'extern "C" void **bpftime___cudaRegisterFatBinary_12(void *fatbin)',f'''#if {flag}
extern "C" void **bpftime___cudaRegisterFatBinary_11(void *fatbin)
{{
    return register_fatbinary_versioned(fatbin, "libcudart.so.11.0",
                                       original_register_fatbinary_11);
}}
__asm__(".symver bpftime___cudaRegisterFatBinary_11,__cudaRegisterFatBinary@libcudart.so.11.0");
#endif
extern "C" void **bpftime___cudaRegisterFatBinary_12(void *fatbin)''')
save(A,'runtime/agent/agent.cpp',s)
s=read(A,'runtime/agent/CMakeLists.txt')
s=replace(s,'set_target_properties(bpftime-agent PROPERTIES CXX_STANDARD 20 LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/agent.version)',f'''set(BPFTIME_AGENT_VERSION_MAP "${{CMAKE_CURRENT_SOURCE_DIR}}/agent.version")
if({flag})
  set(BPFTIME_AGENT_VERSION_MAP "${{CMAKE_CURRENT_SOURCE_DIR}}/agent-cuda11.version")
endif()
set_target_properties(bpftime-agent PROPERTIES CXX_STANDARD 20 LINK_DEPENDS ${{BPFTIME_AGENT_VERSION_MAP}})''')
s=replace(s,'-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/agent.version','-Wl,--version-script=${BPFTIME_AGENT_VERSION_MAP}');save(A,'runtime/agent/CMakeLists.txt',s)
s=read(A,'runtime/agent/agent.version')
s=replace(s,'libcudart.so.12 {\n};','libcudart.so.11.0 {\n};\nlibcudart.so.12 {\n} libcudart.so.11.0;')
before[(A,'runtime/agent/agent-cuda11.version')]='';save(A,'runtime/agent/agent-cuda11.version',s)
a.patch_output.mkdir(parents=True,exist_ok=True)
for root,name in [(H,'gate-cuda11.patch'),(A,'agent-cuda11.patch')]:
    lines=[]
    for (r,relative),old in before.items():
        if r!=root:continue
        new=(r/relative).read_text()
        lines.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+relative if old else '/dev/null',tofile='b/'+relative))
    target=a.patch_output/name
    assert not target.exists()
    target.write_text(''.join(lines))
print(json.dumps({'status':'APPLIED_TO_ISOLATED_COPIES_ONLY','feature':flag,'default':'OFF','patches':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.patch_output.glob('*.patch')}}))
