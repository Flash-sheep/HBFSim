#!/usr/bin/env python3
"""Acquire the CUDA11 provider before its identity hooks, in the isolated port."""
import argparse,difflib,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--patch-output',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];source=a.source.resolve();assert source!=ROOT and ROOT not in source.parents
identity=json.loads((Path(__file__).parent/'source_identity.json').read_text());expected={x['path']:x['after_sha256'] for x in identity['files']['agent']}
changes={}
def read(name):
 path=source/name;assert source in path.resolve().parents
 raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected[name],name
 text=raw.decode();changes[name]=[text,None];return text
def save(name,text):
 changes[name][1]=text;(source/name).write_text(text)
def once(text,before,after):assert text.count(before)==1;return text.replace(before,after)
s=read('attach/nv_attach_impl/nv_attach_impl.cpp')
s=once(s,'} // namespace bpftime::attach\n\nstatic std::vector', '''#if HBFSIM_EXPERIMENTAL_CUDART11
// The CUDA12/13 provider is a link dependency. Acquire the optional CUDA11
// provider locally before collecting its versioned registration hooks too.
void *nv_attach_cudart11_provider()
{
    static void *provider = []() -> void * {
        void *handle = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr ||
            dlvsym(handle, "__cudaRegisterFatBinary", "libcudart.so.11.0") == nullptr)
            return nullptr;
        return handle;
    }();
    return provider;
}
#endif
} // namespace bpftime::attach

static std::vector''')
s=once(s,'\tthis->module_pool = std::make_shared<', '''#if HBFSIM_EXPERIMENTAL_CUDART11
    if (bpftime::attach::nv_attach_cudart11_provider() == nullptr)
        throw std::runtime_error("Exact CUDA11 runtime provider unavailable");
#endif
\tthis->module_pool = std::make_shared<''')
save('attach/nv_attach_impl/nv_attach_impl.cpp',s)
s=read('runtime/agent/agent.cpp')
s=once(s,'using cuda_register_fatbinary_fn = void **(*)(void *);', '''#if HBFSIM_EXPERIMENTAL_CUDART11
namespace bpftime::attach { void *nv_attach_cudart11_provider(); }
#endif
using cuda_register_fatbinary_fn = void **(*)(void *);''')
s=once(s,'''    return register_fatbinary_versioned(fatbin, "libcudart.so.11.0",
                                       original_register_fatbinary_11);''','''    try {
        auto original = original_register_fatbinary_11.load(std::memory_order_acquire);
        if (original == nullptr) {
            void *provider = bpftime::attach::nv_attach_cudart11_provider();
            if (provider == nullptr)
                throw std::runtime_error("Exact CUDA11 runtime provider unavailable");
            dlerror();
            original = reinterpret_cast<cuda_register_fatbinary_fn>(
                dlvsym(provider, "__cudaRegisterFatBinary", "libcudart.so.11.0"));
            const char *error = dlerror();
            if (original == nullptr || error != nullptr)
                throw std::runtime_error(error != nullptr ? error :
                                         "Exact CUDA11 fatbinary symbol unavailable");
            original_register_fatbinary_11.store(original, std::memory_order_release);
        }
        bpftime::vm::compat::llvm::register_llvm_vm_factory();
        return original(fatbin);
    } catch (const std::exception &error) {
        fprintf(stderr, "bpftime-agent: CUDA11 fatbinary registration failed: %s\\n", error.what());
    } catch (...) {
        fprintf(stderr, "bpftime-agent: CUDA11 fatbinary registration failed\\n");
    }
    return nullptr;''')
save('runtime/agent/agent.cpp',s)
patch=[];files=[]
for name,(before,after) in changes.items():
 patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
 files.append({'path':name,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest()})
with a.patch_output.open('x') as f:f.write(''.join(patch))
with a.patch_output.with_suffix('.identity.json').open('x') as f:json.dump({'scope':'default-off CUDA11 provider acquisition; no CUDA12/13 lookup or lifecycle implementation changes','files':files},f,indent=2);f.write('\n')
print(json.dumps({'status':'APPLIED_TO_ISOLATED_COPY','files':files}))
