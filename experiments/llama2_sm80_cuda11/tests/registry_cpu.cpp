// Fixed CPU integration fixture: real nv_attach_impl/Gum with fake CUDA DSOs.
#include "nv_attach_impl.hpp"
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
using namespace bpftime::attach;
static void check(bool value,const char* text) { if(!value) throw std::runtime_error(text); }
static void stub11() {} static void stub12() {} static void stub13() {} static void unknown() {}
int main(int argc,char** argv) {
 try {
  check(argc==2,"manifest path required");
  setenv("BPFTIME_CUDA_DISABLE_CUOBJDUMP","1",1);
  setenv("BPFTIME_PTXPASS_LIBRARIES","/nonexistent-cpu-fixture-pass",1);
  setenv("HBFSIM_NATIVE_BINDING_MANIFEST_PATH",argv[1],1);
  void* handles[3]{};
  const char* versions[]={"libcudart.so.11.0","libcudart.so.12","libcudart.so.13"};
  using Reg=void**(*)(void*);using Unreg=void(*)(void**);
  using Func=void(*)(void**,const void*,const void*,const char*);
  Reg reg[3]{}; Unreg unreg[3]{};Func fn[3]{};
  for(int i=0;i<3;++i) {
   handles[i]=dlopen(versions[i],RTLD_NOW|RTLD_LOCAL);check(handles[i],dlerror());
   reg[i]=(Reg)dlvsym(handles[i],"__cudaRegisterFatBinary",versions[i]);
   unreg[i]=(Unreg)dlvsym(handles[i],"__cudaUnregisterFatBinary",versions[i]);
   fn[i]=(Func)dlvsym(handles[i],"__cudaRegisterFunction",versions[i]);
   check(reg[i]&&unreg[i]&&fn[i],"versioned fake registration exports");
  }
  // The actual constructor installs the same production registration listeners.
  const char* gate_path=std::getenv("CPU_GATE");
  check(gate_path,"gate path required");
  void* gate=dlopen(gate_path,RTLD_NOW|RTLD_GLOBAL);check(gate,dlerror());
  setenv("HBFSIM_BUILD_DIR",std::filesystem::path(gate_path).parent_path().c_str(),1);
  nv_attach_impl impl;
  using Launch=int(*)(const void*,dim3,dim3,void**,size_t,cudaStream_t);
  auto launch=(Launch)dlvsym(gate,"cudaLaunchKernel","libcudart.so.11.0");check(launch,"CUDA11 gate entry missing");
  setenv("HBFSIM_INSTRUMENTATION_POLICY","strict",1);
  check(launch((void*)&stub11,dim3(1),dim3(1),nullptr,0,nullptr)==cudaErrorNotSupported,
        "unready CUDA11 agent must reject before native launch");
  unsetenv("HBFSIM_INSTRUMENTATION_POLICY");
  check(launch((void*)&stub11,dim3(1),dim3(1),nullptr,0,nullptr)==cudaErrorNotSupported,
        "non-strict CUDA11 agent must reject");
  const void* stubs[]={(const void*)&stub11,(const void*)&stub12,(const void*)&stub13};
  void** returned[3]{};
  for(int i=0;i<3;++i) {
   returned[i]=reg[i]((void*)0xF00D);
   fn[i](returned[i],stubs[i],nullptr,"fixture_kernel");
  }
  check(returned[0]==returned[1]&&returned[1]==returned[2],"same numeric handle fixture");
  check(impl.fatbin_handle_to_record.size()==3,"cross-domain handles must remain distinct");
  check(impl.registered_symbol_identity.size()==3,"all three domains registered");
  auto write_manifest=[&](int domain,const char* kernel,const char* sha) {
   std::ofstream out(argv[1]);out<<"{\"schema_version\":1,\"status\":\"READY\",\"bindings\":[{\"runtime_domain\":"<<domain<<",\"kernel\":\""<<kernel<<"\",\"container_sha256\":\""<<sha<<"\"}]}";
  };
  auto orig=(CUfunction)0x123400;
  for(int i=0;i<3;++i) {
   int domain=11+i;write_manifest(domain,"fixture_kernel","WRONG_CONTAINER");
   check(impl.native_binding_candidate(domain,(void*)stubs[i])==1,"exact registered candidate");
   check(impl.auto_bind_registered_function(domain,(void*)stubs[i],orig)==5,"wrong container must reject");
  }
  write_manifest(12,"fixture_kernel","WRONG_CONTAINER");
  check(impl.native_binding_candidate(11,(void*)&stub11)==0,"wrong manifest domain is unselected");
  check(impl.auto_bind_registered_function(11,(void*)&stub11,orig)==5,"wrong manifest domain must reject");
  check(impl.auto_bind_registered_function(14,(void*)&stub11,orig)==1,"wrong registered domain must reject");
  write_manifest(11,"wrong_kernel","WRONG_CONTAINER");
  check(impl.auto_bind_registered_function(11,(void*)&stub11,orig)==5,"wrong entry must reject");
  fn[0]((void**)0xDEAD,(void*)&unknown,nullptr,"fixture_kernel");
  check(!impl.registered_symbol_identity.contains({11,(void*)&unknown}),"unknown handle must not register");
  auto first=impl.registered_symbol_identity.at({11,(void*)&stub11}).generation;
  reg[0]((void*)0xF00D); // Same handle reused: old symbol generation is invalidated.
  check(!impl.registered_symbol_identity.contains({11,(void*)&stub11}),"handle reuse invalidates old symbols");
  fn[0](returned[0],(void*)&stub11,nullptr,"fixture_kernel");
  auto second=impl.registered_symbol_identity.at({11,(void*)&stub11}).generation;
  check(second>first,"reused handle receives fresh generation");
  write_manifest(11,"fixture_kernel","WRONG_CONTAINER");
  // A synthetic stale-generation fault exercises the real candidate/bind guards.
  ++impl.fatbin_handle_to_record.at({11,returned[0]}).generation;
  check(impl.native_binding_candidate(11,(void*)&stub11)==-1,"stale candidate must reject");
  check(impl.auto_bind_registered_function(11,(void*)&stub11,orig)==2,"stale binding must reject");
  --impl.fatbin_handle_to_record.at({11,returned[0]}).generation;
  unreg[0](returned[0]);
  check(!impl.fatbin_handle_to_record.contains({11,returned[0]}),"unregister removes exact handle");
  check(!impl.registered_symbol_identity.contains({11,(void*)&stub11}),"unregister removes exact symbols");
  check(impl.registered_symbol_identity.contains({12,(void*)&stub12})&&impl.registered_symbol_identity.contains({13,(void*)&stub13}),"unregister must preserve other domains");
  check(impl.auto_bind_registered_function(11,(void*)&stub11,orig)==1,"unregistered binding must reject");
  for(int i=1;i<3;++i)unreg[i](returned[i]);
  check(impl.registered_symbol_identity.empty()&&impl.fatbin_handle_to_record.empty(),"all fake identities released");
  using Count=int(*)();
  auto public_calls=(Count)dlsym(handles[0],"fake_public_calls");
  check(public_calls&&public_calls()==0,"rejected CUDA11 entry must not call cudart");
  auto launches=(Count)dlsym(dlopen("libcuda.so.1",RTLD_NOW),"fake_driver_launches");
  check(launches&&launches()==0,"no driver launch is permitted in this CPU fixture");
  std::ifstream maps("/proc/self/maps");std::string line;
  const char* fake_dir=std::getenv("CPU_FAKE_DIR");check(fake_dir,"fake directory missing");
  while(std::getline(maps,line))if(line.find("/libcuda.so")!=std::string::npos)
    check(line.find(fake_dir)!=std::string::npos,"real CUDA driver mapped in CPU fixture");
  std::cout<<"{\"status\":\"PASS\",\"domains\":[11,12,13],\"scope\":\"real registration listeners, unready launch rejection and negative native binding with fake CUDA DSOs\",\"driver_launches\":0,\"hardware_gpu_used\":false}"<<std::endl;
 } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1; }
}
