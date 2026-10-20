// Real agent constructor before CUDA11 is explicitly opened by the application.
#include "nv_attach_impl.hpp"
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using namespace bpftime::attach;
static void stub11() {} static void stub12() {}
static void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main() {
 try {
  setenv("BPFTIME_CUDA_DISABLE_CUOBJDUMP","1",1);
  setenv("BPFTIME_PTXPASS_LIBRARIES","/nonexistent-cpu-fixture-pass",1);
  // CUDA12 is already a link dependency. CUDA11 is not opened by this fixture yet.
  nv_attach_impl impl;
  void* eleven=dlopen("libcudart.so.11.0",RTLD_NOW|RTLD_LOCAL);check(eleven,"fake11 load");
  void* twelve=dlopen("libcudart.so.12",RTLD_NOW|RTLD_NOLOAD|RTLD_LOCAL);check(twelve,"fake12 present");
  using Reg=void**(*)(void*);using Func=void(*)(void**,const void*,const void*,const char*);using Unreg=void(*)(void**);
  void* libs[]={eleven,twelve};const void* stubs[]={(void*)&stub11,(void*)&stub12};void** handles[2]{};
  for(int i=0;i<2;++i){
   const char* version=i==0?"libcudart.so.11.0":"libcudart.so.12";
   auto reg=(Reg)dlvsym(libs[i],"__cudaRegisterFatBinary",version);
   auto fn=(Func)dlvsym(libs[i],"__cudaRegisterFunction",version);check(reg&&fn,"exact fake registration symbols");
   handles[i]=reg((void*)0xF00D);fn(handles[i],stubs[i],nullptr,"cold_kernel");
  }
  check(impl.registered_symbol_identity.contains({12,(void*)&stub12}),"CUDA12 cold registration regression");
#if HBFSIM_EXPERIMENTAL_CUDART11
  check(impl.registered_symbol_identity.contains({11,(void*)&stub11}),"CUDA11 registration missing after late application provider load");
  constexpr bool enabled=true;
#else
  check(!impl.registered_symbol_identity.contains({11,(void*)&stub11}),"default OFF unexpectedly registers CUDA11");
  constexpr bool enabled=false;
#endif
  for(int i=0;i<2;++i){const char* v=i==0?"libcudart.so.11.0":"libcudart.so.12";auto unreg=(Unreg)dlvsym(libs[i],"__cudaUnregisterFatBinary",v);unreg(handles[i]);}
  check(impl.registered_symbol_identity.empty()&&impl.fatbin_handle_to_record.empty(),"cold fixture identity cleanup");
  std::cout<<"{\"status\":\"PASS\",\"cold_provider\":true,\"CUDA11_enabled\":"<<(enabled?"true":"false")<<",\"GPU_tasks_started\":0}"<<std::endl;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
