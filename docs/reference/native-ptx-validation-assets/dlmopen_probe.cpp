#define _GNU_SOURCE
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
int main(int argc,char**argv){
 if(argc!=4)return 2;
 void* h=dlmopen(LM_ID_NEWLM,argv[1],RTLD_NOW|RTLD_LOCAL);
 if(!h){std::cerr<<dlerror()<<'\n';return 3;}
 auto f=reinterpret_cast<int(*)(const char*,int,char*)>(dlsym(h,"process_input"));
 if(!f)return 4;
 std::ifstream in(argv[2]);std::string input{std::istreambuf_iterator<char>(in),{}};
 std::vector<char> out(8<<20);int rc=f(input.c_str(),out.size(),out.data());
 std::ofstream(argv[3])<<out.data()<<'\n';std::cerr<<"process_input rc="<<rc<<'\n';
 return rc;
}
