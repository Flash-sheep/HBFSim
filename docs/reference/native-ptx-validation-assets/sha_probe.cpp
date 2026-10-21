#include <openssl/sha.h>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdio>
#include <cstring>
int main(int argc,char**argv){if(argc!=2)return 2;std::ifstream f(argv[1],std::ios::binary);std::string b{std::istreambuf_iterator<char>(f),{}};unsigned char out[32];if(!SHA256((const unsigned char*)b.data(),b.size(),out))return 3;auto fallback=SHA256((const unsigned char*)b.data(),b.size(),nullptr);if(!fallback||memcmp(out,fallback,32))return 4;for(auto c:out)printf("%02x",c);puts("");}
