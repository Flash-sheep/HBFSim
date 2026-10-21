#include <openssl/sha.h>
#include <cstddef>

// Build-local --wrap shim: use the same OpenSSL SHA-256 algorithm without
// OpenSSL 3's implicit provider/default-context lookup across dlmopen namespaces.
// Hidden symbol; no production source, public HBF API or PTX transform changes.
extern "C" __attribute__((visibility("hidden"))) unsigned char*
__wrap_SHA256(const unsigned char* data, std::size_t length, unsigned char* output)
{
    static thread_local unsigned char fallback[SHA256_DIGEST_LENGTH];
    if (!output) output = fallback;
    SHA256_CTX context;
    if (SHA256_Init(&context) != 1 ||
        SHA256_Update(&context, data, length) != 1 ||
        SHA256_Final(output, &context) != 1) return nullptr;
    return output;
}
