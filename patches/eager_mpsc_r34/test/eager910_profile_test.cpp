#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#if __has_include("../op-plugin/op_plugin/utils/op_api_eager_sdk_profile_r34.h")
#include "../op-plugin/op_plugin/utils/op_api_eager_sdk_profile_r34.h"
#else
namespace hdb_opapi {
struct Eager910MappedBinary { bool mapped; uintmax_t bytes; std::string sha; };
template<class Resolve, class Inspect> bool VerifyEager910Identity(Resolve, Inspect) { return false; }
inline bool Eager910OwnedProfileAllowed(bool legacy, bool, bool) { return legacy; }
struct Eager910EmbeddingShape {
    int64_t rows, dim, indices, offsets, padding;
    bool contiguous_nd_zero_offset, outputs;
    bool Admitted() const noexcept { return false; }
};
}
#endif

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

bool Verify(const char* altered, int corruption) {
    return hdb_opapi::VerifyEager910Identity(
        [&](const char* name) -> void* {
            return corruption == 4 && std::strcmp(name, altered) == 0 ? nullptr : reinterpret_cast<void*>(uintptr_t(1));
        },
        [&](const char* name, void*) {
            const bool nn = std::strcmp(name, "aclnnEmbeddingBag") == 0 ||
                std::strcmp(name, "aclnnEmbeddingBagGetWorkspaceSize") == 0;
            hdb_opapi::Eager910MappedBinary result{true, nn ? 17562248U : 4328144U,
                nn ? "6644fd0f5f1e7a5cdb7e549acc1057703bc81eb269bb95890b878a7d7f7aa131" :
                     "308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b"};
            if (std::strcmp(name, altered) == 0) {
                if (corruption == 1) result.sha[0] = '0';
                if (corruption == 2) ++result.bytes;
                if (corruption == 3) result.mapped = false;
                if (corruption == 5) { result.bytes = 17562248U; result.sha =
                    "6644fd0f5f1e7a5cdb7e549acc1057703bc81eb269bb95890b878a7d7f7aa131"; }
            }
            return result;
        });
}

int main() {
    try {
        for (const char* symbol : {"aclnnEmbeddingBagGetWorkspaceSize", "aclnnEmbeddingBag", "aclDestroyAclOpExecutor",
            "InitHugeMemThreadLocal", "ReleaseHugeMem", "PTAGetExecCache", "PTAFindExecCache", "AddTensorAddrToCachedList"}) {
            for (int corruption = 1; corruption != 5; ++corruption) REQUIRE(!Verify(symbol, corruption));
        }
        REQUIRE(!Verify("PTAGetExecCache", 5)); // A reviewed NN mapping is not a reviewed base-symbol provider.
        REQUIRE(!hdb_opapi::Eager910OwnedProfileAllowed(false, true, false)); // R22 cannot opt 910 in.
        REQUIRE(hdb_opapi::Eager910OwnedProfileAllowed(true, false, false)); // Existing 920 stays admitted.
        REQUIRE(Verify("", 0));
        REQUIRE(hdb_opapi::Eager910OwnedProfileAllowed(false, true, true));
        for (int64_t count : {7,55,117,128,235,283,388,408,566,637,648,1238,1310,1686,1750,1800,
            2164,3701,4892,5536,9832,13085}) {
            const hdb_opapi::Eager910EmbeddingShape ragged{1024,16,count,128,-1,true,true};
            REQUIRE(ragged.Admitted());
        }
        for (int64_t length : {1, 2, 8, 64, 128, 256}) {
            hdb_opapi::Eager910EmbeddingShape shape{1024, 16, 128 * length, 128, -1, true, true};
            REQUIRE(shape.Admitted());
            auto wrong = shape; wrong.rows = 1025; REQUIRE(!wrong.Admitted());
            wrong = shape; wrong.dim = 32; REQUIRE(!wrong.Admitted());
            wrong = shape; wrong.offsets = 129; REQUIRE(!wrong.Admitted());
            wrong = shape; wrong.indices = 128 * 3; REQUIRE(!wrong.Admitted());
            wrong = shape; wrong.padding = -2; REQUIRE(!wrong.Admitted());
            wrong = shape; wrong.contiguous_nd_zero_offset = false; REQUIRE(!wrong.Admitted());
            wrong = shape; wrong.outputs = false; REQUIRE(!wrong.Admitted());
        }
        std::cout << "PASS exact 910 identity, R22 isolation, fixed shapes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
}
