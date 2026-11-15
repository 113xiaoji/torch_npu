#pragma once
#include <cstdint>
#include <string>

namespace hdb_opapi {
// Eager-only CANN 9.1.0 NN-provider lifecycle qualification. Evidence:
// sdk-qualification-v1-complete-003.json, probe source SHA256
// d28b760602bc03da013ee7a1af082c5c14349845e746971ef08ee9280163e80d.
// 192 cases: 144 numerical executions and 48 cancellations; four overlapping
// query workers; fresh-address/worker-local v1-cache miss/hit/cancel/hit.
// This does not broaden R22 or certify arbitrary operators/shapes/cache APIs.
inline constexpr bool kEager910SdkLifecycleValidated = true;
// Separate frozen MetaSyn26 qualification: sdk-probe-v1-ragged-meta26-004.json,
// 25 nonempty tables, 800 cases/200 four-worker waves; original empty table 1
// remains on the original route. Probe source SHA256:
// 6b37b8502bb72af1b400f70893130b903f5108eacf0cf8b2126078e5b1833e5a.
inline constexpr bool kEager910FixedRaggedInputsValidated = true;
inline constexpr const char* kEagerOpApi910Sha =
    "6644fd0f5f1e7a5cdb7e549acc1057703bc81eb269bb95890b878a7d7f7aa131";
inline constexpr const char* kEagerOpBase910Sha =
    "308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b";
struct Eager910MappedBinary { bool mapped; uintmax_t bytes; std::string sha; };

template<class Resolve, class Inspect>
bool VerifyEager910Identity(Resolve&& resolve, Inspect&& inspect) {
    auto verify = [&](const char* symbol, const char* sha, uintmax_t bytes) {
        void* function = resolve(symbol);
        if (!function) return false;
        const auto identity = inspect(symbol, function);
        return identity.mapped && identity.bytes == bytes && identity.sha == sha;
    };
    for (const char* symbol : {"aclnnEmbeddingBagGetWorkspaceSize", "aclnnEmbeddingBag"})
        if (!verify(symbol, kEagerOpApi910Sha, 17562248)) return false;
    for (const char* symbol : {"aclCreateTensor", "aclDestroyTensor", "aclCreateScalar", "aclDestroyScalar",
        "aclCreateIntArray", "aclDestroyIntArray", "aclCreateBoolArray", "aclDestroyBoolArray",
        "aclCreateTensorList", "aclDestroyTensorList", "aclCreateScalarList", "aclDestroyScalarList",
        "aclDestroyAclOpExecutor", "InitHugeMemThreadLocal", "UnInitHugeMemThreadLocal", "ReleaseHugeMem",
        "InitPTACacheThreadLocal", "SetPTAHashKey", "SetPTACacheHashKey", "PTAGetExecCache",
        "PTAFindExecCache", "CanUsePTACache", "UnInitPTACacheThreadLocal", "AddTensorAddrToCachedList"})
        if (!verify(symbol, kEagerOpBase910Sha, 4328144)) return false;
    return true;
}

inline bool Eager910OwnedProfileAllowed(bool legacy920, bool eager910, bool native_job_active) noexcept {
    return legacy920 || (kEager910SdkLifecycleValidated && eager910 && native_job_active);
}

inline bool Eager910ReviewedIndicesCount(int64_t count) noexcept {
    switch (count) {
        case 128: case 256: case 1024: case 8192: case 16384: case 32768:
            return true; // Six independently qualified uniform-probe shapes.
        case 7: case 55: case 117: case 235: case 283: case 388: case 408:
        case 566: case 637: case 648: case 1238: case 1310: case 1686:
        case 1750: case 1800: case 2164: case 3701: case 4892: case 5536:
        case 9832: case 13085:
            return kEager910FixedRaggedInputsValidated;
        default:
            return false;
    }
}

struct Eager910EmbeddingShape {
    int64_t rows, dim, indices, offsets, padding;
    bool contiguous_nd_zero_offset, outputs;
    bool Admitted() const noexcept {
        if (rows != 1024 || dim != 16 || offsets != 128 || padding != -1 ||
            !contiguous_nd_zero_offset || !outputs || indices <= 0 || indices > 32768) return false;
        return Eager910ReviewedIndicesCount(indices);
    }
};
} // namespace hdb_opapi
