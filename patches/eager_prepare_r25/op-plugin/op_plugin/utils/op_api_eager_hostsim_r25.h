#pragma once
#include <cstring>
#include <initializer_list>

namespace hdb_opapi {
// The real SDK verifier remains separate and unchanged. The caller supplies
// mapped-symbol identity attestation; exported names alone never pass this gate.
template<class Resolver, class VerifyMappedSymbol>
bool VerifyHostSimIdentity(const char* opted_in, Resolver resolve, VerifyMappedSymbol verify) {
    if (!opted_in || std::strcmp(opted_in,"1")!=0) return false;
    for (const char* name : {"R25HostSimAbi", "R25HostSimBackendKind", "CanUsePTACache",
        "aclnnEmbeddingBagGetWorkspaceSize", "aclnnEmbeddingBag",
        "aclnnAddGetWorkspaceSize", "aclnnAdd", "aclnnReluGetWorkspaceSize", "aclnnRelu",
        "aclnnMmGetWorkspaceSize", "aclnnMm", "aclCreateTensor", "aclDestroyTensor",
        "aclCreateScalar", "aclDestroyScalar", "aclCreateIntArray", "aclDestroyIntArray",
        "aclCreateBoolArray", "aclDestroyBoolArray", "aclCreateTensorList", "aclDestroyTensorList",
        "aclCreateScalarList", "aclDestroyScalarList", "aclDestroyAclOpExecutor",
        "InitHugeMemThreadLocal", "UnInitHugeMemThreadLocal", "ReleaseHugeMem"}) {
        auto* symbol=resolve(name);
        if (!symbol || !verify(name,symbol)) return false;
    }
    const auto abi=reinterpret_cast<int (*)()>(resolve("R25HostSimAbi"));
    const auto kind=reinterpret_cast<const char* (*)()>(resolve("R25HostSimBackendKind"));
    const auto cache=reinterpret_cast<bool (*)(const char*)>(resolve("CanUsePTACache"));
    const auto* provider_kind=kind();
    return abi()==1 && provider_kind && std::strcmp(provider_kind,"HOST_SIM_R25")==0 &&
        !cache("aclnnEmbeddingBag") && !cache("aclnnAdd") && !cache("aclnnRelu") && !cache("aclnnMm");
}
} // namespace hdb_opapi
