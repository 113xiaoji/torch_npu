#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#include "torch_npu/csrc/core/npu/HostRegionParameterRelease.h"

namespace at_npu {
namespace native {
std::atomic<uint64_t> ExecuteParas::g_pta_correlation_id{0};
void ExecuteParas::Release()
{
    ReleaseLegacyParameters(*this, c10_npu::host_regions::ParameterArrayOwnership::Heap);
}

void ReleaseLegacyParameters(ExecuteParas& value, c10_npu::host_regions::ParameterArrayOwnership ownership)
{
    // if useDynamicCompile, this attr will be freed in dynamic compile.
    if (value.attr != nullptr) {
        aclopDestroyAttr(value.attr);
    }
    DestroyConstParams(value.constParams);
    c10_npu::host_regions::parameter_detail::DestroyArrays(value.paras, ownership,
        [](const aclTensorDesc* desc) { aclDestroyTensorDesc(desc); },
        [](const aclDataBuffer* buffer) { NPU_CHECK_ERROR_WITHOUT_UCE(aclDestroyDataBuffer(buffer)); },
        [](const void* arrays) { std::free(const_cast<void*>(arrays)); });
    value.hostMemory.clear();
    value.customHandler = nullptr;
    return;
}

void ExecuteParas::Copy(ExecuteParas &other)
{
    strncpy(this->opType, other.opType, sizeof(ExecuteParas::opType) - 1);
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
    this->hostMemory = other.hostMemory;
    this->isJitDisable = other.isJitDisable;
    this->customHandler = other.customHandler;
    this->pta_correlation_id = other.pta_correlation_id;
}

void ExecuteParas::CopyEx(ExecuteParas& other)
{
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
}

void ExecuteParasOpApi::Release() { customHandler = nullptr; }

void ExecuteParasOpApi::Copy(ExecuteParasOpApi &other)
{
    strncpy(this->opType, other.opType, sizeof(ExecuteParasOpApi::opType) - 1);
    this->customHandler = std::move(other.customHandler);
}

void ExecuteParasOpApi::Copy(ExecuteParasOpApiV2 &other)
{
    static const auto max_len = sizeof(ExecuteParasOpApi::opType);
    auto len = other.opName->length();
    if (len + 1 < max_len) {
        other.opName->copy(this->opType, len + 1);
    } else {
        other.opName->copy(this->opType, max_len - 1);
    }
    this->customHandler = std::move(*(other.customHandler));
}

NPUStatus DestroyAclParams(ACL_PARAMS& params)
{
    c10_npu::host_regions::parameter_detail::DestroyArrays(params,
        c10_npu::host_regions::ParameterArrayOwnership::Heap,
        [](const aclTensorDesc* desc) { aclDestroyTensorDesc(desc); },
        [](const aclDataBuffer* buffer) { NPU_CHECK_ERROR_WITHOUT_UCE(aclDestroyDataBuffer(buffer)); },
        [](const void* arrays) { std::free(const_cast<void*>(arrays)); });
    return NPU_STATUS_SUCCESS;
}

void DestroyConstParams(CONST_PARAMS &params)
{
    if (params.constList != nullptr) {
        for (int i = 0; i < params.constNum; ++i) {
            if (params.constList[i] != nullptr) {
                delete[] params.constList[i];
            }
        }
    }
    params.constList = nullptr;
    params.constIdx = nullptr;
}
} // namespace native
} // namespace at_npu
