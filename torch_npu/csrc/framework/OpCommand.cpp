#include "torch_npu/csrc/core/npu/HostRegionCapture.h"
#include "torch_npu/csrc/core/npu/HostRegionTensorOwner.h"
#include "torch_npu/csrc/core/npu/HostRegionParameterRelease.h"
#include "torch_npu/csrc/core/npu/HostRegionNativeBindings.h"
#include <ATen/record_function.h>
#include <string>
#include <chrono>
#include <mutex>

#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCmdHelper.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/NpuStorageOffsetGuard.h"
#include "torch_npu/csrc/framework/LazyInitAclops.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/logging/LogContext.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"
#endif

namespace {
const uint64_t kStringOffset = 16UL;
const std::string kStringDType = "string";
static std::unordered_map<at::ScalarType, std::vector<double>> floating_limits_map{
    {at::ScalarType::Double, {std::numeric_limits<double>::max(), std::numeric_limits<double>::min()}},
    {at::ScalarType::Float, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
    {at::ScalarType::BFloat16, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
    {at::ScalarType::Half, {65504, -65504}},
    {at::ScalarType::Float8_e5m2, {57345, -57345}},
    {at::ScalarType::Float8_e4m3fn, {449, -449}}};
static std::unordered_map<at::ScalarType, std::vector<long>> integral_limits_map{
    {at::ScalarType::Long, {std::numeric_limits<long>::max(), std::numeric_limits<long>::min()}},
    {at::ScalarType::Int, {std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}},
    {at::ScalarType::Byte, {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min()}},
    {at::ScalarType::Char, {std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min()}},
    {at::ScalarType::Short, {std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min()}}};
} // namespace

std::atomic<bool> g_used_aclop{false};

namespace at_npu {
namespace native {

// HDB_NATIVE_BINDING_HELPERS_BEGIN
namespace {
namespace binding = c10_npu::host_regions::native_binding_detail;
using NativeBindingSnapshot = binding::TensorMetadata<aclDataType, aclFormat, c10::IntArrayRef>;

bool UseNativeBindings() noexcept
{
    if (!c10_npu::host_regions::UseCapturedNativeBindings()) return false;
#ifndef BUILD_LIBTORCH
    if (c10_npu::impl::NPUTrace::getTrace() ||
        torch_npu::profiler::profDataReportEnable().load(std::memory_order_relaxed)) return false;
#endif
    return true;
}

void CheckNativeBindingSource(const at::Tensor& tensor)
{
    binding::Increment(binding::counters.source_checks);
    if (!c10_npu::host_regions::CapturedStorageSourceKey(tensor, true))
        c10_npu::host_regions::RejectEffect("native builder requires a qualified native Tensor binding");
}

NativeBindingSnapshot SnapshotNativeInput(const at::Tensor& tensor, const string& force_type)
{
    const auto& storage_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor);
    const bool zero_dim = storage_desc.storage_sizes_.empty();
    // The original native zero-dimensional helper ignores force_type and uses
    // one tensor element, ND placement, and no SetFormat/SetShape calls.
    const aclDataType type = zero_dim ? CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type()) :
        CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type(), force_type);
    const bool string_type = type == ACL_STRING;
    const auto elements = zero_dim ? int64_t{1} : c10::multiply_integers(storage_desc.storage_sizes_);
    NativeBindingSnapshot result{type, zero_dim ? ACL_FORMAT_ND : storage_desc.origin_format_,
        storage_desc.npu_format_, zero_dim || string_type ? c10::IntArrayRef{} : c10::IntArrayRef(storage_desc.base_sizes_),
        zero_dim || string_type ? c10::IntArrayRef{} : c10::IntArrayRef(storage_desc.storage_sizes_),
        tensor.data_ptr(), static_cast<size_t>(tensor.itemsize()) * static_cast<size_t>(elements), !zero_dim};
    binding::Increment(binding::counters.metadata_snapshots);
    return result;
}

NativeBindingSnapshot SnapshotNativeOutput(const at::Tensor& tensor, const string& force_type)
{
    const auto& storage_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor);
    NativeBindingSnapshot result{CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type(), force_type),
        storage_desc.origin_format_, storage_desc.npu_format_, tensor.sizes(), storage_desc.storage_sizes_,
        tensor.data_ptr(), static_cast<size_t>(tensor.itemsize()) *
            static_cast<size_t>(c10::multiply_integers(storage_desc.storage_sizes_)), true};
    binding::Increment(binding::counters.metadata_snapshots);
    return result;
}

struct NativeBindingAclApi {
    aclTensorDesc* CreateDescriptor(const NativeBindingSnapshot& metadata) const
    {
        return aclCreateTensorDesc(metadata.data_type, metadata.logical_dims.size(),
            metadata.logical_dims.data(), metadata.origin_format);
    }
    void SetFormat(aclTensorDesc* descriptor, const NativeBindingSnapshot& metadata) const
    {
        aclSetTensorFormat(descriptor, metadata.device_format);
    }
    void SetShape(aclTensorDesc* descriptor, const NativeBindingSnapshot& metadata) const
    {
        aclSetTensorShape(descriptor, metadata.storage_dims.size(), metadata.storage_dims.data());
    }
    void SetName(aclTensorDesc* descriptor, const string& name) const
    {
        aclSetTensorDescName(descriptor, name.c_str());
    }
    aclDataBuffer* CreateBuffer(const NativeBindingSnapshot& metadata) const
    {
        return aclCreateDataBuffer(metadata.data, metadata.buffer_bytes);
    }
    void DestroyDescriptor(aclTensorDesc* descriptor) const noexcept { aclDestroyTensorDesc(descriptor); }
    void DestroyBuffer(aclDataBuffer* buffer) const noexcept { aclDestroyDataBuffer(buffer); }
};
} // namespace
// HDB_NATIVE_BINDING_HELPERS_END

OpCommand::OpCommand()
{
    aclCmds = OpCommandImpls::GetInstance();
    aclCmds->Push(aclCmd);
    aclCmd->SetCustomHandler(nullptr);
}

OpCommand& OpCommand::Name(const string &name)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    aclCmd->SetName(name);
    return *this;
}

OpCommand& OpCommand::SetCustomHandler(PROC_FUNC func)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings() && func)
        c10_npu::host_regions::RejectEffect("custom handler violates the certified native builder contract");
    aclCmd->SetCustomHandler(func);
    return *this;
}

OpCommand& OpCommand::DynamicInputReg(DynamicInputRegFunc func, DyNumAndIndex num_and_index) { return *this; }

OpCommand& OpCommand::Expect(UnifiedResult unified_result)
{
    commonType = unified_result.common_type;
    resultTypeDefined = unified_result.result_type_defined;
    commonShape = unified_result.common_shape;
    return *this;
}

OpCommand& OpCommand::Input() { return AddNoneTensor(); }

OpCommand& OpCommand::Input(const at::Tensor &input, const string &descName,
                            const c10::optional<aclFormat> &sensitive_format, const string &realData)
{
    return AddTensorInput(Contiguous(input), c10::ScalarType::Undefined, descName, realData);
}

OpCommand& OpCommand::InputWithoutContiguous(const at::Tensor &input, const string &descName, const string &realData)
{
    if (input.storage_offset() != 0) {
        TORCH_NPU_WARN_ONCE("[Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy",
                            input.storage_offset());
    }
    return AddTensorInput(const_cast<at::Tensor &>(input));
}

OpCommand& OpCommand::Input(const c10::IntArrayRef &dimListRef, at::ScalarType toType, CompileType compileType,
                            const string &realDtype, const string &descName)
{
    return Input<int64_t>(dimListRef, dimListRef.size(), toType, compileType, realDtype, descName);
}

OpCommand& OpCommand::Input(const c10::ArrayRef<double> &dimListRef, at::IntArrayRef realShape, at::ScalarType toType,
                            CompileType compileType, const string &realDtype)
{
    return Input<double>(dimListRef, realShape, toType, compileType, realDtype);
}

OpCommand& OpCommand::Input(const c10::Scalar &input, const at::ScalarType type, CompileType compileType)
{
    const auto &scalarTensor = CreateScalarTensor(input, type);
    return AddHostTensorInput(scalarTensor, compileType);
}

OpCommand& OpCommand::Inputs(const at::TensorList &inputs)
{
    for (auto &input : inputs) {
        this->Input(input);
    }
    return *this;
}

OpCommand& OpCommand::InputScalarToNPUTensor(const c10::Scalar &input, const at::ScalarType type)
{
    return AddScalarInput(input, type);
}

OpCommand& OpCommand::Output(at::Tensor &output, const string &descName,
                             const c10::optional<aclFormat> &sensitive_format, const string &realType)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    outputTensor.emplace_back(output);
    return AddOutput(output, realType);
}

void OpCommand::Run()
{
    TORCH_CHECK(aclCmd != nullptr && aclCmds != nullptr, "native builder command already terminated", PTA_ERROR(ErrCode::VALUE));
    const bool native_builder = UseNativeBindings();
    auto failure = binding::GuardCommand(native_builder, aclCmd, aclCmds, inputTensor, outputTensor, storage);
    const bool region_capture = c10_npu::host_regions::IsCapturing();
    bool region_source_released = false;
    bool region_stack_popped = false;
    bool region_acl_transferred = false;
    try {
    if (region_capture && (sync || !c10_npu::option::OptionsManager::GetTaskQueueEnable() ||
        c10_npu::option::OptionsManager::CheckBlockingEnable())) {
        c10_npu::host_regions::RejectEffect("synchronous legacy operator");
    }
    // Check for npu graph
    if (aclCmd->CheckCustomHandlerNull()) {
        g_used_aclop = true;
        // true aclop
        at_npu::aclops::LazyInitAclops();
        auto val = c10_npu::option::GetOption("jitCompile");
        NPU_CHECK_ERROR(AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, val->c_str()));
        at_npu::aclops::LazyAclopSet::SetCompileopt();
        c10_npu::assertNotCapturingAclop(aclCmd->GetName());
    }

    aclCmd->SetEnginePriority();
    const string &op_name = aclCmd->GetName();
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable() && !sync) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParas execParams{};
        c10_npu::queue::QueueParas params(c10_npu::queue::COMPILE_AND_EXECUTE, sizeof(ExecuteParas), &execParams);
        if (region_capture) {
            bool exported = false;
            bool packet_prepared = false;
            try {
                packet_prepared = aclCmd->CanExportCapturedParams() &&
#ifndef BUILD_LIBTORCH
                    trigger == nullptr &&
                    !torch_npu::profiler::profDataReportEnable().load(std::memory_order_relaxed) &&
#endif
                    c10_npu::host_regions::PrepareCapturedPacketCommand(
                        inputTensor.data(), inputTensor.size(), outputTensor.data(), outputTensor.size(),
                        storage.data(), storage.size());
                if (native_builder && !packet_prepared)
                    c10_npu::host_regions::RejectEffect("native builder command failed complete packet qualification");
                if (packet_prepared) {
                    aclCmd->ExportCapturedParams(execParams,
                        [](void*, size_t bytes, size_t alignment) -> void* {
                            return c10_npu::host_regions::AllocateCapturedPacketParams(bytes, alignment);
                        }, nullptr);
                } else {
                    aclCmd->ExportParams(execParams);
                }
                exported = true;
                // Frozen ACL buffers contain addresses, not Tensor ownership.
                // Retain inputs, outputs and cast/contiguous temporaries until
                // the original queue owns this command and consumes it.
                if (!packet_prepared) {
                    execParams.hostMemory.reserve(execParams.hostMemory.size() +
                        inputTensor.size() + outputTensor.size() + storage.size());
                    for (const auto& value : inputTensor)
                        execParams.hostMemory.push_back(c10_npu::host_regions::RetainBufferOwner(value, !execParams.customHandler));
                    for (const auto& value : outputTensor)
                        execParams.hostMemory.push_back(c10_npu::host_regions::RetainBufferOwner(value, !execParams.customHandler));
                    for (const auto& value : storage)
                        execParams.hostMemory.push_back(c10_npu::host_regions::RetainBufferOwner(value, !execParams.customHandler));
                }
                c10_npu::enCurrentNPUStream(&params);
                region_acl_transferred = true;
            } catch (...) {
                struct PreparedReset {
                    ~PreparedReset() noexcept { c10_npu::host_regions::CancelCapturedPacketCommand(); }
                } reset;
                if (native_builder) {
                    if (exported) {
                        // The exported arena payload (or Capture) owns ACL.
                        // Detach implementation metadata before its one release
                        // attempt; the outer failure guard must not retry it.
                        aclCmd->releaseSource(false);
                        region_source_released = true;
                        ReleaseLegacyParameters(execParams,
                            c10_npu::host_regions::ParameterArrayOwnership::PacketArena);
                    }
                    // If export failed, the outer command guard cleans its
                    // source once and detaches even a partial cleanup failure.
                    throw;
                }
                // Capture clears execParams' ACL ownership after taking it.
                // A failed capture therefore releases exactly the surviving owner.
                if (exported) {
                    if (packet_prepared) {
                        // ACL cleanup may be partially complete when it throws.
                        // Detach the implementation first so the outer unwind
                        // cannot retry that non-idempotent destruction.
                        region_source_released = true;
                        aclCmd->releaseSource(false);
                        ReleaseLegacyParameters(execParams,
                            c10_npu::host_regions::ParameterArrayOwnership::PacketArena);
                    } else {
                        execParams.Release();
                        aclCmd->releaseSource(false);
                    }
                } else {
                    // Export may fail allocating/copying the owning vector after
                    // allocating its pointer array. Impl still owns ACL objects.
                    if (!packet_prepared) free(execParams.paras.input_desc);
                    if (packet_prepared) region_source_released = true;
                    aclCmd->releaseSource();
                }
                region_source_released = true;
                throw;
            }
        } else {
            aclCmd->ExportParams(execParams);
            c10_npu::enCurrentNPUStream(&params);
        }
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
        aclCmd->releaseSource(false);
        region_source_released = true;
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        aclCmd->Run(sync, sync_index, outputTensor);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                Sync();
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
        aclCmd->releaseSource();
    }
    aclCmds->Pop();
    region_stack_popped = true;
    } catch (...) {
        if (native_builder && region_acl_transferred && !region_source_released)
            aclCmd->releaseSource(false);
        if (region_capture && !native_builder) {
            if (!region_source_released) aclCmd->releaseSource(!region_acl_transferred);
            if (!region_stack_popped) aclCmds->Pop();
        }
        throw;
    }
}

void OpCommand::RunOwnedOpApi(const string &op_name,
    std::unique_ptr<c10_npu::host_regions::OwnedOpApiPayload> payload)
{
    TORCH_CHECK(c10_npu::host_regions::UseCapturedOwnedOpApi(),
        "Owned OpAPI command requires an admitted real native region");
    TORCH_CHECK(c10_npu::host_regions::CaptureOwnedOpApi(op_name.c_str(), payload, c10_npu::current_device()),
        "Owned OpAPI capture did not adopt the command");
}

void OpCommand::RunOpApi(const string &op_name, PROC_FUNC func, bool sync)
{
    if (c10_npu::host_regions::IsCapturing() &&
        (sync || !c10_npu::option::OptionsManager::GetTaskQueueEnable() ||
         c10_npu::option::OptionsManager::CheckBlockingEnable())) {
        c10_npu::host_regions::RejectEffect("synchronous OPAPI operator");
    }
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParasOpApi execParams;
        if (op_name.length() + 1 < sizeof(ExecuteParasOpApi::opType)) {
            op_name.copy(execParams.opType, op_name.length() + 1);
        } else {
            op_name.copy(execParams.opType, sizeof(ExecuteParasOpApi::opType) - 1);
        }
        execParams.customHandler = func;

        c10_npu::queue::QueueParas params(c10_npu::queue::EXECUTE_OPAPI, sizeof(ExecuteParasOpApi), &execParams);
        c10_npu::enCurrentNPUStream(&params);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        OpCommandImpl::RunOpApi(op_name, func);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
    }
    if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
}

void OpCommand::RunOpApiV2(const string &op_name, const PROC_FUNC &func, bool sync)
{
    if (c10_npu::host_regions::IsCapturing() &&
        (sync || !c10_npu::option::OptionsManager::GetTaskQueueEnable() ||
         c10_npu::option::OptionsManager::CheckBlockingEnable())) {
        c10_npu::host_regions::RejectEffect("synchronous OPAPI operator");
    }
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParasOpApiV2 execParams;
        execParams.opName = const_cast<std::string*>(&op_name);
        execParams.customHandler = const_cast<PROC_FUNC*>(&func);

        c10_npu::queue::QueueParas params(c10_npu::queue::EXECUTE_OPAPI_V2, sizeof(ExecuteParasOpApiV2), &execParams);

        auto start = std::chrono::steady_clock::now();

        c10_npu::enCurrentNPUStream(&params);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        TORCH_NPU_DISPATCH_TIME_LOGI("enQueue duration is (ns): %d", duration);

#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        OpCommandImpl::RunOpApi(op_name, func);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
    }
    if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
}

void OpCommand::RunOpApiV3(const string &op_name, const PROC_FUNC &func, bool sync, c10_npu::NPUStream *task_stream)
{
    if (c10_npu::host_regions::IsCapturing() &&
        (sync || !c10_npu::option::OptionsManager::GetTaskQueueEnable() ||
         c10_npu::option::OptionsManager::CheckBlockingEnable())) {
        c10_npu::host_regions::RejectEffect("synchronous OPAPI operator");
    }
#ifndef BUILD_LIBTORCH
    const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
#endif
    auto stream = c10_npu::getCurrentNPUStream();
    if (!stream.isSyncLaunchStream() && c10_npu::option::OptionsManager::GetTaskQueueEnable()) {
        RECORD_FUNCTION(op_name, std::vector<c10::IValue>({}));
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
        ExecuteParasOpApiV2 execParams;
        execParams.opName = const_cast<std::string*>(&op_name);
        execParams.customHandler = const_cast<PROC_FUNC*>(&func);

        c10_npu::queue::QueueParas params(c10_npu::queue::EXECUTE_OPAPI_V2, sizeof(ExecuteParasOpApiV2), &execParams);
        c10_npu::enCurrentNPUStream(&params, -1, task_stream);
#ifndef BUILD_LIBTORCH
        at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
    } else {
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclStartExecution(op_name);
        }
#endif
        OpCommandImpl::RunOpApi(op_name, func);
        if (c10_npu::option::OptionsManager::CheckBlockingEnable()) {
            if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
            }
        }
#ifndef BUILD_LIBTORCH
        if (C10_UNLIKELY(trigger)) {
            trigger->traceNpuAclFinishExecution(op_name);
        }
#endif
    }
    if (sync) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
}

OpCommand& OpCommand::Sync(c10::SmallVector<int64_t, N> &index)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings() && !index.empty())
        c10_npu::host_regions::RejectEffect("synchronous operator violates the certified native builder contract");
    sync_index = index;
    if (!index.empty()) {
        sync = true;
    }
    return *this;
}

OpCommand& OpCommand::Sync()
{
    if (UseNativeBindings()) {
        auto failure = binding::GuardCommand(true, aclCmd, aclCmds, inputTensor, outputTensor, storage);
        c10_npu::host_regions::RejectEffect("synchronize violates the certified native builder contract");
    }
    if (c10_npu::host_regions::IsCapturing()) {
        aclCmd->releaseSource();
        aclCmds->Pop();
        c10_npu::host_regions::RejectEffect("OpCommand sync");
    }
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    return *this;
}

OpCommand& OpCommand::AddTensorInput(at::Tensor &tensor, at::ScalarType forceScaleType, const string &descName,
                                     const string &realData)
{
    const bool native_builder = UseNativeBindings();
    auto failure = binding::GuardCommand(native_builder, aclCmd, aclCmds, inputTensor, outputTensor, storage);
    const bool already_owned = native_builder && binding::OwnsBindingReference(inputTensor, tensor);
    if (native_builder && !already_owned) CheckNativeBindingSource(tensor);
    auto* previous_impl = native_builder ? tensor.unsafeGetTensorImpl() : nullptr;
    std::tuple<aclTensorDesc*, aclDataBuffer*> res;
    if (commonType.has_value() && commonType.value() != tensor.scalar_type()) {
        tensor = custom_ops::_npu_dtype_cast(tensor, commonType.value());
    }
    if (native_builder) {
        if (tensor.unsafeGetTensorImpl() != previous_impl) CheckNativeBindingSource(tensor);
        if (!already_owned) {
            inputTensor.emplace_back(tensor);
            binding::Increment(binding::counters.input_ledger_copies);
        } else {
            binding::Increment(binding::counters.input_duplicates_elided);
        }
        const auto metadata = SnapshotNativeInput(tensor, realData);
        auto pair = binding::PreparePair(metadata, descName, NativeBindingAclApi{});
        aclCmd->AddCapturedInput(pair);
        return *this;
    }
    // as for dim=0, the dtype of tensor can not be `uint16` because of the backend
    if (torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).storage_sizes_.empty()) {
        if (torch_npu::utils::is_npu(tensor)) {
            res = OpCmdHelper::CovertNPUTensorWithZeroDimToAclInput(tensor, descName);
        } else {
            res = OpCmdHelper::CovertTensorWithZeroDimToAclInput(tensor, forceScaleType);
        }
    } else {
        res = OpCmdHelper::CovertTensorToAclInput(tensor, descName, realData);
    }
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    if (c10_npu::host_regions::IsCapturing()) inputTensor.emplace_back(tensor);
    return *this;
}

OpCommand& OpCommand::AddHostTensorInput(
    const at::Tensor &tensor,
    CompileType compileType,
    const string& realDtype,
    const string& descName)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings())
        c10_npu::host_regions::RejectEffect("host memory violates the certified native builder contract");
    std::tuple<aclTensorDesc*, aclDataBuffer*> res =
        OpCmdHelper::CovertHostTensorToAclInput(
            tensor,
            tensor.scalar_type(),
            compileType,
            realDtype,
            descName);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res), tensor);
    return *this;
}

OpCommand& OpCommand::AddNoneTensor()
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings())
        c10_npu::host_regions::RejectEffect("undefined input violates the certified native builder contract");
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED).Get();
    AclTensorBufferMaker buffer(nullptr, 0);
    aclCmd->AddInput(aclDesc, buffer.Get());
    return *this;
}

OpCommand& OpCommand::AddScalarInput(const c10::Scalar& input, at::ScalarType type)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings())
        c10_npu::host_regions::RejectEffect("scalar conversion violates the certified native builder contract");
    at::ScalarType type_bk = type;
    if (commonType.has_value()) {
        type_bk = commonType.value();
    }
    at::Tensor aclInput = CopyHostToDevice(input, type_bk);
    auto res = OpCmdHelper::CovertScalarToAclInput(aclInput, type_bk);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    return *this;
}

OpCommand& OpCommand::AddOutput(at::Tensor &output, const string &realType)
{
    const bool native_builder = UseNativeBindings();
    auto failure = binding::GuardCommand(native_builder, aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (native_builder) CheckNativeBindingSource(output);
    auto* previous_impl = native_builder ? output.unsafeGetTensorImpl() : nullptr;
    if (resultTypeDefined == false && commonType.has_value() && commonType.value() != output.scalar_type()) {
        output = custom_ops::_npu_dtype_cast(output, commonType.value());
    }
    if (native_builder) {
        if (output.unsafeGetTensorImpl() != previous_impl) CheckNativeBindingSource(output);
        // Ordered pre-cast outputs remain untouched for the original Sync
        // contract. Only a different actual TensorImpl needs another binding.
        if (outputTensor.empty() || outputTensor.back().unsafeGetTensorImpl() != output.unsafeGetTensorImpl())
            inputTensor.emplace_back(output);
        else
            binding::Increment(binding::counters.output_duplicates_elided);
        const auto metadata = SnapshotNativeOutput(output, realType);
        auto pair = binding::PreparePair(metadata, string{}, NativeBindingAclApi{});
        aclCmd->AddCapturedOutput(pair);
        return *this;
    }
    auto res = OpCmdHelper::CovertToAclOutput(output, realType);
    aclCmd->AddOutput(std::get<0>(res), std::get<1>(res));
    if (c10_npu::host_regions::IsCapturing()) inputTensor.emplace_back(output);
    return *this;
}

// 由于format_contiguous会生成新Tensor，为了保证其在生命周期内有效，故而放到对象中存储
// 同下，CopyScalarToDevice也有同样问题
at::Tensor& OpCommand::Contiguous(const at::Tensor &input)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings()) {
        CheckNativeBindingSource(input);
        auto contiguous = NpuUtils::format_contiguous_add_copy_optimize(input);
        if (contiguous.unsafeGetTensorImpl() != input.unsafeGetTensorImpl()) CheckNativeBindingSource(contiguous);
        inputTensor.emplace_back(std::move(contiguous));
        binding::Increment(binding::counters.contiguous_ledger_moves);
        return inputTensor.back();
    }
    storage.emplace_back(std::move(NpuUtils::format_contiguous_add_copy_optimize(input)));
    return storage.back();
}

at::Tensor OpCommand::CopyHostToDevice(const c10::Scalar& scalar, at::ScalarType type)
{
    auto tensor = scalar_to_tensor(scalar).to(type);
    return CopyHostToDevice(tensor);
}

at::Tensor OpCommand::CopyHostToDevice(const at::Tensor& cpuTensor)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings())
        c10_npu::host_regions::RejectEffect("host-to-device conversion violates the certified native builder contract");
    at::Tensor cpuPinMemTensor = cpuTensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&deviceIndex));
    auto tensor = cpuPinMemTensor.to(
        c10::Device(c10::DeviceType::PrivateUse1, deviceIndex),
        cpuPinMemTensor.scalar_type(),
        true,
        true);
    storage.emplace_back(tensor);
    return storage.back();
}

at::Tensor& OpCommand::CreateHostTensor(
    void *data,
    at::IntArrayRef size,
    const c10::TensorOptions &options,
    at::ScalarType toType)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings())
        c10_npu::host_regions::RejectEffect("host tensor creation violates the certified native builder contract");
    at::ScalarType dtype = options.dtype().toScalarType();
    auto cpuTensor = at::empty(size, options);
    std::memcpy(cpuTensor.data_ptr(), data, elementSize(dtype) * cpuTensor.numel());
    if (toType != dtype) {
        cpuTensor = cpuTensor.to(toType);
    }

    storage.emplace_back(std::move(cpuTensor));
    return storage.back();
}

bool OpCommand::ScalarIsInLimits(const c10::Scalar &scalar, at::ScalarType type)
{
    bool scalar_flag = false;
    if (at::isFloatingType(type)) {
        auto value = scalar.to<double>();
        scalar_flag = value <= floating_limits_map[type][0] && value >= floating_limits_map[type][1];
    } else if (at::isIntegralType(type, false)) {
        auto value = scalar.to<long>();
        scalar_flag = value <= integral_limits_map[type][0] && value >= integral_limits_map[type][1];
    }
    return scalar_flag;
}

at::Tensor& OpCommand::CreateScalarTensor(const c10::Scalar &scalar, at::ScalarType type)
{
    auto failure = binding::GuardCommand(UseNativeBindings(), aclCmd, aclCmds, inputTensor, outputTensor, storage);
    if (UseNativeBindings())
        c10_npu::host_regions::RejectEffect("scalar tensor creation violates the certified native builder contract");
    if (commonType.has_value()) {
        type = commonType.value();
    }

    if (ScalarIsInLimits(scalar, type)) {
        storage.emplace_back(at::detail::scalar_tensor_static(scalar, type, at::kCPU));
    } else {
        storage.emplace_back(scalar_to_tensor(scalar).to(type));
    }
    return storage.back();
}

} // namespace native
} // namespace at_npu

namespace c10_npu::host_regions {
void PrepareOwnedOpApiCallerState()
{
#ifndef BUILD_LIBTORCH
    TORCH_CHECK(c10_npu::impl::NPUTrace::getTrace() == nullptr &&
        !torch_npu::profiler::profDataReportEnable().load(std::memory_order_relaxed),
        "owned OpAPI does not admit Python trace callbacks or active profiler instrumentation");
#endif
    static std::once_flag initialized;
    static bool deterministic = false;
    std::call_once(initialized, [] {
        // A bottom-MLP prefix may still be executing on the original consumer.
        // Drain that host FIFO once before touching its existing cached setter;
        // this is not a device-wide synchronize and is outside every run lock.
        (void)c10_npu::getCurrentNPUStream(0).stream();
        at_npu::native::SetDeterministic();
        deterministic = at::globalContext().deterministicAlgorithms();
    });
    TORCH_CHECK(deterministic == at::globalContext().deterministicAlgorithms(),
        "owned OpAPI requires its caller configuration to remain frozen");
}
} // namespace c10_npu::host_regions
