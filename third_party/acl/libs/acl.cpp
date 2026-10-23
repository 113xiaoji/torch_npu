#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "acl/acl_mdl.h"


#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cstdint>
static bool qwen_sim_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("QWEN_ACL_HOST_SIM");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static thread_local int32_t qwen_device = 0;
static thread_local void *qwen_context = reinterpret_cast<void *>(uintptr_t(1));
static aclError qwen_alloc(void **ptr, size_t size) {
    if (!qwen_sim_enabled()) return 100000;
    *ptr = std::malloc(size ? size : 1);
    return *ptr ? 0 : 100000;
}

extern "C" {
// 资源初始化，申请与释放
aclError aclInit(const char *configPath){return 0;}
aclError aclFinalize(){return 0;}
aclError aclrtFree(void *devPtr){std::free(devPtr); return 0;}
aclError aclrtGetDevice(int32_t *deviceId){if(deviceId) *deviceId = qwen_device; return 0;}
aclError aclrtResetDevice(int32_t deviceId){return 0;}
aclError aclrtSetDevice(int32_t deviceId){qwen_device = deviceId; return 0;}
aclError aclrtGetDeviceCount(uint32_t *count){if(count) *count = qwen_sim_enabled() ? 1 : 0; return 0;}
aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId){if(canAccessPeer) *canAccessPeer=0; return 0;}
aclError aclrtSynchronizeDevice(void){return 0;}
aclError aclmdlSetDump(const char *configPath){return 0;}
aclError aclmdlInitDump(){return 0;}
aclError aclmdlFinalizeDump(){return 0;}
aclError aclrtDeviceTaskAbort(int32_t deviceId, uint32_t timeout){return 0;}
aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t* value){return 0;}
aclError aclrtSetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t value){return 0;}
aclError aclrtResetDeviceResLimit(int32_t deviceId){return 0;}
aclError aclrtSetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t value){return 0;}
aclError aclrtResetStreamResLimit(aclrtStream stream){return 0;}
aclError aclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t* value){return 0;}
aclError aclrtUseStreamResInCurrentThread(aclrtStream stream){return 0;}
aclError aclrtUnuseStreamResInCurrentThread(aclrtStream stream){return 0;}
aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value){return 0;}

// Stream
aclError aclrtCreateStream(aclrtStream *stream) {if(stream) *stream=std::malloc(1); return 0;}
aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) { return 0; }
aclError aclrtSetOpWaitTimeout(uint32_t timeout) { return 0; }
aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {if(stream) *stream=std::malloc(1); return 0;}
aclError aclrtDestroyStream(aclrtStream stream){std::free(stream); return 0;}
aclError aclrtDestroyStreamForce(aclrtStream stream){std::free(stream); return 0;}
aclError aclrtSynchronizeStream(aclrtStream stream){return 0;}
aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout){return 0;}

// Event
aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status){return 0;}
aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status){return 0;}
aclError aclrtCreateEvent(aclrtEvent *event){if(event) *event=std::malloc(1); return 0;}
aclError aclrtDestroyEvent(aclrtEvent event){std::free(event); return 0;}
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream){return 0;}
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event){return 0;}
aclError aclrtSynchronizeEvent(aclrtEvent event){return 0;}
aclError aclrtEventElapsedTime(float *ms, aclrtEvent start, aclrtEvent end){return 0;}
aclError aclrtIpcGetEventHandle(aclrtEvent event, aclrtIpcEventHandle *handle){return 0;}
aclError aclrtIpcOpenEventHandle(aclrtIpcEventHandle handle, aclrtEvent *event){return 0;}

// memory相关操作
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy){return qwen_alloc(devPtr,size);}
aclError aclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy){return qwen_alloc(devPtr,size);}
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind){if(count>destMax || (!dst && count) || (!src && count)) return 100000; if(count) std::memcpy(dst,src,count); return 0;}
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src,
                          size_t count, aclrtMemcpyKind kind, aclrtStream stream){return aclrtMemcpy(dst,destMax,src,count,kind);}
aclError aclrtMemcpyAsyncWithCondition(void *dst, size_t destMax, const void *src,
                                       size_t count, aclrtMemcpyKind kind, aclrtStream stream){return aclrtMemcpy(dst,destMax,src,count,kind);}
aclError aclrtMallocHost(void **hostPtr, size_t size){return qwen_alloc(hostPtr,size);}
aclError aclrtFreeHost(void *hostPtr){std::free(hostPtr); return 0;}
aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total){if(free) *free=size_t(32)<<30; if(total) *total=size_t(32)<<30; return 0;}
aclError aclrtGetMemUceInfo(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize, size_t *retSize){return 0;}
aclError aclrtMemUceRepair(int32_t deviceId, aclrtMemUceInfo* memUceInfoArray, size_t arraySize){return 0;}
aclError aclrtCmoAsync(void *src, size_t size, aclrtCmoType cmoType, aclrtStream stream){return 0;}
aclError aclrtGetLastError(aclrtLastErrLevel flag){return 0;}
aclError aclrtPeekAtLastError(aclrtLastErrLevel flag){return 0;}


// op相关操作
aclopAttr *aclopCreateAttr(){return NULL;}
void aclopDestroyAttr(const aclopAttr *attr){return;}
aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue){return 0;}
aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue){return 0;}
aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue){return 0;}
aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue){return 0;}
aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues, const int64_t *values){return 0;}
aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues, const float *values){return 0;}
aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues, const uint8_t *values){return 0;}
aclError aclopSetAttrDataType(aclopAttr *attr, const char *attrName, aclDataType values){return 0;}
// Tensor相关
aclTensorDesc *aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format){return NULL;}
void aclDestroyTensorDesc(const aclTensorDesc *desc){return;}
aclDataBuffer *aclCreateDataBuffer(void *data, size_t size){return NULL;}
aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer){return 0;}
void aclSetTensorDescName(aclTensorDesc *desc, const char *name){return;}
aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format){return 0;}
aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims){return 0;}
aclError aclSetTensorShapeRange(aclTensorDesc* desc, size_t dimsCount, int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM]){return 0;}
aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize) {return 0;}

size_t aclGetTensorDescNumDims(const aclTensorDesc *desc) {return 0;}
aclDataType aclGetTensorDescType(const aclTensorDesc *desc) {return ACL_FLOAT;}
int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index) {return 0;}
aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc) {return ACL_FORMAT_NCHW;}
const char *aclGetTensorDescName(aclTensorDesc *desc) {return NULL;}

aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType type) {return 0;};
aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {return 0;};
aclError aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo) {return 0;};

void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line, const char *fmt, ...) {}
aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback) {return 0;}
aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {if(event) *event=std::malloc(1); return 0;}
aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) {if(event) *event=std::malloc(1); return 0;}
aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream){return 0;}
aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status) {return 0;};
aclError aclmdlRIDebugJsonPrint(aclmdlRI modelRI, const char* path, uint32_t flags) {return 0;};
aclError aclrtGetErrorVerbose(int32_t deviceId, aclrtErrorInfo *errorInfo) {return 0;};
aclError aclrtRepairError(int32_t deviceId, const aclrtErrorInfo *errorInfo) {return 0;}
}

extern "C" {
aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) { return 100000; }
aclError aclrtDeviceGetBareTgid(int32_t *pid) { if(pid) *pid=int32_t(getpid()); return 0; }
aclError aclrtFreePhysical(aclrtDrvMemHandle handle) { return 100000; }
aclError aclrtGetCurrentContext(aclrtContext *context) { if(context) *context=qwen_context; return 0; }
const char *aclrtGetSocName() { return "Ascend910B1"; }
aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags) { return 100000; }
aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) { return 100000; }
aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType handleType, uint64_t flags, uint64_t *shareableHandle) { return 100000; }
aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle *handle) { return 100000; }
aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) { if(count>maxCount || (!devPtr && count)) return 100000; if(count) std::memset(devPtr,value,count); return 0; }
aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t *pid, size_t pidNum) { return 100000; }
aclError aclrtReleaseMemAddress(void *virPtr) { return 100000; }
aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags) { return 100000; }
aclError aclrtSetCurrentContext(aclrtContext context) { qwen_context=context; return 0; }
aclError aclrtUnmapMem(void *virPtr) { return 100000; }
}

#include "profiling/aprof_pub.h"
extern "C" {
uint64_t MsprofSysCycleTime(void) { return 0; }
int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle) { return 0; }
int32_t MsprofReportAdditionalInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length) { return 0; }
int32_t MsprofReportCompactInfo(uint32_t nonPersistantFlag, const VOID_PTR data, uint32_t length) { return 0; }
int32_t MsprofReportApi(uint32_t nonPersistantFlag, const struct MsprofApi *api) { return 0; }
uint64_t MsprofGetHashId(const char *hashInfo, size_t length) { return 0; }
}

extern "C" {
aclError aclrtProcessReport(int32_t) { return 0; }
aclError aclrtCtxSetSysParamOpt(aclSysParamOpt, int64_t) { return 0; }
aclError aclrtSetOpExecuteTimeOut(uint32_t) { return 0; }
aclError aclrtSetOpExecuteTimeOutWithMs(uint32_t) { return 0; }
aclError aclrtSetOpExecuteTimeOutV2(uint64_t, uint64_t *actual) { if (actual) *actual = 0; return 0; }
aclError aclGetDeviceCapability(uint32_t, aclDeviceInfo info, int64_t *value) {
    if (!value) return 100000;
    if (info == ACL_DEVICE_INFO_AI_CORE_NUM) *value = 20;
    else if (info == ACL_DEVICE_INFO_VECTOR_CORE_NUM) *value = 40;
    else if (info == ACL_DEVICE_INFO_L2_SIZE) *value = 4 * 1024 * 1024;
    else *value = 1;
    return 0;
}
}

#include "../../../torch_npu/csrc/inductor/mlir/hacl_rt.h"
extern "C" {
rtError_t rtDevBinaryRegister(const rtDevBinary_t *, void **handle) { if (handle) *handle = reinterpret_cast<void *>(uintptr_t(1)); return RT_ERROR_NONE; }
rtError_t rtFunctionRegister(void *, const void *, const char *, const void *, uint32_t) { return RT_ERROR_NONE; }
rtError_t rtKernelLaunch(const void *, uint32_t, void *, uint32_t, rtSmDesc_t *, rtStream_t) { return RT_ERROR_NONE; }
}
