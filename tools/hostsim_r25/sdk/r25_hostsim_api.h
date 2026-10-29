#pragma once
// Standalone, typed ABI for a metadata-only simulation. Never a real SDK claim.
#include <cstddef>
#include <cstdint>
#if defined(_WIN32)
# if defined(R25_HOSTSIM_BUILD)
#  define R25_API __declspec(dllexport)
# else
#  define R25_API __declspec(dllimport)
# endif
#else
# define R25_API __attribute__((visibility("default")))
#endif

// Use CANN's enum declarations when explicitly requested by the remote build.
#if defined(R25_HOSTSIM_USE_ACL_HEADERS)
# include <acl/acl_base.h>
#else
enum aclDataType : int { ACL_DT_UNDEFINED=-1, ACL_FLOAT=0, ACL_FLOAT16=1, ACL_INT8=2,
    ACL_INT32=3, ACL_UINT8=4, ACL_INT16=6, ACL_UINT16=7, ACL_UINT32=8, ACL_INT64=9,
    ACL_UINT64=10, ACL_DOUBLE=11, ACL_BOOL=12, ACL_COMPLEX64=16, ACL_COMPLEX128=17,
    ACL_BF16=27 };
enum aclFormat : int { ACL_FORMAT_NCHW=0, ACL_FORMAT_NHWC=1, ACL_FORMAT_ND=2,
    ACL_FORMAT_NC1HWC0=3, ACL_FORMAT_FRACTAL_Z=4, ACL_FORMAT_NCDHW=27,
    ACL_FORMAT_NCL=47 };
using aclrtStream = void*;
#endif
struct aclTensor;
struct aclScalar;
struct aclIntArray;
struct aclFloatArray;
struct aclBoolArray;
struct aclTensorList;
struct aclScalarList;
struct aclOpExecutor;

enum R25HostSimStatus : int { R25_SUCCESS=0, R25_INVALID_ARGUMENT=100001,
    R25_STALE_HANDLE=100002, R25_WRONG_KIND=100003, R25_UNSUPPORTED=100004,
    R25_OUT_OF_MEMORY=100005, R25_LIFECYCLE_ERROR=100006 };

extern "C" {
R25_API int R25HostSimAbi();
R25_API const char* R25HostSimBackendKind();
// Returned UTF-8 JSON lives in the calling thread until its next diagnostic call.
R25_API const char* R25HostSimDiagnosticsJson();
R25_API int R25HostSimResetDiagnostics(); // Rejects reset while any object/pool lives.
R25_API int R25HostSimSetTracing(bool); // Bounded cold diagnostics; false by default.
R25_API const char* R25HostSimLastError();
R25_API uint64_t R25HostSimExecutorId(const aclOpExecutor*);
R25_API const char* R25HostSimTensorMetadataJson(const aclTensor*);
R25_API const char* R25HostSimExecutorMetadataJson(const aclOpExecutor*);

R25_API aclTensor* aclCreateTensor(const int64_t*, uint64_t, aclDataType,
    const int64_t*, int64_t, aclFormat, const int64_t*, uint64_t, void*);
R25_API int aclDestroyTensor(const aclTensor*);
R25_API aclScalar* aclCreateScalar(void*, aclDataType);
R25_API int aclDestroyScalar(const aclScalar*);
R25_API aclIntArray* aclCreateIntArray(const int64_t*, uint64_t);
R25_API int aclDestroyIntArray(const aclIntArray*);
R25_API aclFloatArray* aclCreateFloatArray(const float*, uint64_t);
R25_API int aclDestroyFloatArray(const aclFloatArray*);
R25_API aclBoolArray* aclCreateBoolArray(const bool*, uint64_t);
R25_API int aclDestroyBoolArray(const aclBoolArray*);
R25_API aclTensorList* aclCreateTensorList(const aclTensor* const*, uint64_t);
R25_API int aclDestroyTensorList(const aclTensorList*);
R25_API aclScalarList* aclCreateScalarList(const aclScalar* const*, uint64_t);
R25_API int aclDestroyScalarList(const aclScalarList*);
R25_API int aclDestroyAclOpExecutor(aclOpExecutor*);

// Schema checked against ops-nn 814ce56 index/embedding_bag/op_api header.
R25_API int aclnnEmbeddingBagGetWorkspaceSize(const aclTensor*, const aclTensor*,
    const aclTensor*, bool, int64_t, bool, const aclTensor*, bool, int64_t,
    aclTensor*, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
R25_API int aclnnEmbeddingBag(void*, uint64_t, aclOpExecutor*, aclrtStream);
R25_API int aclnnAddGetWorkspaceSize(const aclTensor*, const aclTensor*,
    const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
R25_API int aclnnAdd(void*, uint64_t, aclOpExecutor*, aclrtStream);
R25_API int aclnnReluGetWorkspaceSize(const aclTensor*, const aclTensor*,
    uint64_t*, aclOpExecutor**);
R25_API int aclnnRelu(void*, uint64_t, aclOpExecutor*, aclrtStream);
R25_API int aclnnMmGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*,
    int8_t, uint64_t*, aclOpExecutor**);
R25_API int aclnnMm(void*, uint64_t, aclOpExecutor*, aclrtStream);

R25_API int InitHugeMemThreadLocal(void*, bool);
R25_API void UnInitHugeMemThreadLocal(void*, bool);
R25_API void ReleaseHugeMem(void*, bool);
R25_API bool CanUsePTACache(const char*);
R25_API void InitPTACacheThreadLocal();
R25_API void UnInitPTACacheThreadLocal();
R25_API void SetPTAHashKey(uint64_t);
R25_API void SetPTACacheHashKey(uint8_t*, size_t);
R25_API aclOpExecutor* PTAGetExecCache(uint64_t, uint64_t*);
R25_API aclOpExecutor* PTAFindExecCache(uint8_t*, size_t, uint64_t*);
R25_API void AddTensorAddrToCachedList(void*);
}
