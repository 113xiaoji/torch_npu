// CPU numerical emulation behind the native ACL ABI. This is not CANN.
#include "runtime.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <map>
#include <mutex>
#include <memory>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <atomic>

namespace {
struct Tensor {
    std::vector<int64_t> shape,strides;
    void* data=nullptr;
    int64_t offset=0;
    at::ScalarType dtype=at::kFloat;
    at::Tensor value() const {
        static char empty[16];
        auto* start=data?static_cast<char*>(data)+offset*c10::elementSize(dtype):empty;
        return at::from_blob(start,shape,strides,at::TensorOptions().dtype(dtype).device(at::kCPU));
    }
};
enum class Kind {Tensor,Scalar,List,Array,Executor};
struct Object {
    Kind kind=Kind::Tensor;
    Tensor tensor;
    at::Scalar scalar=0;
    std::vector<void*> children;
    std::vector<unsigned char> bytes;
    std::function<void()> run;
    std::string operation;
};
struct Counts {uint64_t prepared=0,executed=0,cancelled=0,failed=0;std::map<uintptr_t,uint64_t> stream_executions;};
std::mutex mutex;
uintptr_t token=0x10000;
std::map<void*,std::shared_ptr<Object>> objects;
std::map<std::string,Counts> operations;
thread_local std::string diagnostic;
std::atomic<uint64_t> descriptors_created{0},descriptors_destroyed{0};
std::atomic<uint64_t> active_kernels{0},peak_active_kernels{0};
at::ScalarType dtype(int value) {
    switch(value){case 0:return at::kFloat;case 1:return at::kHalf;case 2:return at::kChar;
    case 3:return at::kInt;case 4:return at::kByte;case 6:return at::kShort;case 9:return at::kLong;
    case 11:return at::kDouble;case 12:return at::kBool;case 27:return at::kBFloat16;default:throw std::runtime_error("unsupported hostsim dtype");}
}
std::shared_ptr<Object> object(const void* handle,Kind kind) {
    std::lock_guard<std::mutex> lock(mutex);auto it=objects.find(const_cast<void*>(handle));
    if(it==objects.end()||it->second->kind!=kind)throw std::runtime_error("stale or wrong-kind descriptor");return it->second;
}
void* insert(std::shared_ptr<Object> value) {
    std::lock_guard<std::mutex> lock(mutex);auto* handle=reinterpret_cast<void*>(token+=16);objects[handle]=value;
    if(value->kind!=Kind::Executor)++descriptors_created;return handle;
}
Tensor tensor(const void* handle){return object(handle,Kind::Tensor)->tensor;}
at::Scalar scalar(const void* handle){return object(handle,Kind::Scalar)->scalar;}
std::vector<Tensor> tensor_list(const void* handle){std::vector<Tensor> result;for(auto* child:object(handle,Kind::List)->children)result.push_back(tensor(child));return result;}
void write(const Tensor& output,const at::Tensor& value){auto target=output.value();if(target.sizes()!=value.sizes())throw std::runtime_error("hostsim output shape differs");target.copy_(value);}
template<class F> int checked(F function) noexcept {try{function();return 0;}catch(const std::exception& exc){return r27::failure(exc.what());}catch(...){return r27::failure("unknown OpAPI error");}}
template<class F> void* pointer(F function) noexcept {try{return function();}catch(const std::exception& exc){r27::failure(exc.what());return nullptr;}}
int prepare(const char* name,std::function<void()> function,uint64_t* size,void** executor) {
    if(!size||!executor)throw std::runtime_error("null prepare outputs");
    auto value=std::make_shared<Object>();value->kind=Kind::Executor;value->operation=name;value->run=std::move(function);
    *size=0;*executor=insert(value);std::lock_guard<std::mutex> lock(mutex);++operations[name].prepared;return 0;
}
int execute(const char* name,uint64_t size,void* executor,void* stream) {
    return checked([&]{auto value=object(executor,Kind::Executor);if(value->operation!=name||size)throw std::runtime_error("executor/size mismatch");
        {std::lock_guard<std::mutex> lock(mutex);objects.erase(executor);}
        r27::enqueue(stream,[value,stream]{const auto active=++active_kernels;auto peak=peak_active_kernels.load();while(peak<active&&!peak_active_kernels.compare_exchange_weak(peak,active)){}
            try{value->run();std::lock_guard<std::mutex> lock(mutex);auto& counts=operations.at(value->operation);++counts.stream_executions[reinterpret_cast<uintptr_t>(stream)];++counts.executed;}
            catch(...){--active_kernels;std::lock_guard<std::mutex> lock(mutex);++operations.at(value->operation).failed;throw;}
            --active_kernels;});
    });
}
void release(const void* handle,Kind kind){if(!handle)return;auto value=object(handle,kind);std::lock_guard<std::mutex> lock(mutex);if(kind==Kind::List)for(auto* child:value->children){if(objects.erase(child))++descriptors_destroyed;}objects.erase(const_cast<void*>(handle));++descriptors_destroyed;}
}
#define EXECUTE(name) int name(void*,uint64_t size,void* executor,void* stream){return execute(#name,size,executor,stream);}
#define UNARY(name,expression) int name##GetWorkspaceSize(const void* input,const void* output,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input),y=tensor(output);prepare(#name,[=]{write(y,expression);},size,executor);});} EXECUTE(name)
extern "C" {
void* aclCreateTensor(const int64_t* dims,uint64_t rank,int type,const int64_t* strides,int64_t offset,int format,const int64_t*,uint64_t,void* data){return pointer([&]()->void*{
    if(rank>64||offset<0||(rank&&(!dims||!strides))||(format!=0&&format!=2&&format!=47))throw std::runtime_error("unsupported tensor metadata/format");
    auto value=std::make_shared<Object>();value->tensor.dtype=dtype(type);value->tensor.offset=offset;value->tensor.data=data;
    if(rank){value->tensor.shape.assign(dims,dims+rank);value->tensor.strides.assign(strides,strides+rank);}
    bool empty=false;for(size_t i=0;i<rank;++i){if(dims[i]<0||strides[i]<0)throw std::runtime_error("negative dimension/stride");empty|=dims[i]==0;}
    if(!data&&!empty)throw std::runtime_error("null tensor data");return insert(value);});}
int aclDestroyTensor(const void* value){return checked([&]{release(value,Kind::Tensor);});}
void* aclCreateScalar(void* data,int type){return pointer([&]()->void*{if(!data)throw std::runtime_error("null scalar");auto value=std::make_shared<Object>();value->kind=Kind::Scalar;
    value->scalar=at::from_blob(data,{},at::TensorOptions().dtype(dtype(type))).item();return insert(value);});}
int aclDestroyScalar(const void* value){return checked([&]{release(value,Kind::Scalar);});}
void* aclCreateTensorList(const void* const* values,uint64_t count){return pointer([&]()->void*{auto value=std::make_shared<Object>();value->kind=Kind::List;for(uint64_t i=0;i<count;++i){tensor(values[i]);value->children.push_back(const_cast<void*>(values[i]));}return insert(value);});}
int aclDestroyTensorList(const void* value){return checked([&]{release(value,Kind::List);});}
void* aclCreateIntArray(const int64_t* data,uint64_t count){return pointer([&]()->void*{auto value=std::make_shared<Object>();value->kind=Kind::Array;value->bytes.resize(count*8);if(count)std::memcpy(value->bytes.data(),data,count*8);return insert(value);});}
int aclDestroyIntArray(const void* value){return checked([&]{release(value,Kind::Array);});}
void* aclCreateBoolArray(const bool* data,uint64_t count){return pointer([&]()->void*{auto value=std::make_shared<Object>();value->kind=Kind::Array;value->bytes.resize(count);if(count)std::memcpy(value->bytes.data(),data,count);return insert(value);});}
int aclDestroyBoolArray(const void* value){return checked([&]{release(value,Kind::Array);});}
void* aclCreateFloatArray(const float* data,uint64_t count){return pointer([&]()->void*{auto value=std::make_shared<Object>();value->kind=Kind::Array;value->bytes.resize(count*4);if(count)std::memcpy(value->bytes.data(),data,count*4);return insert(value);});}
int aclDestroyFloatArray(const void* value){return checked([&]{release(value,Kind::Array);});}
int aclDestroyAclOpExecutor(void* handle){return checked([&]{auto value=object(handle,Kind::Executor);std::lock_guard<std::mutex> lock(mutex);++operations[value->operation].cancelled;objects.erase(handle);});}
UNARY(aclnnRelu,at::relu(x.value()))
UNARY(aclnnSigmoid,at::sigmoid(x.value()))
int aclnnInplaceReluGetWorkspaceSize(const void* input,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input);prepare("aclnnInplaceRelu",[=]{write(x,at::relu(x.value()));},size,executor);});} EXECUTE(aclnnInplaceRelu)
int aclnnCastGetWorkspaceSize(const void* input,int type,const void* output,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input),y=tensor(output);auto kind=dtype(type);prepare("aclnnCast",[=]{write(y,x.value().to(kind));},size,executor);});} EXECUTE(aclnnCast)
int aclnnInplaceCopyGetWorkspaceSize(const void* input,const void* other,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input),y=tensor(other);prepare("aclnnInplaceCopy",[=]{write(x,y.value());},size,executor);});} EXECUTE(aclnnInplaceCopy)
int aclnnAddGetWorkspaceSize(const void* input,const void* other,const void* alpha,const void* output,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input),y=tensor(other),z=tensor(output);auto a=scalar(alpha);prepare("aclnnAdd",[=]{write(z,at::add(x.value(),y.value(),a));},size,executor);});} EXECUTE(aclnnAdd)
int aclnnMmGetWorkspaceSize(const void* input,const void* other,const void* output,int8_t,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input),y=tensor(other),z=tensor(output);prepare("aclnnMm",[=]{write(z,at::mm(x.value(),y.value()));},size,executor);});} EXECUTE(aclnnMm)
int aclnnBatchMatMulGetWorkspaceSize(const void* input,const void* other,const void* output,int8_t,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input),y=tensor(other),z=tensor(output);prepare("aclnnBatchMatMul",[=]{write(z,at::bmm(x.value(),y.value()));},size,executor);});} EXECUTE(aclnnBatchMatMul)
int aclnnAddmmGetWorkspaceSize(const void* bias,const void* input,const void* other,const void* beta,const void* alpha,const void* output,int8_t,uint64_t* size,void** executor){return checked([&]{auto b=tensor(bias),x=tensor(input),y=tensor(other),z=tensor(output);auto a=scalar(alpha),bscale=scalar(beta);prepare("aclnnAddmm",[=]{write(z,at::addmm(b.value(),x.value(),y.value(),bscale,a));},size,executor);});} EXECUTE(aclnnAddmm)
int aclnnCatGetWorkspaceSize(const void* inputs,int64_t dim,const void* output,uint64_t* size,void** executor){return checked([&]{auto values=tensor_list(inputs);auto z=tensor(output);prepare("aclnnCat",[=]{std::vector<at::Tensor> tensors;for(const auto& value:values)tensors.push_back(value.value());write(z,at::cat(tensors,dim));},size,executor);});} EXECUTE(aclnnCat)
int aclnnIndexGetWorkspaceSize(const void* input,const void* indices,const void* output,uint64_t* size,void** executor){return checked([&]{auto x=tensor(input),z=tensor(output);auto values=tensor_list(indices);prepare("aclnnIndex",[=]{c10::List<std::optional<at::Tensor>> ix;for(const auto& value:values){auto v=value.value();
    // OpPlugin IndexKernelNpuOpApi.cpp represents None with an empty tensor
    // created from self's FP32 options. Empty Int/Long tensors are real indices.
    if(v.numel()==0&&v.scalar_type()==at::kFloat)ix.push_back(std::nullopt);else ix.push_back(v);
    }write(z,at::index(x.value(),ix));},size,executor);});} EXECUTE(aclnnIndex)
int aclnnEmbeddingBagGetWorkspaceSize(const void* weight,const void* indices,const void* offsets,bool scale,int64_t mode,bool sparse,const void* samples,bool last,int64_t padding,const void* output,const void* offset2bag,const void* bagsize,const void* maxindices,uint64_t* size,void** executor){return checked([&]{
    auto w=tensor(weight),i=tensor(indices),o=tensor(offsets),y=tensor(output),a=tensor(offset2bag),b=tensor(bagsize),c=tensor(maxindices);
    std::optional<Tensor> sample;if(samples)sample=tensor(samples);
    if(mode!=0||scale||sample)throw std::runtime_error("R27 numerical EmbeddingBag supports inference sum without sample weights only");
    prepare("aclnnEmbeddingBag",[=]{auto values=at::_embedding_bag(w.value(),i.value(),o.value(),scale,mode,sparse,std::nullopt,last,padding);write(y,std::get<0>(values));
        // For this sum inference profile auxiliary tensors are not consumed.
        // Match defined CPU values when sizes agree; zero otherwise, declared
        // simulation policy only, never a claim about unspecified CANN cells.
        auto aux=[&](const Tensor& dst,const at::Tensor& src){auto target=dst.value();if(target.sizes()==src.sizes())target.copy_(src);else target.zero_();};
        aux(a,std::get<1>(values));aux(b,std::get<2>(values));aux(c,std::get<3>(values));
    },size,executor);
});} EXECUTE(aclnnEmbeddingBag)
bool CanUsePTACache(const char*){return false;}
int InitHugeMemThreadLocal(void*,bool){return 0;}
void UnInitHugeMemThreadLocal(void*,bool){}
void ReleaseHugeMem(void*,bool){}
void InitPTACacheThreadLocal(){r27::failure("PTA cache unsupported");}
void UnInitPTACacheThreadLocal(){}
const char* R27HostSimOpDiagnosticsJson(){std::lock_guard<std::mutex> lock(mutex);std::ostringstream out;out<<"{\"backend_kind\":\"HOST_SIM_R27_CPU\",\"numeric_supported\":true,\"numeric_scope\":\"CPU DLRM inference operator closure; EmbeddingBag sum without scale or sample weights\",\"embedding_auxiliary_policy\":\"CPU values on shape match, zero otherwise\",\"auxiliary_numeric_qualified\":false,\"descriptors_created\":"<<descriptors_created<<",\"descriptors_destroyed\":"<<descriptors_destroyed<<",\"live_objects\":"<<objects.size()<<",\"active_kernels\":"<<active_kernels<<",\"peak_active_kernels\":"<<peak_active_kernels<<",\"operations\":{";bool comma=false;for(const auto& item:operations){if(comma)out<<',';comma=true;out<<'"'<<item.first<<"\":{\"prepared\":"<<item.second.prepared<<",\"executed\":"<<item.second.executed<<",\"cancelled\":"<<item.second.cancelled<<",\"failed\":"<<item.second.failed<<",\"stream_executions\":{";bool stream_comma=false;for(const auto& stream:item.second.stream_executions){if(stream_comma)out<<',';stream_comma=true;out<<'"'<<stream.first<<"\":"<<stream.second;}out<<"}}";}out<<"}}";diagnostic=out.str();return diagnostic.c_str();}
}
