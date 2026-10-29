#include "r25_hostsim_api.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

#define CHECK(condition) do { if (!(condition)) { std::cerr << __LINE__ << ": " #condition "\n"; std::abort(); } } while (false)

aclTensor* tensor(aclDataType type=ACL_FLOAT, int64_t rows=8, int64_t cols=4) {
    int64_t dims[]={rows,cols}, strides[]={cols,1}, storage[]={rows*cols};
    return aclCreateTensor(dims,2,type,strides,0,ACL_FORMAT_ND,storage,1,reinterpret_cast<void*>(0x1000));
}
aclTensor* index_tensor(int64_t size=2) {
    int64_t stride=1;
    return aclCreateTensor(&size,1,ACL_INT64,&stride,0,ACL_FORMAT_ND,&size,1,reinterpret_cast<void*>(0x2000));
}
struct Bag {
    aclTensor* weight=tensor();
    aclTensor* indices=index_tensor();
    aclTensor* offsets=index_tensor();
    aclTensor* out=tensor(ACL_FLOAT,2,4);
    aclTensor* offset2bag=index_tensor();
    aclTensor* sizes=index_tensor();
    aclTensor* maxindex=index_tensor();
    int prepare(aclOpExecutor** executor, uint64_t* ws) {
        return aclnnEmbeddingBagGetWorkspaceSize(weight,indices,offsets,false,0,false,nullptr,
            false,-1,out,offset2bag,sizes,maxindex,ws,executor);
    }
    ~Bag() { CHECK(aclDestroyTensor(weight)==0); CHECK(aclDestroyTensor(indices)==0);
        CHECK(aclDestroyTensor(offsets)==0); CHECK(aclDestroyTensor(out)==0);
        CHECK(aclDestroyTensor(offset2bag)==0); CHECK(aclDestroyTensor(sizes)==0);
        CHECK(aclDestroyTensor(maxindex)==0); }
};

int main() {
    CHECK(R25HostSimAbi()==1); CHECK(std::strcmp(R25HostSimBackendKind(),"HOST_SIM_R25")==0);
    CHECK(R25HostSimResetDiagnostics()==0); CHECK(R25HostSimSetTracing(true)==0);
    {
        Bag b;
        uint64_t ws=123; aclOpExecutor *first=nullptr,*second=nullptr;
        // Phase1 runs on workers; phase2 follows the caller's selected order.
        std::thread a([&] { CHECK(InitHugeMemThreadLocal(nullptr,false)==0);
            CHECK(b.prepare(&first,&ws)==0); UnInitHugeMemThreadLocal(nullptr,false); }); a.join();
        CHECK(ws==0 && first); const auto first_id=R25HostSimExecutorId(first);
        CHECK(first_id!=0); CHECK(b.prepare(&second,&ws)==0);
        CHECK(first!=second && first_id!=R25HostSimExecutorId(second));
        CHECK(aclnnEmbeddingBag(nullptr,0,second,nullptr)==0);
        CHECK(aclnnEmbeddingBag(nullptr,0,first,nullptr)==0);
        ReleaseHugeMem(nullptr,false);
        CHECK(aclnnEmbeddingBag(nullptr,0,first,nullptr)!=0);
        CHECK(aclDestroyAclOpExecutor(first)!=0);
        CHECK(R25HostSimResetDiagnostics()!=0);
        CHECK(b.prepare(&first,&ws)==0); CHECK(aclDestroyAclOpExecutor(first)==0);
        CHECK(aclDestroyAclOpExecutor(first)!=0);
        CHECK(aclnnEmbeddingBag(nullptr,0,first,nullptr)!=0);
        auto* saved=b.out; b.out=nullptr; first=reinterpret_cast<aclOpExecutor*>(0x1); ws=123;
        CHECK(b.prepare(&first,&ws)!=0); CHECK(first==nullptr && ws==0); b.out=saved;
        CHECK(aclnnEmbeddingBagGetWorkspaceSize(reinterpret_cast<aclTensor*>(0xdead),b.indices,b.offsets,
            false,0,false,nullptr,false,-1,b.out,b.offset2bag,b.sizes,b.maxindex,&ws,&first)!=0);
        auto* bad=tensor(ACL_INT32); saved=b.weight; b.weight=bad;
        CHECK(b.prepare(&first,&ws)!=0); CHECK(aclDestroyTensor(bad)==0); b.weight=saved;
        CHECK(b.prepare(&first,&ws)==0); CHECK(aclDestroyTensor(b.out)==0);
        CHECK(aclnnEmbeddingBag(nullptr,0,first,nullptr)!=0); // consumes failed Execute too
        CHECK(aclDestroyAclOpExecutor(first)!=0); b.out=tensor(ACL_FLOAT,2,4);
    }
    {
        int64_t dims[]={3,4}, strides[]={4,1}, storage[]={12};
        auto* t=aclCreateTensor(dims,2,ACL_FLOAT,strides,2,ACL_FORMAT_ND,storage,1,reinterpret_cast<void*>(0x3000));
        CHECK(t); dims[0]=99; strides[0]=99; storage[0]=99;
        const std::string metadata=R25HostSimTensorMetadataJson(t);
        CHECK(metadata.find("\"shape\":[3,4]")!=std::string::npos);
        CHECK(metadata.find("\"strides\":[4,1]")!=std::string::npos);
        CHECK(metadata.find("\"storage_shape\":[12]")!=std::string::npos);
        CHECK(aclDestroyTensor(t)==0); CHECK(aclDestroyTensor(t)!=0);
        CHECK(!aclCreateTensor(nullptr,1,ACL_FLOAT,strides,0,ACL_FORMAT_ND,storage,1,nullptr));
        CHECK(!aclCreateTensor(dims,2,static_cast<aclDataType>(999),strides,0,ACL_FORMAT_ND,storage,1,nullptr));
        float alpha=2; auto* scalar=aclCreateScalar(&alpha,ACL_FLOAT); CHECK(scalar);
        auto* x=tensor(); auto* y=tensor(); auto* out=tensor();
        aclOpExecutor* ex=nullptr; uint64_t ws=0;
        CHECK(aclnnAddGetWorkspaceSize(x,y,scalar,out,&ws,&ex)==0); alpha=99;
        CHECK(std::string(R25HostSimExecutorMetadataJson(ex)).find("00000040")!=std::string::npos);
        CHECK(aclnnRelu(nullptr,0,ex,nullptr)!=0); // wrong entry consumes executor
        CHECK(aclnnAdd(nullptr,0,ex,nullptr)!=0);
        CHECK(aclnnReluGetWorkspaceSize(x,out,&ws,&ex)==0);
        CHECK(aclnnRelu(nullptr,0,ex,nullptr)==0);
        CHECK(aclnnMmGetWorkspaceSize(x,y,out,1,&ws,&ex)==0);
        CHECK(aclnnMm(nullptr,0,ex,nullptr)==0);
        CHECK(aclDestroyScalar(scalar)==0);
        const aclTensor* children[]={x,y}; auto* list=aclCreateTensorList(children,2); CHECK(list);
        CHECK(aclDestroyTensorList(list)==0); // list destroy owns its converted children
        CHECK(aclDestroyTensor(x)!=0); CHECK(aclDestroyTensor(y)!=0); CHECK(aclDestroyTensor(out)==0);
        int64_t integers[]={1,2}; auto* ints=aclCreateIntArray(integers,2); CHECK(ints); CHECK(aclDestroyIntArray(ints)==0);
        float floats[]={1,2}; auto* fs=aclCreateFloatArray(floats,2); CHECK(fs); CHECK(aclDestroyFloatArray(fs)==0);
        bool bools[]={true,false}; auto* bs=aclCreateBoolArray(bools,2); CHECK(bs); CHECK(aclDestroyBoolArray(bs)==0);
        scalar=aclCreateScalar(&alpha,ACL_FLOAT); const aclScalar* scalars[]={scalar};
        auto* sl=aclCreateScalarList(scalars,1); CHECK(sl); CHECK(aclDestroyScalarList(sl)==0);
        CHECK(aclDestroyScalar(scalar)!=0);
    }
    // Concurrent descriptor/executor churn catches registry races and ABA reuse.
    std::atomic<int> failures{0}; std::vector<std::thread> workers;
    for (int i=0;i<8;++i) workers.emplace_back([&] {
        for (int j=0;j<100;++j) {
            auto* x=tensor(); auto* y=tensor(); aclOpExecutor* ex=nullptr; uint64_t ws=0;
            if (aclnnReluGetWorkspaceSize(x,y,&ws,&ex) || aclnnRelu(nullptr,0,ex,nullptr) ||
                aclDestroyTensor(x) || aclDestroyTensor(y)) ++failures;
        }
    });
    for (auto& thread:workers) thread.join();
    CHECK(failures==0);
    CHECK(!CanUsePTACache("aclnnEmbeddingBag"));
    CHECK(std::string(R25HostSimDiagnosticsJson()).find("\"cache_stub_calls\":0")!=std::string::npos);
    // Detailed trace is checked by the ctypes suite; keep this output compact.
    CHECK(R25HostSimSetTracing(false)==0);
    CHECK(R25HostSimResetDiagnostics()==0); // rejects all descriptor/executor/pool leaks
    CHECK(std::string(R25HostSimDiagnosticsJson()).find("\"live_descriptors\":0")!=std::string::npos);
    std::cout << "HOST_SIM_R25 native contracts passed\n";
}
