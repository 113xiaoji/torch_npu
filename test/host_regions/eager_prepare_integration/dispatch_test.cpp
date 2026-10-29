#include "torch_npu/csrc/core/npu/EagerPrepareEligibility.h"
#include "torch_npu/csrc/core/npu/EagerPrepareDispatch.h"
#include <cassert>
#include <stdexcept>
#include <tuple>
namespace {
bool enabled=false,qualified=true,domain=true;
bool sdk_failure=false;
int calls=0,alive=0,original=0,queued=0,fences=0;
struct Temporary {
    Temporary() { ++alive; }
    Temporary(const Temporary&)=delete;
    ~Temporary() { --alive; }
};
Temporary make() { ++calls; return {}; }
void original_call(const Temporary&) { assert(alive==1); ++original; }
}
namespace c10_npu::eager_prepare {
bool Enabled() { return enabled; }
bool Admit(const char*,bool profile,const char*,bool arguments) { if (!profile || !arguments) { ++fences; return false; } return true; }
}
namespace hdb_opapi {
constexpr bool HasEagerTypedSchema(const char*) { return true; }
bool EagerSdkQualified(const char*) { return qualified; }
template<class T> bool EagerArgumentDomain(const char*,const T&) { return domain; }
template<class T> void EnqueueEagerArguments(const char*,bool,const T&) {
    assert(alive==1); ++queued;
    if (sdk_failure) throw std::runtime_error("unique-sdk-or-allocation-failure");
}
}
#define ORIGINAL(api, ...) do { original_call(__VA_ARGS__); } while(false)
#define CALL(...) HDB_EAGER_PREPARE_R24(aclnnEmbeddingBag, ORIGINAL, true, __VA_ARGS__)
int main() {
    CALL(make()); assert(calls==1 && original==1 && queued==0 && fences==0 && alive==0);
    enabled=true; qualified=false; CALL(make()); assert(calls==2 && original==2 && queued==0 && fences==1 && alive==0);
    qualified=true; domain=false; CALL(make()); assert(calls==3 && original==3 && queued==0 && fences==2 && alive==0);
    domain=true; CALL(make()); assert(calls==4 && original==3 && queued==1 && alive==0);
    sdk_failure=true;
    try { CALL(make()); assert(false); }
    catch (const std::runtime_error& e) { assert(std::string(e.what())=="unique-sdk-or-allocation-failure"); }
    assert(calls==5 && original==3 && queued==2 && alive==0); // Never rerun after failure.
    using c10_npu::eager_prepare::EmbeddingBagDomain;
    EmbeddingBagDomain valid{true,true,true,true,true,true,false,0,true,false,false,true};
    assert(valid.Admitted());
    auto mean=valid; mean.mode=1; assert(!mean.Admitted());
    auto maximum=valid; maximum.mode=2; assert(!maximum.Admitted());
    auto empty=valid; empty.nonempty=false; assert(!empty.Admitted());
    auto half=valid; half.fp32_weight=false; assert(!half.Admitted());
    auto int32=valid; int32.int64_indices_offsets=false; assert(!int32.Admitted());
    return 0;
}
