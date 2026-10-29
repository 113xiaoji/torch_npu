#include "torch_npu/csrc/core/npu/eager_parallel/EntryDiagnostics.h"
#include <cassert>
#include <stdexcept>
#include <thread>

int main() {
    using c10_npu::eager_parallel::EntryDiagnostics;
    EntryDiagnostics state;
    assert(!state.enabled());
    auto off=state.enter("aten::cat","dispatcher",2);
    assert(off==0);
    state.configure(true,64,536870912);
    auto outer=state.enter("aten::_embedding_bag","dispatcher",3);
    auto inner=state.enter("aten::_embedding_bag","op_plugin",3);
    assert(outer && inner);
    bool refused=false;
    try { state.close(); } catch(const std::logic_error&) { refused=true; }
    assert(refused && state.enabled());
    state.complete(inner,4);
    state.complete(outer,4);
    auto one=state.snapshot();
    assert(one.entered==1 && one.completed==1 && one.nested==1 && one.active==0);
    assert(one.operators.at("aten::_embedding_bag").outputs==4);
    std::thread foreign([&] { assert(state.enter("aten::relu","dispatcher",1)==0); });
    foreign.join();
    assert(state.snapshot().foreign_calls==1 && state.snapshot().coverage_incomplete);
    state.boundary("storage.resize");
    assert(state.snapshot().boundaries.at("storage.resize")==1);
    state.close();
    assert(!state.enabled() && state.snapshot().closed);
    state.configure(true,64,536870912);
    auto failed=state.enter("aten::cat","dispatcher",2);
    state.fail(failed,"original error");
    auto failed2=state.enter("aten::bmm","dispatcher",2);
    state.fail(failed2,"later error");
    assert(state.snapshot().first_error=="original error");
    assert(state.snapshot().failed==2 && state.snapshot().active==0);
    state.close();
    state.configure(true,64,536870912);
    auto token=state.enter("aten::bmm","dispatcher",2);
    state.complete(token,1);
    state.complete(token,1); // A duplicate ACK is diagnosed, never counted twice.
    assert(state.snapshot().completed==1 && state.snapshot().coverage_incomplete);
    state.close();
    assert(state.snapshot().active==0);
    state.configure(true,64,536870912);
    for (int i=0;i<600;++i) {
        const auto name="aten::schema_"+std::to_string(i);
        const auto entry=state.enter(name.c_str(),"dispatcher",0);
        state.complete(entry,0);
    }
    assert(state.snapshot().coverage_incomplete && state.snapshot().operators.size()==512);
    state.complete(state.enter("aten::schema_0","dispatcher",0),0);
    assert(state.snapshot().operators.at("aten::schema_0").completed==2);
    for(int i=0;i<70;++i) { const auto name="boundary_"+std::to_string(i);state.boundary(name.c_str()); }
    state.boundary("boundary_0");
    assert(state.snapshot().boundaries.size()==64 && state.snapshot().boundaries.at("boundary_0")==2);
    state.close();
}
