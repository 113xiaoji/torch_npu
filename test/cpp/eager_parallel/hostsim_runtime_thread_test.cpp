// Link the actual tools/hostsim_r27/runtime.cpp directly, without Torch or CANN.
#include "runtime.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::size_t kRounds=1000, kWidth=32;
struct Case {
    std::array<int,kWidth> a{},b{},middle_a{},middle_b{},out_a{},out_b{};
    void* event_a=nullptr;
    void* event_b=nullptr;
    std::atomic<bool> ready_a{false},ready_b{false};
};
void success(int code) {
    if(code!=0) { std::cerr<<R27HostSimLastError()<<'\n'; std::abort(); }
}
std::uint64_t field(const std::string& text,const std::string& name) {
    const auto prefix='"'+name+"\":";
    const auto offset=text.find(prefix); assert(offset!=std::string::npos);
    return std::stoull(text.substr(offset+prefix.size()));
}
}
int main() {
    void *producer_a=nullptr,*producer_b=nullptr,*consumer=nullptr;
    success(R27StreamCreate(&producer_a)); success(R27StreamCreate(&producer_b)); success(R27StreamCreate(&consumer));
    std::vector<Case> cases(kRounds);
    for(std::size_t i=0;i<kRounds;++i) {
        auto& item=cases[i];
        for(std::size_t j=0;j<kWidth;++j) {
            item.a[j]=static_cast<int>(i*37+j); item.b[j]=-static_cast<int>(i*71+j+1);
            item.middle_a[j]=item.middle_b[j]=item.out_a[j]=item.out_b[j]=-987654;
        }
        success(R27EventCreate(&item.event_a)); success(R27EventCreate(&item.event_b));
    }
    std::atomic<bool> done{false};
    std::thread diagnostics([&]{
        while(!done.load(std::memory_order_acquire)) {
            const std::string text=R27HostSimDiagnosticsJson();
            assert(text.find("HOST_SIM_R27_CPU")!=std::string::npos);
            std::this_thread::yield();
        }
    });
    std::thread submit_a([&]{
        for(auto& item:cases) {
            success(R27CopyAsync(item.middle_a.data(),sizeof(item.middle_a),item.a.data(),sizeof(item.a),producer_a));
            success(R27EventRecord(item.event_a,producer_a));
            item.ready_a.store(true,std::memory_order_release);
        }
    });
    std::thread submit_b([&]{
        for(auto& item:cases) {
            success(R27CopyAsync(item.middle_b.data(),sizeof(item.middle_b),item.b.data(),sizeof(item.b),producer_b));
            success(R27EventRecord(item.event_b,producer_b));
            item.ready_b.store(true,std::memory_order_release);
        }
    });
    for(auto& item:cases) {
        while(!item.ready_a.load(std::memory_order_acquire) || !item.ready_b.load(std::memory_order_acquire)) std::this_thread::yield();
        success(R27EventWait(consumer,item.event_a)); success(R27EventWait(consumer,item.event_b));
        // Wait packets own their captured event generation and source stream.
        success(R27EventDestroy(item.event_a)); success(R27EventDestroy(item.event_b));
        success(R27CopyAsync(item.out_a.data(),sizeof(item.out_a),item.middle_a.data(),sizeof(item.middle_a),consumer));
        success(R27CopyAsync(item.out_b.data(),sizeof(item.out_b),item.middle_b.data(),sizeof(item.middle_b),consumer));
    }
    submit_a.join(); submit_b.join(); success(R27StreamSync(consumer));
    done.store(true,std::memory_order_release); diagnostics.join();
    for(const auto& item:cases) { assert(item.out_a==item.a); assert(item.out_b==item.b); }

    // Controlled negative case: omit the edge and require stale data. This is
    // a test-only pause, never part of model or performance measurements.
    int input=17,middle=-1,output=-2;
    success(R27HostSimSetStreamPaused(producer_a,true));
    success(R27CopyAsync(&middle,sizeof(middle),&input,sizeof(input),producer_a));
    success(R27CopyAsync(&output,sizeof(output),&middle,sizeof(middle),consumer));
    success(R27StreamSync(consumer)); assert(output==-1);
    success(R27HostSimSetStreamPaused(producer_a,false)); success(R27StreamSync(producer_a)); assert(middle==17);

    // Re-record must not retarget a wait that already captured the first event.
    void* event=nullptr; success(R27EventCreate(&event));
    success(R27EventRecord(event,producer_a)); success(R27EventWait(consumer,event));
    success(R27HostSimSetStreamPaused(producer_b,true));
    success(R27EventRecord(event,producer_b));
    int status=-1; success(R27EventQuery(event,&status)); assert(status==0);
    success(R27EventDestroy(event));
    success(R27CopyAsync(&output,sizeof(output),&middle,sizeof(middle),consumer));
    success(R27StreamSync(consumer)); assert(output==17);
    success(R27HostSimSetStreamPaused(producer_b,false)); success(R27StreamSync(producer_b));
    success(R27StreamQuery(consumer,&status)); assert(status==0);
    success(R27StreamDestroy(consumer)); success(R27StreamDestroy(producer_a)); success(R27StreamDestroy(producer_b));
    const std::string final=R27HostSimDiagnosticsJson();
    assert(field(final,"submitted")==field(final,"completed"));
    assert(field(final,"errors")==0 && field(final,"workers")==0 && field(final,"live_events")==0);
    assert(field(final,"live_streams")==1); // inert framework default stream
    std::cout<<"PASS actual hostsim runtime: 1000 rounds, 2 host producers, 3 device workers, concurrent diagnostics\n"<<final<<'\n';
}
