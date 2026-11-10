#include "torch_npu/csrc/core/npu/eager_parallel/ImmediateProtocol.h"
#include <cassert>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace c10_npu::eager_parallel;
struct FakeBackend : ImmediateEventBackend {
    std::vector<std::string> log;
    bool fail_record = false, fail_wait = false, fail_sync = false;
    int fail_sync_stream = -1;
    std::function<void()> record_callback;
    void record(std::size_t event, unsigned stream) override {
        if (fail_record) throw std::runtime_error("record failed");
        log.push_back("r" + std::to_string(event) + ":" + std::to_string(stream));
        if (record_callback) record_callback();
    }
    void wait(unsigned stream, std::size_t event) override {
        if (fail_wait) throw std::runtime_error("wait failed");
        log.push_back("w" + std::to_string(stream) + ":" + std::to_string(event));
    }
    void synchronize(unsigned stream) override {
        log.push_back("s" + std::to_string(stream));
        if (fail_sync || static_cast<int>(stream)==fail_sync_stream) throw std::runtime_error("sync failed");
    }
};
bool contains(const FakeBackend& b, const std::string& x) {
    for (const auto& item : b.log) if (item == x) return true;
    return false;
}
int main() {
    {
        FakeBackend b; ImmediateProtocol p(b, 8, 32);
        std::uintptr_t weight = 9, a = 10, c = 11, cat = 12;
        assert(p.begin(true, &weight, 1) == 1); p.commit(&a, 1);
        assert(p.begin(true, &weight, 1) == 2); p.commit(&c, 1);
        assert(!contains(b, "w2:1")); // shared read-only weight is independent
        std::uintptr_t reads[] = {a, c};
        assert(p.begin(false, reads, 2) == 3); p.commit(&cat, 1);
        assert(contains(b, "w3:1") && contains(b, "w3:2")); // real fan-in
        assert(p.begin(true, &cat, 1) == 1); p.commit(&weight, 1);
        assert(contains(b, "w1:3")); // consumer -> later producer dependency
        p.drain();
        assert(contains(b, "w0:4") && contains(b, "w0:2") && contains(b, "w0:3"));
        assert(contains(b, "s0") && p.calls() == 0);
        b.log.clear();
        assert(p.begin(true, &weight, 1) == 1); p.commit(&a, 1);
        assert(contains(b, "r0:0") && contains(b, "w1:0")); // refreshed baseline frontier
        p.drain();
    }
    {
        FakeBackend b; ImmediateProtocol p(b, 1, 4);
        std::uintptr_t key = 1;
        p.begin(true, nullptr, 0); p.commit(&key, 1);
        bool rejected = false;
        try { p.begin(true, &key, 1); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected); p.drain();
    }
    {
        FakeBackend b; ImmediateProtocol p(b, 4, 8);
        p.begin(true, nullptr, 0); b.fail_record = true;
        bool rejected = false;
        try { p.commit(nullptr, 0); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && p.poisoned());
        b.fail_sync = true;
        assert(!p.recover()); // completion not established: owners must remain quarantined
        b.fail_sync = false;
        assert(p.recover() && p.poisoned()); // recovery never silently resumes admission
        assert(contains(b, "s1") && contains(b, "s2") && contains(b, "s3"));
    }
    {
        FakeBackend b; ImmediateProtocol p(b, 4, 8);
        b.fail_wait = true;
        bool rejected = false;
        try { p.begin(true, nullptr, 0); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && p.poisoned());
        b.fail_wait = false; assert(p.recover());
    }
    {
        FakeBackend b; ImmediateProtocol p(b, 4, 8);
        b.fail_record=true;
        bool rejected=false;
        try { p.begin(true,nullptr,0); } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && p.active() && p.poisoned());
        b.fail_record=false; b.fail_sync_stream=1;
        assert(!p.recover());
        // One failed stream must not prevent checking every remaining stream.
        assert(contains(b,"s0") && contains(b,"s1") && contains(b,"s2") && contains(b,"s3"));
        b.fail_sync_stream=-1; assert(p.recover());
    }
    {
        FakeBackend b; ImmediateProtocol p(b, 4, 8);
        // A callback may swallow its own reentry exception. Its durable poison
        // receipt must still stop the outer backend call from admitting a body.
        b.record_callback=[&]{p.poison();};
        bool rejected=false;
        try { p.begin(true,nullptr,0); } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && p.poisoned() && !contains(b,"w1:0"));
        assert(p.recover());
    }
    {
        FakeBackend b; ImmediateProtocol p(b, 4, 8);
        std::uintptr_t key=5;
        p.begin(true,nullptr,0); p.commit(&key,1);
        b.fail_wait=true;
        bool rejected=false;
        try { p.drain(); } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected && p.active() && p.calls()==1 && p.poisoned());
        b.fail_wait=false; assert(p.recover());
    }


    {
        using C=EmbeddingLengthClass;
        for (int64_t length: {63,64,65}) {
            assert(ClassifyEmbeddingLength(2,1,1,128*length,128,false,64)==(length<64?C::Small:C::Large));
            assert(ClassifyEmbeddingLength(2,1,1,128*length,129,true,64)==(length<64?C::Small:C::Large));
        }
        assert(ClassifyEmbeddingLength(2,1,1,8191,128,false,64)==C::Small); // ragged mean
        assert(ClassifyEmbeddingLength(2,1,1,0,1,false,64)==C::Small);
        assert(ClassifyEmbeddingLength(2,1,1,0,0,false,64)==C::Uncertain);
        assert(ClassifyEmbeddingLength(2,1,1,0,1,true,64)==C::Uncertain);
        assert(ClassifyEmbeddingLength(2,2,1,64,1,false,64)==C::Uncertain);
        assert(ClassifyEmbeddingLength(2,1,1,-1,1,false,64)==C::Uncertain);
        assert(ClassifyEmbeddingLength(2,1,1,INT64_MAX,INT64_MAX,false,64)==C::Small); // no product overflow
        assert(ClassifyEmbeddingLength(2,1,1,INT64_MAX,1,false,64)==C::Large);
        assert(StorageIntervalsOverlap({100,32,true},{120,32,true}));
        assert(!StorageIntervalsOverlap({100,20,true},{120,32,true}));
        assert(StorageIntervalsOverlap({0,0,false},{120,32,true}));
        assert(!MakeStorageInterval(UINTPTR_MAX-2,4).valid);
    }
    {
        FakeBackend b; ImmediateProtocol p(b,4,16,true);
        std::uintptr_t a=10,c=11;
        assert(p.begin(true,nullptr,0)==1); p.commit(&a,1);
        p.mark_caller_dirty(); // independent original-caller small body
        assert(p.begin(false,&a,1)==3); p.commit(&c,1);
        assert(contains(b,"r2:0") && contains(b,"w3:2") && contains(b,"w3:1"));
        p.mark_caller_dirty();
        assert(p.begin(true,&c,1)==2); p.commit(&a,1);
        assert(contains(b,"r4:0") && contains(b,"w2:4") && contains(b,"w2:3"));
        assert(p.frontier_records()==3 && p.frontier_waits()==3);
        p.mark_caller_dirty();
        assert(p.begin(true,nullptr,0)==1); p.commit(&a,1);
        assert(contains(b,"r6:0") && contains(b,"w1:6") && contains(b,"r7:1"));
        assert(!p.can_begin());
        p.drain();
        b.log.clear();
        p.mark_caller_dirty(); p.drain(); // all-small normal fences retain caller ordering
        assert(b.log.empty() && p.caller_dirty());
        p.drain(true); assert(contains(b,"s0") && !p.caller_dirty());
        b.log.clear();
        p.mark_caller_dirty();
        assert(p.begin(true,nullptr,0)==1); p.commit(&a,1);
        assert(contains(b,"r0:0") && contains(b,"r1:1"));
        p.drain();
    }
    {
        FakeBackend b; ImmediateProtocol p(b,1,4,true);
        p.mark_caller_dirty(); b.fail_sync=true;
        bool failed=false; try {p.drain(true);} catch(...) {failed=true;}
        assert(failed && p.poisoned() && p.caller_dirty());
        b.fail_sync=false; assert(p.recover() && !p.caller_dirty());
        assert(contains(b,"s0") && contains(b,"s1") && contains(b,"s2") && contains(b,"s3"));
    }
    std::cout << "PASS R27 immediate protocol\n";
}
