#pragma once
#include <cstdint>
namespace c10_npu::eager_prepare {
// Constant space, no allocations or timestamps. Caller serializes updates.
// Hash is an order receipt; the exact sequential check also detects gaps.
struct SequenceOrderEvidence {
    uint64_t count=0, hash=14695981039346656037ULL;
    bool ordered=true;
    void Record(uint64_t sequence,const char* operation) noexcept {
        ordered=ordered && sequence==count;
        ++count;
        auto byte=[this](unsigned char value) { hash=(hash^value)*1099511628211ULL; };
        for (unsigned i=0;i<8;++i) byte(static_cast<unsigned char>(sequence>>(i*8)));
        while (*operation) byte(static_cast<unsigned char>(*operation++));
        byte(0);
    }
};
} // namespace c10_npu::eager_prepare
