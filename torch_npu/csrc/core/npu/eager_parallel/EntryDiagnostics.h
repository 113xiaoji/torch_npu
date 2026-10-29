#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace c10_npu::eager_parallel {
// Shadow bookkeeping only. No device completion or allocation lease is inferred.
class EntryDiagnostics {
public:
    struct Counts { uint64_t entered=0, completed=0, failed=0, inputs=0, outputs=0; };
    struct Snapshot {
        bool enabled=false, closed=true, coverage_incomplete=false;
        uint64_t entered=0, completed=0, failed=0, nested=0, active=0, foreign_calls=0;
        uint64_t capacity=64, byte_limit=536870912;
        std::string first_error;
        std::map<std::string,Counts> operators;
        std::map<std::string,uint64_t> boundaries;
    };
    bool enabled() const noexcept { return enabled_.load(std::memory_order_acquire); }
    void configure(bool enabled,uint64_t capacity,uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        if (enabled_.load() || !stack_.empty()) throw std::logic_error("close the native shadow domain first");
        if (!capacity || capacity>1024 || !bytes || bytes>536870912)
            throw std::invalid_argument("native shadow bounds exceeded");
        view_=Snapshot{};view_.capacity=capacity;view_.byte_limit=bytes;
        view_.enabled=enabled;view_.closed=!enabled;owner_=std::this_thread::get_id();
        bookkeeping_failed_.store(false);
        enabled_.store(enabled,std::memory_order_release);
    }
    uint64_t enter(const char* schema,const char* route,uint64_t inputs) noexcept {
        if (!enabled()) return 0;
        try {
        std::lock_guard<std::mutex> lock(mu_);
        if (!enabled_.load()) return 0;
        if (std::this_thread::get_id()!=owner_) {
            ++view_.foreign_calls;view_.coverage_incomplete=true;return 0;
        }
        if (!schema || !route || stack_.size()>=64) {
            view_.coverage_incomplete=true;return 0;
        }
        std::string key(schema);
        if (key.size()>256 || (view_.operators.size()>=512 && !view_.operators.count(key))) {
            view_.coverage_incomplete=true;return 0;
        }
        auto& counts=view_.operators[key];
        const bool outer=stack_.empty();const uint64_t token=++sequence_;
        stack_.push_back(Frame{token,std::move(key),outer});
        if (outer) { ++view_.entered;++view_.active;++counts.entered;counts.inputs+=inputs; }
        else ++view_.nested;
        return token;
        } catch (...) { bookkeeping_failed_.store(true,std::memory_order_release);return 0; }
    }
    void complete(uint64_t token,uint64_t outputs) noexcept { finish(token,outputs,nullptr); }
    void fail(uint64_t token,const char* error) noexcept { finish(token,0,error ? error : "native call failed"); }
    void mark_incomplete() noexcept { bookkeeping_failed_.store(true,std::memory_order_release); }
    void boundary(const char* reason) noexcept {
        if (!enabled()) return;
        try {
        std::lock_guard<std::mutex> lock(mu_);
        if (!enabled_.load()) return;
        if (!reason || std::string(reason).size()>256 ||
            (view_.boundaries.size()>=64 && !view_.boundaries.count(reason))) {
            view_.coverage_incomplete=true;return;
        }
        if (std::this_thread::get_id()!=owner_) { ++view_.foreign_calls;view_.coverage_incomplete=true; }
        ++view_.boundaries[reason];
        } catch (...) { bookkeeping_failed_.store(true,std::memory_order_release); }
    }
    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!enabled_.load()) return;
        if (std::this_thread::get_id()!=owner_ || !stack_.empty())
            throw std::logic_error("close requires the idle shadow owner");
        enabled_.store(false,std::memory_order_release);view_.enabled=false;view_.closed=true;
    }
    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);auto copy=view_;
        copy.coverage_incomplete=copy.coverage_incomplete || bookkeeping_failed_.load();
        return copy;
    }
private:
    struct Frame { uint64_t token;std::string name;bool outer; };
    void finish(uint64_t token,uint64_t outputs,const char* error) noexcept {
        if (!token) return;
        try {
            std::lock_guard<std::mutex> lock(mu_);
            if (std::this_thread::get_id()!=owner_ || stack_.empty() || stack_.back().token!=token) {
                view_.coverage_incomplete=true;return;
            }
            auto frame=std::move(stack_.back());stack_.pop_back();
            if (frame.outer) {
                --view_.active;auto& counts=view_.operators.at(frame.name);
                if (error) { ++view_.failed;++counts.failed; }
                else { ++view_.completed;++counts.completed;counts.outputs+=outputs; }
            }
            if (error && view_.first_error.empty()) view_.first_error=std::string(error).substr(0,1024);
        } catch (...) { bookkeeping_failed_.store(true,std::memory_order_release); }
    }
    mutable std::mutex mu_;
    std::atomic<bool> enabled_{false},bookkeeping_failed_{false};
    std::thread::id owner_;
    uint64_t sequence_=0;
    Snapshot view_;
    std::vector<Frame> stack_;
};
} // namespace c10_npu::eager_parallel
