#pragma once
#include <Python.h>
#include <atomic>
#include <exception>
#include <mutex>
#include <utility>
#include <condition_variable>
#include <cstdio>
#include <thread>
#include <chrono>

namespace c10_npu::eager_prepare {
class InterpreterDetach final {
public:
    InterpreterDetach() noexcept
        : saved_(PyThreadState_GetUnchecked()!=nullptr ? PyEval_SaveThread() : nullptr) {}
    ~InterpreterDetach() noexcept { if (saved_) PyEval_RestoreThread(saved_); }
    InterpreterDetach(const InterpreterDetach&)=delete;
    InterpreterDetach& operator=(const InterpreterDetach&)=delete;
private:
    PyThreadState* saved_;
};

class LifecycleGuard final {
public:
    explicit LifecycleGuard(std::mutex& mutex) : lock_(mutex) {}
    LifecycleGuard(const LifecycleGuard&)=delete;
    LifecycleGuard& operator=(const LifecycleGuard&)=delete;
private:
    // Construction detaches BEFORE lock contention. Reverse destruction
    // releases the lifecycle lock BEFORE any interpreter/GIL reattachment.
    InterpreterDetach detach_;
    std::unique_lock<std::mutex> lock_;
};

class ColdCloseOnce final {
public:
    // Caller serializes Run/Reset with LifecycleGuard. Atomic status fields
    // may be read by a nonlinearizable diagnostic snapshot without that lock.
    template<class Work> void Run(Work&& work) {
        if (attempted_.load(std::memory_order_acquire)) {
            if (error_) std::rethrow_exception(error_);
            return;
        }
        attempted_.store(true,std::memory_order_release);
        try { std::forward<Work>(work)(); passed_.store(true,std::memory_order_release); }
        catch (...) { error_=std::current_exception(); throw; }
    }
    void Reset() noexcept { error_=nullptr; passed_=false; attempted_=false; }
    bool Attempted() const noexcept { return attempted_.load(std::memory_order_acquire); }
    bool Passed() const noexcept { return passed_.load(std::memory_order_acquire); }
private:
    std::atomic<bool> attempted_{false},passed_{false};
    std::exception_ptr error_;
};

[[noreturn]] inline void FailStopFinalize(std::mutex& mutex,std::condition_variable& changed) noexcept {
    std::fputs("R24_EAGER_PREPARE_FINALIZE_FAILSTOP\n",stderr); std::fflush(stderr);
    InterpreterDetach detached;
    // All synchronization objects were constructed before accepting SDK work.
    // Neither signal/diagnostic errors nor a spurious wake acknowledge cleanup.
    for (;;) {
        try { std::unique_lock<std::mutex> lock(mutex); changed.wait(lock); }
        catch (...) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
    }
}
} // namespace c10_npu::eager_prepare
