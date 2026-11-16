#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <time.h>
#endif

#include "torch_npu/csrc/core/npu/NPUEventManager.h"

using Manager = c10_npu::NPUEventManager;
using namespace std::chrono_literals;

namespace {

void Require(bool value, const char* message)
{
    if (!value) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

template <typename T, typename = void>
struct HasHostWait : std::false_type {};

template <typename T>
struct HasHostWait<T, std::void_t<decltype(std::declval<T&>().WaitForEventRecorded(nullptr))>>
    : std::true_type {};

template <typename T>
void Wait(T& manager, aclrtEvent event)
{
    if constexpr (HasHostWait<T>::value) {
        manager.WaitForEventRecorded(event);
    } else {
        Require(false, "NPUEventManager is missing native WaitForEventRecorded");
    }
}

const char* Mode()
{
    using Getter = const char* (*)() noexcept;
#ifdef _WIN32
    auto getter = reinterpret_cast<Getter>(
        GetProcAddress(GetModuleHandleW(nullptr), "torch_npu_event_host_wait_mode"));
#else
    auto getter = reinterpret_cast<Getter>(dlsym(RTLD_DEFAULT, "torch_npu_event_host_wait_mode"));
#endif
    Require(getter != nullptr, "effective host-wait mode getter must be exported");
    return getter();
}

void SetModeEnvironment(const char* value)
{
#ifdef _WIN32
    _putenv_s("TORCH_NPU_EVENT_HOST_WAIT", value);
#else
    setenv("TORCH_NPU_EVENT_HOST_WAIT", value, 1);
#endif
}

double ThreadCpuSeconds()
{
#ifdef _WIN32
    FILETIME creation, exit, kernel, user;
    Require(GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user) != 0,
        "GetThreadTimes must succeed");
    ULARGE_INTEGER k, u;
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;
    u.HighPart = user.dwHighDateTime;
    return static_cast<double>(k.QuadPart + u.QuadPart) * 1e-7;
#else
    timespec value{};
    Require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0,
        "thread CPU clock must be available");
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) * 1e-9;
#endif
}

struct Waiter {
    std::promise<void> entered;
    std::promise<void> done;
    std::future<void> ready = entered.get_future();
    std::future<void> complete = done.get_future();
    std::thread worker;

    Waiter(Manager& manager, aclrtEvent event)
        : worker([this, &manager, event] {
              entered.set_value();
              Wait(manager, event);
              done.set_value();
          })
    {
        Require(ready.wait_for(2s) == std::future_status::ready, "waiter must start");
    }

    void MustRemainPending()
    {
        Require(complete.wait_for(20ms) == std::future_status::timeout,
            "pending record must keep host waiter blocked");
    }

    void Join()
    {
        Require(complete.wait_for(2s) == std::future_status::ready, "waiter must be notified");
        worker.join();
    }
};

void CountAndNotifyBeforeWait(Manager& manager, aclrtEvent event)
{
    Require(manager.IsEventRecorded(event), "absent count means recorded");
    manager.IncreaseUnrecordedCount(event);
    Require(!manager.IsEventRecorded(event), "increment publishes pending record");
    manager.DecreaseUnrecordedCount(event);
    Waiter waiter(manager, event);
    waiter.Join();
    std::cout << "PASS notify_before_wait" << std::endl;
}

void PendingAndMultipleRecords(Manager& manager, aclrtEvent event)
{
    manager.IncreaseUnrecordedCount(event);
    manager.IncreaseUnrecordedCount(event);
    Waiter waiter(manager, event);
    waiter.MustRemainPending();
    // These operations also prove a blocked waiter releases the counter mutex.
    Require(!manager.IsEventRecorded(event), "query must acquire mutex while waiter sleeps");
    manager.DecreaseUnrecordedCount(event);
    Require(!manager.IsEventRecorded(event), "one decrement must retain second pending record");
    waiter.MustRemainPending();
    manager.DecreaseUnrecordedCount(event);
    waiter.Join();
    Require(manager.IsEventRecorded(event), "last decrement erases map entry");
    std::cout << "PASS multiple_pending_records" << std::endl;
}

void ReuseAndIndependentEvents(Manager& manager, aclrtEvent event, aclrtEvent other)
{
    for (int iteration = 0; iteration != 64; ++iteration) {
        manager.IncreaseUnrecordedCount(event);
        Waiter waiter(manager, event);
        manager.DecreaseUnrecordedCount(event);
        waiter.Join();
    }
    manager.IncreaseUnrecordedCount(event);
    manager.IncreaseUnrecordedCount(other);
    Waiter first(manager, event);
    Waiter second(manager, other);
    manager.DecreaseUnrecordedCount(other);
    second.Join();
    first.MustRemainPending();
    // Record again before the old record completes: no completion latch is allowed.
    manager.IncreaseUnrecordedCount(event);
    manager.DecreaseUnrecordedCount(event);
    first.MustRemainPending();
    manager.DecreaseUnrecordedCount(event);
    first.Join();
    std::cout << "PASS rerecord_reuse_and_independent_events" << std::endl;
}

void AllWaitersAndClear(Manager& manager, aclrtEvent event, aclrtEvent other)
{
    manager.IncreaseUnrecordedCount(event);
    std::vector<std::unique_ptr<Waiter>> waiters;
    for (int index = 0; index != 8; ++index) {
        waiters.emplace_back(new Waiter(manager, event));
    }
    manager.DecreaseUnrecordedCount(event);
    for (auto& waiter : waiters) {
        waiter->Join();
    }
    manager.IncreaseUnrecordedCount(event);
    manager.IncreaseUnrecordedCount(other);
    Waiter first(manager, event);
    Waiter second(manager, other);
    first.MustRemainPending();
    second.MustRemainPending();
    // Queue error cleanup clears all pending records; both predicates become true.
    manager.ClearUnrecordedCount();
    first.Join();
    second.Join();
    Require(manager.IsEventRecorded(event) && manager.IsEventRecorded(other),
        "clear must retain absent-means-recorded behavior");
    manager.IncreaseUnrecordedCount(event);
    Waiter reused(manager, event);
    reused.MustRemainPending();
    manager.DecreaseUnrecordedCount(event);
    reused.Join();
    std::cout << "PASS multiple_waiters_and_queue_error_clear" << std::endl;
}

void RollbackAndInvalidDecrement(Manager& manager, aclrtEvent event)
{
    // Existing private prefix-event cleanup rolls back one unpublished record
    // with DecreaseUnrecordedCount. Generic enqueue failure has no such rollback.
    manager.IncreaseUnrecordedCount(event);
    Waiter waiter(manager, event);
    waiter.MustRemainPending();
    manager.DecreaseUnrecordedCount(event);
    waiter.Join();
    bool rejected = false;
    try {
        manager.DecreaseUnrecordedCount(event);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    Require(rejected, "decrement without enqueue must still reject");
    std::cout << "PASS rollback_and_invalid_decrement" << std::endl;
}

void UnpublishedRecordNeedsExistingCleanup(Manager& manager, aclrtEvent event)
{
    // Model Increase followed by a failed/rejected generic queue publication:
    // the baseline has no automatic decrement in that path. Waiting alone must
    // not invent a completion or cancellation result.
    manager.IncreaseUnrecordedCount(event);
    Waiter waiter(manager, event);
    waiter.MustRemainPending();
    Require(!manager.IsEventRecorded(event), "unpublished count must remain pending");
    manager.ClearUnrecordedCount();
    waiter.Join();
    std::cout << "PASS unpublished_record_requires_existing_cleanup" << std::endl;
}

void IdleCpuAndCachedMode(Manager& manager, aclrtEvent event, bool condition)
{
    // Changing the environment after qualification must not change this binary's
    // effective mode or trigger mode parsing on each request.
    const std::string original = Mode();
    SetModeEnvironment(condition ? "spin" : "condition");
    Require(original == Mode(), "effective mode must be cached");
    manager.IncreaseUnrecordedCount(event);
    std::promise<void> started;
    auto ready = started.get_future();
    std::promise<double> cpu;
    auto used = cpu.get_future();
    std::thread worker([&] {
        const double before = ThreadCpuSeconds();
        started.set_value();
        Wait(manager, event);
        cpu.set_value(ThreadCpuSeconds() - before);
    });
    Require(ready.wait_for(2s) == std::future_status::ready, "CPU probe must start");
    std::this_thread::sleep_for(120ms);
    manager.DecreaseUnrecordedCount(event);
    Require(used.wait_for(2s) == std::future_status::ready, "CPU probe must complete");
    const double cpu_seconds = used.get();
    worker.join();
    if (condition) {
        Require(cpu_seconds < 0.04, "condition waiter must release CPU while record is pending");
    }
    std::cout << "PASS cached_mode_and_idle_cpu mode=" << original
              << " waiter_cpu_seconds=" << cpu_seconds << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    Require(argc == 2, "pass expected spin, condition, or invalid mode");
    Require(HasHostWait<Manager>::value, "NPUEventManager is missing native WaitForEventRecorded");
    const std::string expected = argv[1];
    Require(expected == Mode(), "effective mode must match process configuration");
    if (expected == "invalid") {
        bool rejected = false;
        try {
            Manager::GetInstance();
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        Require(rejected, "invalid host-wait mode must be rejected during manager initialization");
        std::cout << "PASS invalid_mode_rejected" << std::endl;
        return 0;
    }
    auto& manager = Manager::GetInstance();
    int first = 0, second = 0;
    aclrtEvent event = &first;
    aclrtEvent other = &second;
    CountAndNotifyBeforeWait(manager, event);
    PendingAndMultipleRecords(manager, event);
    ReuseAndIndependentEvents(manager, event, other);
    AllWaitersAndClear(manager, event, other);
    RollbackAndInvalidDecrement(manager, event);
    UnpublishedRecordNeedsExistingCleanup(manager, event);
    IdleCpuAndCachedMode(manager, event, expected == "condition");
    // Native waiters have joined before singleton destruction, as required by
    // existing manager lifetime. This test does not claim shutdown cancellation.
    manager.ClearUnrecordedCount();
    std::cout << "PASS joined_waiters_before_manager_shutdown" << std::endl;
}
