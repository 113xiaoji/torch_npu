#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <c10/core/thread_pool.h>
#include <c10/util/flat_hash_map.h>
#include <third_party/acl/inc/acl/acl.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

#define ACL_EVENT_DEFAULT 0x0000000Eu

namespace c10_npu {

// Cached process configuration for qualification before a measured region.
// Returns "spin", "condition", or "invalid" without throwing across the C ABI.
extern "C" TORCH_NPU_API const char* torch_npu_event_host_wait_mode() noexcept;

class NPUEventManager {
public:
    static NPUEventManager& GetInstance();
    aclError QueryAndDestroyEvent();
    aclError LazyDestroy(aclrtEvent npu_event);
    void ClearEvent();
    void IncreaseUnrecordedCount(aclrtEvent event);
    void DecreaseUnrecordedCount(aclrtEvent event);
    bool IsEventRecorded(aclrtEvent event);
    void WaitForEventRecorded(aclrtEvent event);
    void IncreaseUnwaitedCount(aclrtEvent event);
    void DecreaseUnwaitedCount(aclrtEvent event);
    bool IsEventWaited(aclrtEvent event);
    void ClearUnrecordedCount();
    void AddIpcEvent(aclrtEvent event);
    ~NPUEventManager() {}

private:
    void run(aclrtEvent event);
    void RemoveIpcEvent(aclrtEvent event);
    bool IsIpcEvent(aclrtEvent event);

private:
    std::mutex event_queue_mutex_;
    NPUEventManager();
    std::deque<aclrtEvent> npu_events_;
    std::shared_ptr<c10::TaskThreadPool> thread_pool_;

    std::mutex event_unrecorded_count_mutex_;
    ska::flat_hash_map<aclrtEvent, int> event_unrecorded_count_;
    std::mutex event_unwaited_count_mutex_;
    ska::flat_hash_map<aclrtEvent, int> event_unwaited_count_;

    std::mutex ipc_event_mutex_;
    std::unordered_set<aclrtEvent> ipc_events_;
    std::condition_variable event_unrecorded_count_cv_;
};

} // namespace c10_npu
