#pragma once
#include <cstdint>
namespace c10_npu::prefix {
struct H2DPrewaitGuard final {
    bool bound_owner = false;
    bool default_stream = false;
    bool same_owner_stream_device = false;
    bool task_queue = false;
    bool shared_queue = false;
    bool running_generation = false;
    bool resume_mode = false;
    int32_t stream_timeout = -1;
    bool sync_debug = false;
    bool trace_active = false;
    bool sync_launch = false;
    bool reentrant = false;
};
inline bool AllowsH2DPrewait(const H2DPrewaitGuard& g) noexcept
{
    return g.bound_owner && g.default_stream && g.same_owner_stream_device &&
        g.task_queue && g.shared_queue && g.running_generation &&
        !g.resume_mode && g.stream_timeout == -1 && !g.sync_debug &&
        !g.trace_active && !g.sync_launch && !g.reentrant;
}
} // namespace c10_npu::prefix
