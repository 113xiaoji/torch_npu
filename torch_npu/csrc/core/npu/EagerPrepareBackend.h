#pragma once
namespace c10_npu::eager_prepare {
// A build identity, including off snapshots. Runtime flags cannot turn a real
// SDK build into a simulation build or qualify simulation as real SDK evidence.
inline constexpr bool BuildUsesHostSimulation() noexcept {
#ifdef TORCH_NPU_EAGER_PREPARE_HOST_SIM
    return true;
#else
    return false;
#endif
}
inline constexpr const char* BackendName() noexcept {
    return BuildUsesHostSimulation() ? "host_sim" : "real_sdk";
}
} // namespace c10_npu::eager_prepare
