#pragma once
#include <cstdint>
namespace c10_npu::eager_prepare {
struct EmbeddingBagDomain {
    bool native_storage,fp32_weight,int64_indices_offsets,ranks,nonempty,outputs_disjoint;
    bool scale_grad_by_freq;
    int64_t mode;
    bool sparse,per_sample_weights,include_last_offset,negative_padding;
    bool Admitted() const noexcept {
        return native_storage && fp32_weight && int64_indices_offsets && ranks && nonempty && outputs_disjoint &&
            !scale_grad_by_freq && mode==0 && sparse && !per_sample_weights && !include_last_offset && negative_padding;
    }
};
} // namespace c10_npu::eager_prepare
