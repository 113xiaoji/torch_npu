#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>

namespace c10_npu::host_regions {

// Only the pointer-array backing store differs. ACL objects retain their
// original destruction order and API calls for both allocation policies.
enum class ParameterArrayOwnership { Heap, PacketArena };

namespace parameter_detail {
template<class Parameters, class InputDesc, class InputBuffer, class OutputDesc, class OutputBuffer,
    class Allocate>
inline void ExportArrays(Parameters& params, const InputDesc& inputs, const InputBuffer& input_buffers,
    const OutputDesc& outputs, const OutputBuffer& output_buffers, Allocate&& allocate)
{
    const size_t input_count = inputs.size(), output_count = outputs.size();
    if (input_count != input_buffers.size() || output_count != output_buffers.size() ||
        input_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        output_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        input_count > std::numeric_limits<size_t>::max() / (2 * sizeof(void*)) ||
        output_count > std::numeric_limits<size_t>::max() / (2 * sizeof(void*)) - input_count)
        throw std::length_error("Captured ACL parameter arrays are too large");
    const size_t input_bytes = input_count * sizeof(void*);
    const size_t output_bytes = output_count * sizeof(void*);
    const size_t total_bytes = 2 * (input_bytes + output_bytes);
    auto* base = static_cast<unsigned char*>(allocate(total_bytes ? total_bytes : 1, alignof(void*)));
    if (!base) throw std::bad_alloc();
    auto input_desc = reinterpret_cast<decltype(params.input_desc)>(base);
    auto input_data = reinterpret_cast<decltype(params.input_data_buf)>(base + input_bytes);
    auto output_desc = reinterpret_cast<decltype(params.output_desc)>(base + 2 * input_bytes);
    auto output_data = reinterpret_cast<decltype(params.output_data_buf)>(base + 2 * input_bytes + output_bytes);
    std::copy(inputs.begin(), inputs.end(), input_desc);
    std::copy(input_buffers.begin(), input_buffers.end(), input_data);
    std::copy(outputs.begin(), outputs.end(), output_desc);
    std::copy(output_buffers.begin(), output_buffers.end(), output_data);
    // Allocation and all nonthrowing pointer copies finish before ownership is
    // published into the destination. A failed arena allocation leaves it empty.
    params.input_num = static_cast<int>(input_count);
    params.output_num = static_cast<int>(output_count);
    params.input_desc = input_desc;
    params.input_data_buf = input_data;
    params.output_desc = output_desc;
    params.output_data_buf = output_data;
}

template<class Parameters, class DestroyDesc, class DestroyBuffer, class Free>
inline void DestroyArrays(Parameters& params, ParameterArrayOwnership ownership,
    DestroyDesc&& destroy_desc, DestroyBuffer&& destroy_buffer, Free&& free_array)
{
    if (params.input_num != 0) {
        if (params.input_desc != nullptr)
            for (int i = 0; i < params.input_num; ++i) destroy_desc(params.input_desc[i]);
        if (params.input_data_buf != nullptr)
            for (int i = 0; i < params.input_num; ++i) destroy_buffer(params.input_data_buf[i]);
        params.input_num = 0;
    }
    if (params.output_num != 0) {
        if (params.output_desc != nullptr)
            for (int i = 0; i < params.output_num; ++i) destroy_desc(params.output_desc[i]);
        if (params.output_data_buf != nullptr)
            for (int i = 0; i < params.output_num; ++i) destroy_buffer(params.output_data_buf[i]);
        params.output_num = 0;
    }
    if (ownership == ParameterArrayOwnership::Heap) free_array(params.input_desc);
    params.input_desc = nullptr;
    params.input_data_buf = nullptr;
    params.output_desc = nullptr;
    params.output_data_buf = nullptr;
}
} // namespace parameter_detail
} // namespace c10_npu::host_regions
