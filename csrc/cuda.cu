#include <torch/types.h>
#include <cuda.h>
#include <bit>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

template<typename T>
__global__ void pe_kernel(const T* data, const uint32_t data_len, const uint64_t* vec, const uint32_t vec_len, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float shift_f = std::bit_cast<float>(0x30000000);
    const uint32_t out_len = data_len * vec_len * 2;

    __shared__ T _data[32];
    __shared__ uint64_t _vec[256];
    for (int i = threadIdx.x; i < vec_len; i+=32){
        _vec[i] = vec[i];
    }

    _data[threadIdx.x] = data[idx];

    uint32_t vec_pos = threadIdx.x;
    uint32_t data_pos = 0;

    uint32_t out_pos = blockIdx.x * blockDim.x * vec_len * 2 + threadIdx.x;

    __shared__ float o[2][33];

    for(int i = 0; i < vec_len; i++){
        uint32_t enc = (_data[data_pos] * _vec[vec_pos]) >> 32;
        int32_t code = std::bit_cast<int32_t>(enc);

        float code_f = code * shift_f;

        float s;
        float c;
        sincospif(code_f, &s, &c);
        o[0][threadIdx.x] = s;
        o[1][threadIdx.x] = c;

        float o1 = o[threadIdx.x % 2][threadIdx.x / 2];
        float o2 = o[threadIdx.x % 2][threadIdx.x / 2 + 16];
        if (out_pos < out_len){
            out[out_pos] = o1;
        }
        out_pos += 32;
        if (out_pos < out_len){
            out[out_pos] = o2;
        }
        out_pos += 32;
        vec_pos += 32;
        if (vec_pos >= vec_len){
            vec_pos -= vec_len;
            ++data_pos;
        }
    }
}

torch::Tensor pe(torch::Tensor input, torch::Tensor vec) {
    constexpr uint32_t width = 64;
    constexpr uint32_t threads = 32;
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCUDA, input.device().index())
        .requires_grad(false);
    
    auto out_size = input.sizes().vec();
    out_size.push_back(vec.size(0) * 2);
    const auto blocks = (torch::numel(input) + threads - 1) / threads;

    auto result = torch::empty(torch::IntArrayRef({out_size}), options);

    dim3 threads_per_block(threads);
    dim3 number_of_blocks(blocks);

    auto type = input.scalar_type();

    if (type == torch::kUInt64){
        pe_kernel<<<number_of_blocks, threads_per_block>>>(
            input.data_ptr<uint64_t>(), torch::numel(input), vec.data_ptr<uint64_t>(), torch::numel(vec), result.data_ptr<float>()
        );
    } else if (type == torch::kUInt32)
    {
        pe_kernel<<<number_of_blocks, threads_per_block>>>(
            input.data_ptr<uint32_t>(), torch::numel(input), vec.data_ptr<uint64_t>(), torch::numel(vec), result.data_ptr<float>()
        );
    } else {
        throw std::runtime_error("Unsupported type");
    }
    
    return result;
}