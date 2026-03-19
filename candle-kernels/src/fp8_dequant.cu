#include "cuda_utils.cuh"
#include <stdint.h>

#if __CUDA_ARCH__ >= 800

// Blockwise FP8 E4M3 -> BF16 dequantization.
//
// weight:    [rows, cols] in fp8_e4m3
// scale_inv: [rows/block_r, cols/block_c] in float32
// output:    [rows, cols] in bf16
//
// For each element (i,j):  output[i][j] = (float)weight[i][j] * scale_inv[i/block_r][j/block_c]
//
// The kernel is launched with total threads = rows * cols.
extern "C" __global__ void fp8_blockwise_dequant_bf16(
    const __nv_fp8_e4m3 *weight,
    const float *scale_inv,
    __nv_bfloat16 *output,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t block_r,
    const uint32_t block_c
) {
    const uint32_t scale_cols = (cols + block_c - 1) / block_c;

    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < rows * cols;
         idx += blockDim.x * gridDim.x)
    {
        uint32_t r = idx / cols;
        uint32_t c = idx % cols;

        float w = __half2float(__nv_cvt_fp8_to_halfraw(weight[idx].__x, __NV_E4M3));
        float s = scale_inv[(r / block_r) * scale_cols + (c / block_c)];

        output[idx] = __float2bfloat16(w * s);
    }
}

// Same but output in f16
extern "C" __global__ void fp8_blockwise_dequant_f16(
    const __nv_fp8_e4m3 *weight,
    const float *scale_inv,
    __half *output,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t block_r,
    const uint32_t block_c
) {
    const uint32_t scale_cols = (cols + block_c - 1) / block_c;

    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < rows * cols;
         idx += blockDim.x * gridDim.x)
    {
        uint32_t r = idx / cols;
        uint32_t c = idx % cols;

        float w = __half2float(__nv_cvt_fp8_to_halfraw(weight[idx].__x, __NV_E4M3));
        float s = scale_inv[(r / block_r) * scale_cols + (c / block_c)];

        output[idx] = __float2half(w * s);
    }
}

// Same but output in f32
extern "C" __global__ void fp8_blockwise_dequant_f32(
    const __nv_fp8_e4m3 *weight,
    const float *scale_inv,
    float *output,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t block_r,
    const uint32_t block_c
) {
    const uint32_t scale_cols = (cols + block_c - 1) / block_c;

    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < rows * cols;
         idx += blockDim.x * gridDim.x)
    {
        uint32_t r = idx / cols;
        uint32_t c = idx % cols;

        float w = __half2float(__nv_cvt_fp8_to_halfraw(weight[idx].__x, __NV_E4M3));
        float s = scale_inv[(r / block_r) * scale_cols + (c / block_c)];

        output[idx] = w * s;
    }
}

#endif
