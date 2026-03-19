#include "cuda_utils.cuh"
#include <stdint.h>

#if __CUDA_ARCH__ >= 800

// Blockwise dynamic quantization: BF16 -> FP8 E4M3 with 128x128 block scales.
//
// Divides the [m, k] activation matrix into 128x128 sub-blocks. For each block,
// computes absmax and scale = absmax / 448.0, then quantizes all elements.
//
// The scale output is laid out in column-major order matching what cuBLASLt
// expects for BLK128x128 scaling on B(OP_N) stored as col-major (k, m):
//   scale[k_block + m_block * num_k_blocks]
//
// input:  [m, k] bf16, row-major
// output: [m, k] fp8 e4m3, row-major
// scales: [ceil(k/128) * ceil(m/128)] f32, col-major (k_blocks × m_blocks)
//
// Launch with grid = (ceil(k/128), ceil(m/128)), block = (128,)

extern "C" __global__ void bf16_to_fp8_e4m3_blockwise(
    const __nv_bfloat16 *input,
    __nv_fp8_e4m3 *output,
    float *scales,
    const uint32_t m,
    const uint32_t k
) {
    const uint32_t BSIZE = 128;
    const uint32_t k_block = blockIdx.x;
    const uint32_t m_block = blockIdx.y;
    const uint32_t tid = threadIdx.x;

    const uint32_t k_start = k_block * BSIZE;
    const uint32_t m_start = m_block * BSIZE;
    const uint32_t col = k_start + tid;
    const bool valid = col < k;

    const uint32_t m_end = min(m_start + BSIZE, m);
    const uint32_t m_extent = m_end - m_start;

    // Phase 1: each thread scans its column, finding local absmax
    float local_max = 0.0f;
    if (valid) {
        for (uint32_t i = 0; i < m_extent; i++) {
            float val = __bfloat162float(input[(m_start + i) * k + col]);
            float a = fabsf(val);
            if (a > local_max) local_max = a;
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }

    // Cross-warp reduction via shared memory (128 threads = 4 warps)
    __shared__ float warp_max[4];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();

    __shared__ float block_scale;
    __shared__ float block_inv_scale;
    if (tid == 0) {
        float bmax = fmaxf(fmaxf(warp_max[0], warp_max[1]),
                           fmaxf(warp_max[2], warp_max[3]));
        const float FP8_E4M3_MAX = 448.0f;
        float s = (bmax > 0.0f) ? (bmax / FP8_E4M3_MAX) : 1.0f;
        block_scale = s;
        block_inv_scale = 1.0f / s;

        uint32_t num_k_blocks = (k + BSIZE - 1) / BSIZE;
        scales[k_block + m_block * num_k_blocks] = s;
    }
    __syncthreads();

    // Phase 2: quantize using the block scale
    if (valid) {
        const float FP8_E4M3_MAX = 448.0f;
        float inv_s = block_inv_scale;
        for (uint32_t i = 0; i < m_extent; i++) {
            uint64_t idx = (uint64_t)(m_start + i) * k + col;
            float val = __bfloat162float(input[idx]) * inv_s;
            val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            __half h = __float2half(val);
            __nv_fp8_e4m3 fp8;
            fp8.__x = __nv_cvt_halfraw_to_fp8(*((__half_raw*)&h), __NV_SATFINITE, __NV_E4M3);
            output[idx] = fp8;
        }
    }
}

// Per-tensor scalar quantization: BF16 -> FP8 E4M3.
//
// Computes absmax over the entire tensor, derives scale = absmax / 448,
// then quantizes all elements.
//
// input:  [numel] bf16
// output: [numel] fp8 e4m3
// scale:  [1] f32 (output: the computed scale)
//
// Launch with grid = (ceil(numel/256),), block = (256,)
// Requires two passes via atomicMax for global reduction.
// We use a two-kernel approach: first compute absmax, then quantize.

extern "C" __global__ void fp8_zero_scale(float *buf) {
    buf[0] = 0.0f;
}

extern "C" __global__ void bf16_absmax(
    const __nv_bfloat16 *input,
    float *absmax_out,
    const uint32_t numel
) {
    __shared__ float smem[8];
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + tid;

    float local_max = 0.0f;
    for (uint32_t i = gid; i < numel; i += gridDim.x * blockDim.x) {
        float val = fabsf(__bfloat162float(input[i]));
        if (val > local_max) local_max = val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }

    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (tid == 0) {
        float bmax = 0.0f;
        for (int i = 0; i < (int)(blockDim.x / 32); i++) {
            if (smem[i] > bmax) bmax = smem[i];
        }
        atomicMax((int*)absmax_out, __float_as_int(bmax));
    }
}

// Compute scale from absmax: scale = max(absmax / 448.0, 1e-12)
// absmax_buf must be pre-filled by bf16_absmax. Scale is written in-place.
extern "C" __global__ void fp8_compute_scale(float *absmax_buf) {
    const float FP8_E4M3_MAX = 448.0f;
    float am = *absmax_buf;
    *absmax_buf = (am > 0.0f) ? (am / FP8_E4M3_MAX) : 1.0f;
}

extern "C" __global__ void bf16_to_fp8_e4m3_scalar(
    const __nv_bfloat16 *input,
    __nv_fp8_e4m3 *output,
    const float *scale,
    const uint32_t numel
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;

    const float FP8_E4M3_MAX = 448.0f;
    float inv_s = 1.0f / (*scale);
    float val = __bfloat162float(input[gid]) * inv_s;
    val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    __half h = __float2half(val);
    __nv_fp8_e4m3 fp8;
    fp8.__x = __nv_cvt_halfraw_to_fp8(*((__half_raw*)&h), __NV_SATFINITE, __NV_E4M3);
    output[gid] = fp8;
}

#endif
