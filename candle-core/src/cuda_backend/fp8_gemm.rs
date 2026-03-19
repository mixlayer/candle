use super::device::CudaDevice;
use super::error::WrapErr;
use crate::Result;
use cudarc::cublas::sys as cublas_sys;
use cudarc::cublaslt::{result as lt, sys as lt_sys, MatmulShared};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::ffi::c_void;

struct MatmulDescGuard(lt_sys::cublasLtMatmulDesc_t);
impl Drop for MatmulDescGuard {
    fn drop(&mut self) {
        unsafe { let _ = lt::destroy_matmul_desc(self.0); }
    }
}

struct MatrixLayoutGuard(lt_sys::cublasLtMatrixLayout_t);
impl Drop for MatrixLayoutGuard {
    fn drop(&mut self) {
        unsafe { let _ = lt::destroy_matrix_layout(self.0); }
    }
}

struct MatmulPrefGuard(lt_sys::cublasLtMatmulPreference_t);
impl Drop for MatmulPrefGuard {
    fn drop(&mut self) {
        unsafe { let _ = lt::destroy_matmul_pref(self.0); }
    }
}

/// Perform an FP8 GEMM via cuBLASLt with blockwise scaling on BOTH operands.
///
/// Computes `D = (act_scale · act_fp8) @ (w_scale · w_fp8)^T` where:
///
/// - `act`:                `[m, k]` row-major FP8 E4M3 activations
/// - `w`:                  `[n, k]` row-major FP8 E4M3 weights
/// - `w_scale_colmaj`:     Weight BLK128x128 scales in col-major `[ceil(n/128), ceil(k/128)]`
/// - `act_scale_colmaj`:   Activation BLK128x128 scales in col-major `[ceil(k/128), ceil(m/128)]`
///
/// Both operands use `BLK128x128_32F` scaling so cuBLASLt can select true
/// FP8 tensor core kernels on Hopper (SM90).
///
/// Returns `[m * n]` BF16 output in **row-major** layout (no transpose needed).
///
/// **m and k must be multiples of 128** for BLK128x128 alignment.
pub fn fp8_matmul(
    device: &CudaDevice,
    act: &CudaSlice<float8::F8E4M3>,
    w: &CudaSlice<float8::F8E4M3>,
    w_scale_colmaj: &CudaSlice<f32>,
    act_scale_colmaj: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<half::bf16>> {
    if m % 128 != 0 {
        crate::bail!("fp8_matmul: m={m} must be a multiple of 128 for BLK128x128 scaling");
    }
    if k % 128 != 0 {
        crate::bail!("fp8_matmul: k={k} must be a multiple of 128 for BLK128x128 scaling");
    }

    let stream = device.cuda_stream();
    let lt_handle = *device.cublaslt_handle().handle();

    // Produce row-major D[m,n] directly by placing weights as A (OP_T)
    // and activations as B (OP_N).
    //
    // Row-major [p,q] data lives in memory as col-major (q,p).
    //
    // cuBLASLt: D_col = op(A) · op(B)
    //   A = weights  col-major (k,n), OP_T → logical (n,k)
    //   B = activations col-major (k,m), OP_N → logical (k,m)
    //   D_col = (n,k)·(k,m) = col-major (n,m) with ld=n
    //
    // Col-major (n,m) with ld=n is exactly row-major (m,n). No transpose.
    //
    // BLK128x128 on A(OP_T): scale indexed on POST-transpose shape (n,k),
    // so scale grid is [ceil(n/128), ceil(k/128)] in col-major.
    //
    // BLK128x128 on B(OP_N): scale indexed on the stored shape (k,m),
    // so scale grid is [ceil(k/128), ceil(m/128)] in col-major.

    let matmul_desc = MatmulDescGuard(lt::create_matmul_desc(
        lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        lt_sys::cudaDataType::CUDA_R_32F,
    ).w()?);

    let transa: i32 = cublas_sys::cublasOperation_t::CUBLAS_OP_T as i32;
    let transb: i32 = cublas_sys::cublasOperation_t::CUBLAS_OP_N as i32;
    unsafe {
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &transa)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &transb)?;
    }

    let blk_mode: i32 = lt_sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F as i32;

    let (ws_ptr, _g_ws) = w_scale_colmaj.device_ptr(&stream);
    let (as_ptr, _g_as) = act_scale_colmaj.device_ptr(&stream);

    unsafe {
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws_ptr)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &blk_mode)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as_ptr)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &blk_mode)?;
    }

    // A = weights col-major (k,n), B = activations col-major (k,m)
    let layout_a = MatrixLayoutGuard(lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_8F_E4M3, k as u64, n as u64, k as i64).w()?);
    let layout_b = MatrixLayoutGuard(lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_8F_E4M3, k as u64, m as u64, k as i64).w()?);
    // D col-major (n,m) with ld=n = row-major (m,n)
    let layout_d = MatrixLayoutGuard(lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_16BF, n as u64, m as u64, n as i64).w()?);

    let mut d_buf: CudaSlice<half::bf16> = unsafe { device.alloc::<half::bf16>(m * n)? };

    let workspace = device.fp8_workspace();
    let workspace_size: usize = workspace.len();

    let matmul_pref = MatmulPrefGuard(lt::create_matmul_pref().w()?);
    unsafe {
        lt::set_matmul_pref_attribute(
            matmul_pref.0,
            lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const c_void,
            std::mem::size_of_val(&workspace_size),
        ).w()?;
    }

    let heuristic = unsafe {
        lt::get_matmul_algo_heuristic(lt_handle, matmul_desc.0, layout_a.0, layout_b.0, layout_d.0, layout_d.0, matmul_pref.0)
    }.w()?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    {
        let (w_data, _g1) = w.device_ptr(&stream);
        let (a_data, _g2) = act.device_ptr(&stream);
        let (d_ptr, _g3) = d_buf.device_ptr_mut(&stream);
        let (ws_data, _g4) = workspace.device_ptr(&stream);

        unsafe {
            lt::matmul(
                lt_handle, matmul_desc.0,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                w_data as *const c_void, layout_a.0,
                a_data as *const c_void, layout_b.0,
                d_ptr as *const c_void, layout_d.0,
                d_ptr as *mut c_void, layout_d.0,
                &heuristic.algo as *const _,
                ws_data as *mut c_void, workspace_size,
                stream.cu_stream() as *mut _,
            ).w()?;
        }
    }

    Ok(d_buf)
}

/// FP8 GEMM with per-tensor scalar scaling.
///
/// Computes `D = alpha * (a_scale * act) @ (w_scale * w)^T`
///
/// - `act`:     `[m, k]` row-major FP8 E4M3
/// - `w`:       `[n, k]` row-major FP8 E4M3
/// - `w_scale`: scalar F32 scale for weights
/// - `a_scale`: scalar F32 scale for activations
///
/// Returns `[m * n]` BF16 output in row-major layout.
pub fn fp8_matmul_scalar(
    device: &CudaDevice,
    act: &CudaSlice<float8::F8E4M3>,
    w: &CudaSlice<float8::F8E4M3>,
    w_scale: &CudaSlice<f32>,
    a_scale: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<half::bf16>> {
    let stream = device.cuda_stream();
    let lt_handle = *device.cublaslt_handle().handle();

    let matmul_desc = MatmulDescGuard(lt::create_matmul_desc(
        lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        lt_sys::cudaDataType::CUDA_R_32F,
    ).w()?);

    let transa: i32 = cublas_sys::cublasOperation_t::CUBLAS_OP_T as i32;
    let transb: i32 = cublas_sys::cublasOperation_t::CUBLAS_OP_N as i32;
    unsafe {
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &transa)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &transb)?;
    }

    let scalar_mode: i32 = lt_sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F as i32;

    let (ws_ptr, _g_ws) = w_scale.device_ptr(&stream);
    let (as_ptr, _g_as) = a_scale.device_ptr(&stream);

    unsafe {
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws_ptr)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scalar_mode)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as_ptr)?;
        set_desc_attr(matmul_desc.0, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scalar_mode)?;
    }

    let layout_a = MatrixLayoutGuard(lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_8F_E4M3, k as u64, n as u64, k as i64).w()?);
    let layout_b = MatrixLayoutGuard(lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_8F_E4M3, k as u64, m as u64, k as i64).w()?);
    let layout_d = MatrixLayoutGuard(lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_16BF, n as u64, m as u64, n as i64).w()?);

    let mut d_buf: CudaSlice<half::bf16> = unsafe { device.alloc::<half::bf16>(m * n)? };

    let workspace = device.fp8_workspace();
    let workspace_size: usize = workspace.len();

    let matmul_pref = MatmulPrefGuard(lt::create_matmul_pref().w()?);
    unsafe {
        lt::set_matmul_pref_attribute(
            matmul_pref.0,
            lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const c_void,
            std::mem::size_of_val(&workspace_size),
        ).w()?;
    }

    let heuristic = unsafe {
        lt::get_matmul_algo_heuristic(lt_handle, matmul_desc.0, layout_a.0, layout_b.0, layout_d.0, layout_d.0, matmul_pref.0)
    }.w()?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    {
        let (w_data, _g1) = w.device_ptr(&stream);
        let (a_data, _g2) = act.device_ptr(&stream);
        let (d_ptr, _g3) = d_buf.device_ptr_mut(&stream);
        let (ws_data, _g4) = workspace.device_ptr(&stream);

        unsafe {
            lt::matmul(
                lt_handle, matmul_desc.0,
                &alpha as *const _ as *const c_void,
                &beta as *const _ as *const c_void,
                w_data as *const c_void, layout_a.0,
                a_data as *const c_void, layout_b.0,
                d_ptr as *const c_void, layout_d.0,
                d_ptr as *mut c_void, layout_d.0,
                &heuristic.algo as *const _,
                ws_data as *mut c_void, workspace_size,
                stream.cu_stream() as *mut _,
            ).w()?;
        }
    }

    Ok(d_buf)
}

unsafe fn set_desc_attr<T>(
    desc: lt_sys::cublasLtMatmulDesc_t,
    attr: lt_sys::cublasLtMatmulDescAttributes_t,
    value: &T,
) -> Result<()> {
    lt::set_matmul_desc_attribute(
        desc,
        attr,
        value as *const _ as *const c_void,
        std::mem::size_of::<T>(),
    )
    .w()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::PushKernelArg;

    fn probe_fp8_heuristic(m: usize, n: usize, k: usize, scale_mode: i32, mode_name: &str) -> bool {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let lt = cudarc::cublaslt::CudaBlasLT::new(stream.clone()).unwrap();
        let lt_handle = *lt.handle();

        let desc = lt::create_matmul_desc(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt_sys::cudaDataType::CUDA_R_32F,
        ).unwrap();

        let transa: i32 = cublas_sys::cublasOperation_t::CUBLAS_OP_T as i32;
        let transb: i32 = cublas_sys::cublasOperation_t::CUBLAS_OP_N as i32;
        unsafe {
            lt::set_matmul_desc_attribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &transa as *const _ as *const c_void, 4).unwrap();
            lt::set_matmul_desc_attribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &transb as *const _ as *const c_void, 4).unwrap();
        }

        // Allocate dummy scale pointers
        let scale_a: CudaSlice<f32> = unsafe { stream.alloc::<f32>(1024).unwrap() };
        let scale_b: CudaSlice<f32> = unsafe { stream.alloc::<f32>(1024).unwrap() };
        let (sa_ptr, _) = scale_a.device_ptr(&stream);
        let (sb_ptr, _) = scale_b.device_ptr(&stream);

        unsafe {
            lt::set_matmul_desc_attribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa_ptr as *const _ as *const c_void, std::mem::size_of::<u64>()).unwrap();
            lt::set_matmul_desc_attribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode as *const _ as *const c_void, 4).unwrap();
            lt::set_matmul_desc_attribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb_ptr as *const _ as *const c_void, std::mem::size_of::<u64>()).unwrap();
            lt::set_matmul_desc_attribute(desc, lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode as *const _ as *const c_void, 4).unwrap();
        }

        let layout_a = lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_8F_E4M3, k as u64, n as u64, k as i64).unwrap();
        let layout_b = lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_8F_E4M3, k as u64, m as u64, k as i64).unwrap();
        let layout_d = lt::create_matrix_layout(lt_sys::cudaDataType::CUDA_R_16BF, n as u64, m as u64, n as i64).unwrap();

        let workspace_size: usize = 33_554_432;
        let pref = lt::create_matmul_pref().unwrap();
        unsafe {
            lt::set_matmul_pref_attribute(pref, lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size as *const _ as *const c_void, 8).unwrap();
        }

        let result = unsafe {
            lt::get_matmul_algo_heuristic(lt_handle, desc, layout_a, layout_b, layout_d, layout_d, pref)
        };

        unsafe {
            let _ = lt::destroy_matmul_desc(desc);
            let _ = lt::destroy_matrix_layout(layout_a);
            let _ = lt::destroy_matrix_layout(layout_b);
            let _ = lt::destroy_matrix_layout(layout_d);
            let _ = lt::destroy_matmul_pref(pref);
        }

        let ok = result.is_ok();
        if ok {
            eprintln!("  OK  m={m:>5} n={n:>5} k={k:>5} mode={mode_name}");
        } else {
            eprintln!("  FAIL m={m:>5} n={n:>5} k={k:>5} mode={mode_name} err={:?}", result.err());
        }
        ok
    }

    #[test]
    fn test_fp8_scaling_modes_probe() {
        let modes: Vec<(i32, &str)> = vec![
            (0, "SCALAR_32F"),
            (4, "VEC128_32F"),
            (5, "BLK128x128_32F"),
        ];

        // Qwen3-8B dimensions
        let qwen_dims = vec![
            (1, 4096, 4096, "decode b=1, Q/O proj"),
            (3, 4096, 4096, "decode b=3, Q/O proj"),
            (3, 1024, 4096, "decode b=3, K/V proj"),
            (3, 12288, 4096, "decode b=3, gate/up proj"),
            (3, 4096, 12288, "decode b=3, down proj"),
            (128, 4096, 4096, "padded128, Q/O proj"),
            (128, 1024, 4096, "padded128, K/V proj"),
            (128, 12288, 4096, "padded128, gate/up proj"),
            (128, 4096, 12288, "padded128, down proj"),
            (256, 4096, 4096, "prefill chunk 256"),
            (512, 4096, 4096, "prefill chunk 512"),
            (1000, 4096, 4096, "prefill 1000"),
            (1024, 4096, 4096, "prefill 1024"),
        ];

        eprintln!("\n=== FP8 cuBLASLt Scaling Mode Probe ===\n");
        for (mode_val, mode_name) in &modes {
            eprintln!("--- {mode_name} ---");
            for (m, n, k, desc) in &qwen_dims {
                eprint!("  [{desc}] ");
                probe_fp8_heuristic(*m, *n, *k, *mode_val, mode_name);
            }
            eprintln!();
        }
    }
}
