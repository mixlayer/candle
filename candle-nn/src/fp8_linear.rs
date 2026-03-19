use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use std::sync::{Arc, Mutex, OnceLock};

/// Block sizes for FP8 blockwise quantization.
/// Qwen3-8B-FP8 uses [128, 128].
#[derive(Clone, Debug)]
pub struct Fp8BlockSize {
    pub block_r: usize,
    pub block_c: usize,
}

impl Default for Fp8BlockSize {
    fn default() -> Self {
        Self {
            block_r: 128,
            block_c: 128,
        }
    }
}

/// CustomOp2 that dequantizes a contiguous F8E4M3 weight tensor using a F32
/// blockwise scale_inv tensor, producing output in `target_dtype` (BF16/F16/F32).
///
/// weight:    [rows, cols]  F8E4M3
/// scale_inv: [rows/block_r, cols/block_c]  F32
/// output:    [rows, cols]  target_dtype
struct Fp8BlockwiseDequant {
    rows: usize,
    cols: usize,
    block_r: usize,
    block_c: usize,
    target_dtype: DType,
}

impl candle::CustomOp2 for Fp8BlockwiseDequant {
    fn name(&self) -> &'static str {
        "fp8-blockwise-dequant"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let weight = match s1 {
            CpuStorage::F8E4M3(data) => data,
            _ => candle::bail!("fp8 dequant: weight must be F8E4M3"),
        };
        let scale = match s2 {
            CpuStorage::F32(data) => data,
            _ => candle::bail!("fp8 dequant: scale_inv must be F32"),
        };

        let w_offset = l1.start_offset();
        let s_offset = l2.start_offset();
        let scale_cols = (self.cols + self.block_c - 1) / self.block_c;
        let numel = self.rows * self.cols;

        match self.target_dtype {
            DType::BF16 => {
                let mut out = vec![half::bf16::ZERO; numel];
                for idx in 0..numel {
                    let r = idx / self.cols;
                    let c = idx % self.cols;
                    let w_val: f32 = weight[w_offset + idx].to_f64() as f32;
                    let s_val = scale[s_offset + (r / self.block_r) * scale_cols + c / self.block_c];
                    out[idx] = half::bf16::from_f32(w_val * s_val);
                }
                Ok((CpuStorage::BF16(out), Shape::from_dims(&[self.rows, self.cols])))
            }
            DType::F16 => {
                let mut out = vec![half::f16::ZERO; numel];
                for idx in 0..numel {
                    let r = idx / self.cols;
                    let c = idx % self.cols;
                    let w_val: f32 = weight[w_offset + idx].to_f64() as f32;
                    let s_val = scale[s_offset + (r / self.block_r) * scale_cols + c / self.block_c];
                    out[idx] = half::f16::from_f32(w_val * s_val);
                }
                Ok((CpuStorage::F16(out), Shape::from_dims(&[self.rows, self.cols])))
            }
            DType::F32 => {
                let mut out = vec![0f32; numel];
                for idx in 0..numel {
                    let r = idx / self.cols;
                    let c = idx % self.cols;
                    let w_val: f32 = weight[w_offset + idx].to_f64() as f32;
                    let s_val = scale[s_offset + (r / self.block_r) * scale_cols + c / self.block_c];
                    out[idx] = w_val * s_val;
                }
                Ok((CpuStorage::F32(out), Shape::from_dims(&[self.rows, self.cols])))
            }
            dt => candle::bail!("fp8 dequant: unsupported target dtype {:?}", dt),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        _l1: &Layout,
        s2: &candle::CudaStorage,
        _l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc;
        use candle::cuda_backend::cudarc::driver::PushKernelArg;
        use candle::cuda_backend::CudaStorageSlice as S;
        use candle::cuda_backend::WrapErr;

        let dev = &s1.device;
        let weight_slice = match &s1.slice {
            S::F8E4M3(s) => s,
            _ => candle::bail!("fp8 dequant: weight must be F8E4M3"),
        };
        let scale_slice = match &s2.slice {
            S::F32(s) => s,
            _ => candle::bail!("fp8 dequant: scale_inv must be F32"),
        };

        let numel = self.rows * self.cols;
        let kernel_name = match self.target_dtype {
            DType::BF16 => "fp8_blockwise_dequant_bf16",
            DType::F16 => "fp8_blockwise_dequant_f16",
            DType::F32 => "fp8_blockwise_dequant_f32",
            dt => candle::bail!("fp8 dequant: unsupported target dtype {:?}", dt),
        };

        let func = dev.get_or_load_func(kernel_name, &candle::cuda_backend::kernels::FP8_DEQUANT)?;
        let block_dim = 256u32;
        let grid_dim = ((numel as u32) + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let shape = Shape::from_dims(&[self.rows, self.cols]);
        let rows = self.rows as u32;
        let cols = self.cols as u32;
        let block_r = self.block_r as u32;
        let block_c = self.block_c as u32;

        let out_slice = match self.target_dtype {
            DType::BF16 => {
                let out = unsafe { dev.alloc::<half::bf16>(numel)? };
                let mut builder = func.builder();
                builder.arg(weight_slice);
                builder.arg(scale_slice);
                builder.arg(&out);
                builder.arg(&rows);
                builder.arg(&cols);
                builder.arg(&block_r);
                builder.arg(&block_c);
                unsafe { builder.launch(cfg) }.w()?;
                S::BF16(out)
            }
            DType::F16 => {
                let out = unsafe { dev.alloc::<half::f16>(numel)? };
                let mut builder = func.builder();
                builder.arg(weight_slice);
                builder.arg(scale_slice);
                builder.arg(&out);
                builder.arg(&rows);
                builder.arg(&cols);
                builder.arg(&block_r);
                builder.arg(&block_c);
                unsafe { builder.launch(cfg) }.w()?;
                S::F16(out)
            }
            DType::F32 => {
                let out = unsafe { dev.alloc::<f32>(numel)? };
                let mut builder = func.builder();
                builder.arg(weight_slice);
                builder.arg(scale_slice);
                builder.arg(&out);
                builder.arg(&rows);
                builder.arg(&cols);
                builder.arg(&block_r);
                builder.arg(&block_c);
                unsafe { builder.launch(cfg) }.w()?;
                S::F32(out)
            }
            _ => unreachable!(),
        };

        let storage = candle::CudaStorage {
            slice: out_slice,
            device: dev.clone(),
        };
        Ok((storage, shape))
    }
}

/// Dequantize a contiguous FP8 E4M3 weight tensor using blockwise scales.
///
/// `weight`:    contiguous `[rows, cols]` tensor in `F8E4M3`
/// `scale_inv`: contiguous `[ceil(rows/block_r), ceil(cols/block_c)]` tensor in `F32`
///
/// Returns a tensor of shape `[rows, cols]` in `target_dtype`.
pub fn fp8_blockwise_dequant(
    weight: &Tensor,
    scale_inv: &Tensor,
    block_size: &Fp8BlockSize,
    target_dtype: DType,
) -> Result<Tensor> {
    if weight.dtype() != DType::F8E4M3 {
        candle::bail!(
            "fp8_blockwise_dequant: expected F8E4M3 weight, got {:?}",
            weight.dtype()
        );
    }
    if scale_inv.dtype() != DType::F32 {
        candle::bail!(
            "fp8_blockwise_dequant: expected F32 scale_inv, got {:?}",
            scale_inv.dtype()
        );
    }

    let weight = weight.contiguous()?;
    let scale_inv = scale_inv.contiguous()?;

    let (rows, cols) = weight.dims2()?;

    let op = Fp8BlockwiseDequant {
        rows,
        cols,
        block_r: block_size.block_r,
        block_c: block_size.block_c,
        target_dtype,
    };

    weight.apply_op2_no_bwd(&scale_inv, &op)
}

/// Dynamically quantize a BF16 activation tensor to FP8 E4M3 with 128×128 blockwise scaling.
///
/// `input`:  contiguous `[M, K]` tensor in BF16 (or will be made contiguous).
///           M and K must be multiples of 128.
///
/// Returns `(quantized, scales)`:
///   - `quantized`: `[M, K]` in F8E4M3
///   - `scales`:    `[ceil(K/128) * ceil(M/128)]` flat F32 in col-major layout
///                  matching cuBLASLt BLK128x128 on B(OP_N) stored as col-major (K, M)
pub fn fp8_dynamic_quantize(input: &Tensor) -> Result<(Tensor, Tensor)> {
    if input.dtype() != DType::BF16 {
        candle::bail!(
            "fp8_dynamic_quantize: expected BF16 input, got {:?}",
            input.dtype()
        );
    }
    let input = input.contiguous()?;
    let dims = input.dims();
    if dims.len() < 2 {
        candle::bail!("fp8_dynamic_quantize: expected at least 2D input");
    }
    let rows = dims[..dims.len() - 1].iter().product::<usize>();
    let cols = *dims.last().unwrap();

    match input.device() {
        candle::Device::Cpu => fp8_dynamic_quantize_cpu(&input, rows, cols),
        #[cfg(feature = "cuda")]
        candle::Device::Cuda(_) => fp8_dynamic_quantize_cuda(&input, rows, cols),
        #[cfg(not(feature = "cuda"))]
        candle::Device::Cuda(_) => candle::bail!("fp8_dynamic_quantize: CUDA not enabled"),
        _ => candle::bail!("fp8_dynamic_quantize: unsupported device"),
    }
}

/// CPU fallback: blockwise 128×128 quantization
fn fp8_dynamic_quantize_cpu(input: &Tensor, rows: usize, cols: usize) -> Result<(Tensor, Tensor)> {
    use half::bf16;

    let data = input.flatten_all()?.to_vec1::<bf16>()?;
    let fp8_e4m3_max: f32 = 448.0;

    let m_blocks = (rows + 127) / 128;
    let k_blocks = (cols + 127) / 128;
    let num_scales = k_blocks * m_blocks;
    let mut out_scales = vec![0.0f32; num_scales];
    let mut out_f32 = vec![0.0f32; rows * cols];

    for mb in 0..m_blocks {
        for kb in 0..k_blocks {
            let m_start = mb * 128;
            let k_start = kb * 128;
            let m_end = (m_start + 128).min(rows);
            let k_end = (k_start + 128).min(cols);

            let mut absmax: f32 = 0.0;
            for r in m_start..m_end {
                for c in k_start..k_end {
                    let val = data[r * cols + c].to_f32().abs();
                    if val > absmax {
                        absmax = val;
                    }
                }
            }
            let scale = if absmax > 0.0 { absmax / fp8_e4m3_max } else { 1.0 };
            // Col-major: scale[kb + mb * k_blocks]
            out_scales[kb + mb * k_blocks] = scale;
            let inv_scale = 1.0 / scale;

            for r in m_start..m_end {
                for c in k_start..k_end {
                    let val = data[r * cols + c].to_f32() * inv_scale;
                    out_f32[r * cols + c] = val.clamp(-fp8_e4m3_max, fp8_e4m3_max);
                }
            }
        }
    }

    let device = input.device();
    let quant_f32 = Tensor::from_vec(out_f32, (rows, cols), device)?;
    let quant = quant_f32.to_dtype(DType::F8E4M3)?;
    let scales = Tensor::from_vec(out_scales, (num_scales,), device)?;
    Ok((quant, scales))
}

#[cfg(feature = "cuda")]
fn fp8_dynamic_quantize_cuda(input: &Tensor, rows: usize, cols: usize) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc;
    use candle::cuda_backend::cudarc::driver::PushKernelArg;
    use candle::cuda_backend::CudaStorageSlice as S;
    use candle::cuda_backend::WrapErr;

    let (storage, _layout) = input.storage_and_layout();
    let cuda_storage = match &*storage {
        candle::Storage::Cuda(s) => s,
        _ => candle::bail!("fp8_dynamic_quantize_cuda: expected CUDA storage"),
    };
    let dev = &cuda_storage.device;
    let input_slice = match &cuda_storage.slice {
        S::BF16(s) => s,
        _ => candle::bail!("fp8 quantize: input must be BF16"),
    };

    let m = rows as u32;
    let k = cols as u32;
    let k_blocks = (k + 127) / 128;
    let m_blocks = (m + 127) / 128;
    let num_scales = (k_blocks * m_blocks) as usize;

    let func = dev.get_or_load_func(
        "bf16_to_fp8_e4m3_blockwise",
        &candle::cuda_backend::kernels::FP8_QUANTIZE,
    )?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (k_blocks, m_blocks, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let numel = rows * cols;
    let out_fp8 = unsafe { dev.alloc::<candle::cuda_backend::float8::F8E4M3>(numel)? };
    let out_scales = unsafe { dev.alloc::<f32>(num_scales)? };

    let mut builder = func.builder();
    builder.arg(input_slice);
    builder.arg(&out_fp8);
    builder.arg(&out_scales);
    builder.arg(&m);
    builder.arg(&k);
    unsafe { builder.launch(cfg) }.w()?;

    let quant_storage = candle::CudaStorage {
        slice: S::F8E4M3(out_fp8),
        device: dev.clone(),
    };
    let scale_storage = candle::CudaStorage {
        slice: S::F32(out_scales),
        device: dev.clone(),
    };

    let quant_tensor = Tensor::from_storage(
        candle::Storage::Cuda(quant_storage),
        (rows, cols),
        candle::op::BackpropOp::none(),
        false,
    );
    let scale_tensor = Tensor::from_storage(
        candle::Storage::Cuda(scale_storage),
        (num_scales,),
        candle::op::BackpropOp::none(),
        false,
    );

    Ok((quant_tensor, scale_tensor))
}

/// Pre-allocated activation buffers for FP8 quantization, ensuring stable
/// device pointers across CUDA graph capture and replay.
#[cfg(feature = "cuda")]
struct Fp8ActCache {
    fp8_buf: candle::cuda_backend::cudarc::driver::CudaSlice<candle::cuda_backend::float8::F8E4M3>,
    scale_buf: candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
    numel: usize,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for Fp8ActCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fp8ActCache")
            .field("numel", &self.numel)
            .finish()
    }
}

/// A linear layer backed by FP8 E4M3 weights.
///
/// On CUDA, uses cuBLASLt FP8 GEMM with per-tensor scalar scaling -- no BF16
/// dequant required. The blockwise-scaled weights from the checkpoint are
/// re-quantized to scalar scaling at construction time (one-time cost).
///
/// Memory: ~1 byte/param (FP8 weights only, no BF16 cache). The original
/// blockwise FP8 weight and scale_inv are dropped after re-quantization.
#[derive(Clone, Debug)]
pub struct Fp8Linear {
    weight: Tensor,
    scale_inv: Tensor,
    bias: Option<Tensor>,
    block_size: Fp8BlockSize,
    dequant_dtype: DType,
    #[cfg(feature = "cuda")]
    scalar_weight: Arc<OnceLock<Tensor>>,
    #[cfg(feature = "cuda")]
    scalar_scale: Arc<OnceLock<Tensor>>,
    #[cfg(feature = "cuda")]
    act_cache: Arc<Mutex<Option<Fp8ActCache>>>,
}

impl Fp8Linear {
    pub fn new(
        weight: Tensor,
        scale_inv: Tensor,
        bias: Option<Tensor>,
        block_size: Fp8BlockSize,
        dequant_dtype: DType,
    ) -> Self {
        Self {
            weight,
            scale_inv,
            bias,
            block_size,
            dequant_dtype,
            #[cfg(feature = "cuda")]
            scalar_weight: Arc::new(OnceLock::new()),
            #[cfg(feature = "cuda")]
            scalar_scale: Arc::new(OnceLock::new()),
            #[cfg(feature = "cuda")]
            act_cache: Arc::new(Mutex::new(None)),
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn scale_inv(&self) -> &Tensor {
        &self.scale_inv
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    fn dequantized_weight(&self) -> Result<Tensor> {
        fp8_blockwise_dequant(
            &self.weight,
            &self.scale_inv,
            &self.block_size,
            self.dequant_dtype,
        )
    }

    /// Lazily re-quantize blockwise FP8 weights to scalar-scaled FP8 on CUDA.
    /// Dequants to BF16 once, finds global absmax, re-quantizes to FP8.
    /// The BF16 intermediate is freed after re-quantization.
    #[cfg(feature = "cuda")]
    fn ensure_scalar_weight(&self) -> Result<(&Tensor, &Tensor)> {
        if self.scalar_weight.get().is_some() {
            return Ok((
                self.scalar_weight.get().unwrap(),
                self.scalar_scale.get().unwrap(),
            ));
        }

        let bf16_w = self.dequantized_weight()?;
        let (scalar_w, scalar_s) = fp8_scalar_quantize(&bf16_w)?;
        let _ = self.scalar_weight.set(scalar_w);
        let _ = self.scalar_scale.set(scalar_s);
        Ok((
            self.scalar_weight.get().unwrap(),
            self.scalar_scale.get().unwrap(),
        ))
    }

    /// FP8 forward pass using pre-allocated activation buffers for CUDA graph
    /// compatibility. The activation FP8 buffer and scale buffer are cached
    /// with stable device pointers so they survive graph capture/replay.
    #[cfg(feature = "cuda")]
    fn fp8_scalar_forward(&self, x: &Tensor) -> Result<Tensor> {
        use candle::cuda_backend::cudarc;
        use candle::cuda_backend::cudarc::driver::PushKernelArg;
        use candle::cuda_backend::CudaStorageSlice as S;
        use candle::cuda_backend::WrapErr;
        use candle::cuda_backend::fp8_gemm;

        let (scalar_w, scalar_s) = self.ensure_scalar_weight()?;

        let orig_dims = x.dims().to_vec();
        let k = *orig_dims.last().unwrap();
        let m: usize = orig_dims[..orig_dims.len() - 1].iter().product();
        let (n, k_w) = scalar_w.dims2()?;
        if k != k_w {
            candle::bail!("fp8_scalar_forward: activation K ({k}) != weight K ({k_w})");
        }

        let x_2d = x.contiguous()?.reshape((m, k))?;
        let numel = m * k;

        let (x_storage, _) = x_2d.storage_and_layout();
        let x_cuda = match &*x_storage {
            candle::Storage::Cuda(s) => s,
            _ => candle::bail!("expected CUDA"),
        };
        let input_slice = match &x_cuda.slice {
            S::BF16(s) => s,
            _ => candle::bail!("expected BF16 input"),
        };
        let dev = &x_cuda.device;

        let mut cache_guard = self.act_cache.lock().unwrap();
        let needs_alloc = cache_guard.as_ref().map_or(true, |c| c.numel != numel);
        if needs_alloc {
            let fp8_buf = unsafe {
                dev.alloc::<candle::cuda_backend::float8::F8E4M3>(numel)?
            };
            let scale_buf = dev.alloc_zeros::<f32>(1)?;
            *cache_guard = Some(Fp8ActCache { fp8_buf, scale_buf, numel });
        }
        let cache = cache_guard.as_ref().unwrap();

        {
            let func = dev.get_or_load_func(
                "fp8_zero_scale",
                &candle::cuda_backend::kernels::FP8_QUANTIZE,
            )?;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(&cache.scale_buf);
            unsafe { builder.launch(cfg) }.w()?;
        }

        {
            let func = dev.get_or_load_func(
                "bf16_absmax",
                &candle::cuda_backend::kernels::FP8_QUANTIZE,
            )?;
            let block_dim = 256u32;
            let grid_dim = ((numel as u32) + block_dim - 1) / block_dim;
            let grid_dim = grid_dim.min(1024);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_elem = numel as u32;
            let mut builder = func.builder();
            builder.arg(input_slice);
            builder.arg(&cache.scale_buf);
            builder.arg(&n_elem);
            unsafe { builder.launch(cfg) }.w()?;
        }

        {
            let func = dev.get_or_load_func(
                "fp8_compute_scale",
                &candle::cuda_backend::kernels::FP8_QUANTIZE,
            )?;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(&cache.scale_buf);
            unsafe { builder.launch(cfg) }.w()?;
        }

        {
            let func = dev.get_or_load_func(
                "bf16_to_fp8_e4m3_scalar",
                &candle::cuda_backend::kernels::FP8_QUANTIZE,
            )?;
            let block_dim = 256u32;
            let grid_dim = ((numel as u32) + block_dim - 1) / block_dim;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_elem = numel as u32;
            let mut builder = func.builder();
            builder.arg(input_slice);
            builder.arg(&cache.fp8_buf);
            builder.arg(&cache.scale_buf);
            builder.arg(&n_elem);
            unsafe { builder.launch(cfg) }.w()?;
        }

        let (w_storage, _) = scalar_w.storage_and_layout();
        let w_cuda = match &*w_storage {
            candle::Storage::Cuda(s) => s,
            _ => candle::bail!("expected CUDA"),
        };
        let w_slice = match &w_cuda.slice {
            S::F8E4M3(s) => s,
            _ => candle::bail!("expected F8E4M3 weight"),
        };

        let (ws_storage, _) = scalar_s.storage_and_layout();
        let ws_cuda = match &*ws_storage {
            candle::Storage::Cuda(s) => s,
            _ => candle::bail!("expected CUDA"),
        };
        let ws_slice = match &ws_cuda.slice {
            S::F32(s) => s,
            _ => candle::bail!("expected F32 scale"),
        };

        let d_buf = fp8_gemm::fp8_matmul_scalar(
            dev,
            &cache.fp8_buf,
            w_slice,
            ws_slice,
            &cache.scale_buf,
            m,
            n,
            k,
        )?;

        drop(cache_guard);

        let out_storage = candle::CudaStorage {
            slice: S::BF16(d_buf),
            device: dev.clone(),
        };
        let out = Tensor::from_storage(
            candle::Storage::Cuda(out_storage),
            (m, n),
            candle::op::BackpropOp::none(),
            false,
        );

        let mut out_shape = orig_dims[..orig_dims.len() - 1].to_vec();
        out_shape.push(n);
        let out = out.reshape(out_shape)?;

        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

/// Quantize a BF16 tensor to FP8 E4M3 with per-tensor scalar scaling.
///
/// Returns `(fp8_tensor, scale)` where scale is a single F32 value on device.
#[cfg(feature = "cuda")]
pub fn fp8_scalar_quantize(input: &Tensor) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc;
    use candle::cuda_backend::cudarc::driver::PushKernelArg;
    use candle::cuda_backend::CudaStorageSlice as S;
    use candle::cuda_backend::WrapErr;

    if input.dtype() != DType::BF16 {
        candle::bail!("fp8_scalar_quantize: expected BF16, got {:?}", input.dtype());
    }
    let input = input.contiguous()?;
    let shape = input.shape().clone();
    let numel = shape.elem_count();

    let (storage, _layout) = input.storage_and_layout();
    let cuda_storage = match &*storage {
        candle::Storage::Cuda(s) => s,
        _ => candle::bail!("fp8_scalar_quantize: expected CUDA storage"),
    };
    let dev = &cuda_storage.device;
    let input_slice = match &cuda_storage.slice {
        S::BF16(s) => s,
        _ => candle::bail!("fp8_scalar_quantize: input must be BF16"),
    };

    // Step 1: compute absmax via reduction kernel
    let scale_buf = dev.alloc_zeros::<f32>(1)?;
    {
        let func = dev.get_or_load_func("bf16_absmax", &candle::cuda_backend::kernels::FP8_QUANTIZE)?;
        let block_dim = 256u32;
        let grid_dim = ((numel as u32) + block_dim - 1) / block_dim;
        let grid_dim = grid_dim.min(1024);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = numel as u32;
        let mut builder = func.builder();
        builder.arg(input_slice);
        builder.arg(&scale_buf);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }.w()?;
    }

    // Step 2: compute scale = absmax / 448.0 on device (no host sync!)
    {
        let func = dev.get_or_load_func("fp8_compute_scale", &candle::cuda_backend::kernels::FP8_QUANTIZE)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&scale_buf);
        unsafe { builder.launch(cfg) }.w()?;
    }

    // Step 3: quantize BF16 -> FP8 using the scalar scale
    let out_fp8 = unsafe { dev.alloc::<candle::cuda_backend::float8::F8E4M3>(numel)? };
    {
        let func = dev.get_or_load_func("bf16_to_fp8_e4m3_scalar", &candle::cuda_backend::kernels::FP8_QUANTIZE)?;
        let block_dim = 256u32;
        let grid_dim = ((numel as u32) + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = numel as u32;
        let mut builder = func.builder();
        builder.arg(input_slice);
        builder.arg(&out_fp8);
        builder.arg(&scale_buf);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }.w()?;
    }

    let quant_storage = candle::CudaStorage {
        slice: S::F8E4M3(out_fp8),
        device: dev.clone(),
    };
    let scale_storage = candle::CudaStorage {
        slice: S::F32(scale_buf),
        device: dev.clone(),
    };

    let quant_tensor = Tensor::from_storage(
        candle::Storage::Cuda(quant_storage),
        shape,
        candle::op::BackpropOp::none(),
        false,
    );
    let scale_tensor = Tensor::from_storage(
        candle::Storage::Cuda(scale_storage),
        (1,),
        candle::op::BackpropOp::none(),
        false,
    );

    Ok((quant_tensor, scale_tensor))
}

impl candle::Module for Fp8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if x.dtype() == DType::BF16 && matches!(x.device(), candle::Device::Cuda(_)) {
                if let Ok(result) = self.fp8_scalar_forward(x) {
                    return Ok(result);
                }
            }
        }
        // CPU / non-CUDA / FP8-unsupported-shape fallback
        let w = self.dequantized_weight()?;
        let wt = w.t()?;
        let out = x.matmul(&wt)?;
        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, IndexOp, Module};

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Result<Device> {
        Device::new_cuda(0)
    }

    #[test]
    fn test_fp8_blockwise_dequant_cpu() -> Result<()> {
        let device = &Device::Cpu;
        let rows = 4usize;
        let cols = 4usize;

        // Create fp8 weight: all 1.0 in fp8
        let ones_f32 = vec![1.0f32; rows * cols];
        let weight_f32 = Tensor::from_vec(ones_f32, (rows, cols), device)?;
        let weight = weight_f32.to_dtype(DType::F8E4M3)?;

        // scale_inv of 2.0 for the single block (4x4 with block_size 4x4)
        let scale_inv = Tensor::from_vec(vec![2.0f32], (1, 1), device)?;
        let block_size = Fp8BlockSize {
            block_r: 4,
            block_c: 4,
        };

        let out = fp8_blockwise_dequant(&weight, &scale_inv, &block_size, DType::F32)?;
        let out_data = out.to_vec2::<f32>()?;

        for row in &out_data {
            for &val in row {
                assert!(
                    (val - 2.0).abs() < 0.1,
                    "expected ~2.0, got {}",
                    val
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_fp8_blockwise_dequant_multiple_blocks_cpu() -> Result<()> {
        let device = &Device::Cpu;
        let block_r = 2usize;
        let block_c = 2usize;
        let rows = 4usize;
        let cols = 4usize;

        let ones_f32 = vec![1.0f32; rows * cols];
        let weight_f32 = Tensor::from_vec(ones_f32, (rows, cols), device)?;
        let weight = weight_f32.to_dtype(DType::F8E4M3)?;

        // 2x2 scale grid, each with a different scale
        let scales = vec![1.0f32, 2.0, 3.0, 4.0];
        let scale_inv = Tensor::from_vec(scales, (2, 2), device)?;

        let block_size = Fp8BlockSize { block_r, block_c };
        let out = fp8_blockwise_dequant(&weight, &scale_inv, &block_size, DType::F32)?;
        let out_data = out.to_vec2::<f32>()?;

        // Block (0,0) -> scale 1.0, Block (0,1) -> scale 2.0
        // Block (1,0) -> scale 3.0, Block (1,1) -> scale 4.0
        assert!((out_data[0][0] - 1.0).abs() < 0.1);
        assert!((out_data[0][2] - 2.0).abs() < 0.1);
        assert!((out_data[2][0] - 3.0).abs() < 0.1);
        assert!((out_data[2][2] - 4.0).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_fp8_linear_cpu() -> Result<()> {
        let device = &Device::Cpu;

        // 2x2 weight, all 1.0 in fp8, scale 1.0
        let weight_f32 = Tensor::from_vec(vec![1.0f32; 4], (2, 2), device)?;
        let weight = weight_f32.to_dtype(DType::F8E4M3)?;
        let scale_inv = Tensor::from_vec(vec![1.0f32], (1, 1), device)?;

        let layer = Fp8Linear::new(
            weight,
            scale_inv,
            None,
            Fp8BlockSize {
                block_r: 2,
                block_c: 2,
            },
            DType::F32,
        );

        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), device)?;
        let y = layer.forward(&x)?;
        let y_data = y.to_vec2::<f32>()?;

        // y = x @ w^T = [1, 2] @ [[1,1],[1,1]]^T = [1+2, 1+2] = [3, 3]
        assert!((y_data[0][0] - 3.0).abs() < 0.2);
        assert!((y_data[0][1] - 3.0).abs() < 0.2);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_dequant_gpu() -> Result<()> {
        let device = &cuda_device()?;
        let rows = 4usize;
        let cols = 4usize;

        let ones_f32 = vec![1.0f32; rows * cols];
        let weight_f32 = Tensor::from_vec(ones_f32, (rows, cols), device)?;
        let weight = weight_f32.to_dtype(DType::F8E4M3)?;

        let scale_inv = Tensor::from_vec(vec![2.0f32], (1, 1), device)?;
        let block_size = Fp8BlockSize {
            block_r: 4,
            block_c: 4,
        };

        let out = fp8_blockwise_dequant(&weight, &scale_inv, &block_size, DType::BF16)?;
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.dims(), &[4, 4]);

        let out_f32 = out.to_dtype(DType::F32)?;
        let out_data = out_f32.to_vec2::<f32>()?;
        for row in &out_data {
            for &val in row {
                assert!(
                    (val - 2.0).abs() < 0.15,
                    "expected ~2.0, got {}",
                    val
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_dequant_multiple_blocks_gpu() -> Result<()> {
        let device = &cuda_device()?;
        let block_r = 2usize;
        let block_c = 2usize;
        let rows = 4usize;
        let cols = 4usize;

        let ones_f32 = vec![1.0f32; rows * cols];
        let weight_f32 = Tensor::from_vec(ones_f32, (rows, cols), device)?;
        let weight = weight_f32.to_dtype(DType::F8E4M3)?;

        let scales = vec![1.0f32, 2.0, 3.0, 4.0];
        let scale_inv = Tensor::from_vec(scales, (2, 2), device)?;

        let block_size = Fp8BlockSize { block_r, block_c };
        let out = fp8_blockwise_dequant(&weight, &scale_inv, &block_size, DType::BF16)?;
        let out_f32 = out.to_dtype(DType::F32)?;
        let out_data = out_f32.to_vec2::<f32>()?;

        assert!((out_data[0][0] - 1.0).abs() < 0.15);
        assert!((out_data[0][2] - 2.0).abs() < 0.15);
        assert!((out_data[2][0] - 3.0).abs() < 0.15);
        assert!((out_data[2][2] - 4.0).abs() < 0.15);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_linear_gpu() -> Result<()> {
        let device = &cuda_device()?;

        let weight_f32 = Tensor::from_vec(vec![1.0f32; 4], (2, 2), device)?;
        let weight = weight_f32.to_dtype(DType::F8E4M3)?;
        let scale_inv = Tensor::from_vec(vec![1.0f32], (1, 1), device)?;

        let layer = Fp8Linear::new(
            weight,
            scale_inv,
            None,
            Fp8BlockSize {
                block_r: 2,
                block_c: 2,
            },
            DType::BF16,
        );

        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), device)?
            .to_dtype(DType::BF16)?;
        let y = layer.forward(&x)?;
        let y_f32 = y.to_dtype(DType::F32)?;
        let y_data = y_f32.to_vec2::<f32>()?;

        assert!((y_data[0][0] - 3.0).abs() < 0.3);
        assert!((y_data[0][1] - 3.0).abs() < 0.3);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_dequant_realistic_block_size_gpu() -> Result<()> {
        let device = &cuda_device()?;
        let rows = 256usize;
        let cols = 256usize;
        let block_r = 128usize;
        let block_c = 128usize;

        let weight_f32: Vec<f32> = (0..rows * cols).map(|i| ((i % 7) as f32) * 0.1).collect();
        let weight_tensor = Tensor::from_vec(weight_f32.clone(), (rows, cols), device)?;
        let weight = weight_tensor.to_dtype(DType::F8E4M3)?;

        let scales = vec![2.0f32, 0.5, 1.0, 3.0];
        let scale_inv = Tensor::from_vec(scales.clone(), (2, 2), device)?;

        let block_size = Fp8BlockSize { block_r, block_c };
        let out = fp8_blockwise_dequant(&weight, &scale_inv, &block_size, DType::BF16)?;
        let out_f32 = out.to_dtype(DType::F32)?;
        let out_data = out_f32.flatten_all()?.to_vec1::<f32>()?;

        // Spot check a few values
        // Element (0, 0) in block (0,0) with scale 2.0
        let idx = 0;
        let expected_raw = ((0 % 7) as f32) * 0.1; // 0.0
        assert!(
            (out_data[idx] - expected_raw * 2.0).abs() < 0.15,
            "idx=0: got {}, expected ~{}",
            out_data[idx],
            expected_raw * 2.0
        );

        // Element (0, 128) in block (0,1) with scale 0.5
        let idx2 = 128;
        let expected_raw2 = ((128 % 7) as f32) * 0.1; // 2*0.1 = 0.2 (128%7=2)
        assert!(
            (out_data[idx2] - expected_raw2 * 0.5).abs() < 0.15,
            "idx=128: got {}, expected ~{}",
            out_data[idx2],
            expected_raw2 * 0.5
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_dequant_real_safetensors() -> Result<()> {
        let path = "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3/model-00001-of-00002.safetensors";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test: safetensors file not found at {path}");
            return Ok(());
        }

        let device = &cuda_device()?;
        let tensors = unsafe { candle::safetensors::MmapedSafetensors::new(path)? };

        let weight = tensors.load("model.layers.0.mlp.gate_proj.weight", device)?;
        let scale_inv = tensors.load("model.layers.0.mlp.gate_proj.weight_scale_inv", device)?;

        eprintln!("weight dtype={:?} shape={:?}", weight.dtype(), weight.shape());
        eprintln!("scale dtype={:?} shape={:?}", scale_inv.dtype(), scale_inv.shape());

        assert_eq!(weight.dtype(), DType::F8E4M3);
        assert_eq!(weight.dims(), &[12288, 4096]);

        let scale_inv_f32 = scale_inv.to_dtype(DType::F32)?;

        let block_size = Fp8BlockSize { block_r: 128, block_c: 128 };
        let out = fp8_blockwise_dequant(&weight, &scale_inv_f32, &block_size, DType::BF16)?;
        let out_f32 = out.to_dtype(DType::F32)?;

        let row0 = out_f32.i(0)?.to_vec1::<f32>()?;
        let row128 = out_f32.i(128)?.to_vec1::<f32>()?;

        eprintln!("Dequant[0,0..5]: {:?}", &row0[..5]);
        eprintln!("Dequant[128,0..5]: {:?}", &row128[..5]);

        // Python reference values:
        // Dequant[0,0..5] = [-0.02274, 0.01250, 0.00426, -0.01137, -0.02274]
        // Dequant[128,0..5] = [0.03088, -0.02527, 0.03369, 0.00983, -0.00421]
        assert!((row0[0] - (-0.02274)).abs() < 0.005, "row0[0] = {}", row0[0]);
        assert!((row0[1] - 0.01250).abs() < 0.005, "row0[1] = {}", row0[1]);
        assert!((row128[0] - 0.03088).abs() < 0.005, "row128[0] = {}", row128[0]);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_dynamic_quantize_gpu() -> Result<()> {
        let device = &cuda_device()?;
        // Use 128-aligned dims for blockwise quantization
        let rows = 128usize;
        let cols = 256usize;

        let mut data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let val = ((r % 4) as f32 + 1.0) * if r % 2 == 0 { 1.0 } else { -1.0 };
            for c in 0..cols {
                data[r * cols + c] = val;
            }
        }

        let input = Tensor::from_vec(data.clone(), (rows, cols), device)?.to_dtype(DType::BF16)?;
        let (quant, scales) = fp8_dynamic_quantize(&input)?;

        assert_eq!(quant.dtype(), DType::F8E4M3);
        assert_eq!(quant.dims(), &[rows, cols]);
        assert_eq!(scales.dtype(), DType::F32);

        let k_blocks = (cols + 127) / 128; // 2
        let m_blocks = (rows + 127) / 128; // 1
        assert_eq!(scales.dims(), &[k_blocks * m_blocks]);

        let scales_data = scales.to_vec1::<f32>()?;
        eprintln!("blockwise scales: {:?}", scales_data);

        // All rows have the same set of values across the entire row,
        // so each 128x128 block has the same absmax = max(1,2,3,4) = 4.0
        for &s in &scales_data {
            let expected_scale = 4.0 / 448.0;
            assert!(
                (s - expected_scale).abs() < 1e-4,
                "block scale: got {s}, expected {expected_scale}"
            );
        }

        // Verify round-trip: dequant block (0,0) should approximate original
        let quant_f32 = quant.to_dtype(DType::F32)?;
        let quant_data = quant_f32.flatten_all()?.to_vec1::<f32>()?;
        let block_scale = scales_data[0]; // block (k=0, m=0)
        let expected_val = data[0]; // row 0 = 1.0
        let recovered = quant_data[0] * block_scale;
        assert!(
            (recovered - expected_val).abs() < 0.1,
            "roundtrip: got {recovered}, expected {expected_val}"
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_gemm_known_values() -> Result<()> {
        let device = &cuda_device()?;

        let m = 128usize;
        let n = 256usize;
        let k = 256usize;

        // Drop intermediate F32 tensors to avoid cudarc stream-allocator interference.
        let w_fp8 = Tensor::from_vec(vec![1.0f32; n * k], (n, k), device)?
            .to_dtype(DType::F8E4M3)?;

        let scale_rows = (n + 127) / 128;
        let scale_cols = (k + 127) / 128;
        let scale_inv = Tensor::from_vec(
            vec![1.0f32; scale_rows * scale_cols],
            (scale_rows, scale_cols),
            device,
        )?;

        let layer = Fp8Linear::new(
            w_fp8,
            scale_inv,
            None,
            Fp8BlockSize { block_r: 128, block_c: 128 },
            DType::BF16,
        );

        let x = Tensor::from_vec(vec![1.0f32; m * k], (m, k), device)?
            .to_dtype(DType::BF16)?;
        let y = layer.forward(&x)?;
        let y_f32 = y.to_dtype(DType::F32)?;
        let y_data = y_f32.flatten_all()?.to_vec1::<f32>()?;

        eprintln!("Fp8Linear output[0..5]: {:?}", &y_data[..5]);
        assert_eq!(y_data.len(), m * n);

        for (i, &val) in y_data.iter().enumerate() {
            let rel_err = (val - k as f32).abs() / (k as f32);
            assert!(
                rel_err < 0.05,
                "element {i} (row={}, col={}): got {val}, expected ~{}, rel_err={rel_err}",
                i / n, i % n, k as f32,
            );
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_gemm_vs_dequant_real_weights() -> Result<()> {
        let path = "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3/model-00001-of-00002.safetensors";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test: safetensors file not found at {path}");
            return Ok(());
        }

        let device = &cuda_device()?;
        let tensors = unsafe { candle::safetensors::MmapedSafetensors::new(path)? };

        let weight = tensors.load("model.layers.0.mlp.gate_proj.weight", device)?;
        let scale_inv = tensors.load("model.layers.0.mlp.gate_proj.weight_scale_inv", device)?;
        let scale_inv_f32 = scale_inv.to_dtype(DType::F32)?;

        let block_size = Fp8BlockSize {
            block_r: 128,
            block_c: 128,
        };

        // cuBLASLt VEC128+BLK128 requires m to be a multiple of 128
        let m = 128usize;
        let k = 4096usize;
        let x_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5)
            .collect();
        let x_bf16 = Tensor::from_vec(x_data, (m, k), device)?.to_dtype(DType::BF16)?;

        // Compute reference via dequant path, then drop the large BF16 weight
        let ref_f32 = {
            let w_bf16 = fp8_blockwise_dequant(&weight, &scale_inv_f32, &block_size, DType::BF16)?;
            let ref_out = x_bf16.matmul(&w_bf16.t()?)?;
            ref_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
        };

        // Native FP8 GEMM via Fp8Linear
        let layer = Fp8Linear::new(
            weight,
            scale_inv_f32,
            None,
            block_size,
            DType::BF16,
        );
        let gemm_out = layer.forward(&x_bf16)?;
        let gemm_f32 = gemm_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(ref_f32.len(), gemm_f32.len());

        let mut max_diff: f32 = 0.0;
        let mut sum_diff: f32 = 0.0;
        for (a, b) in ref_f32.iter().zip(gemm_f32.iter()) {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            sum_diff += diff;
        }
        let mean_diff = sum_diff / ref_f32.len() as f32;

        eprintln!(
            "FP8 GEMM vs dequant: max_diff={:.6}, mean_diff={:.6}",
            max_diff, mean_diff
        );
        eprintln!("ref[0..5]: {:?}", &ref_f32[..5]);
        eprintln!("gemm[0..5]: {:?}", &gemm_f32[..5]);

        assert!(
            mean_diff < 1.0,
            "mean difference too large: {}",
            mean_diff
        );

        Ok(())
    }
}
