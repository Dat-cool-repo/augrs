//! Low-level image operations on HWC arrays. All functions take and return
//! standard-layout arrays.

/// Define `$vis fn $name(args) -> ret` that runs `$imp` (an `#[inline(always)]`
/// function) compiled with AVX2 enabled when the CPU supports it, and the
/// baseline build otherwise. FMA is deliberately *not* enabled, so both paths
/// produce bit-identical results (seeded runs stay reproducible across CPUs).
macro_rules! avx2_dispatch {
    ($(#[$m:meta])* $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? = $imp:path) => {
        $(#[$m])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn avx2($($arg: $ty),*) $(-> $ret)? {
                    $imp($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU supports AVX2 (checked at runtime).
                    return unsafe { avx2($($arg),*) };
                }
            }
            $imp($($arg),*)
        }
    };
}
pub(crate) use avx2_dispatch;

/// Like [`avx2_dispatch!`] but with FMA enabled too, for kernels that must
/// reproduce OpenCV's AVX2 build bit for bit (GCC contracts `a - b * c` into
/// FMA there). Such kernels use `f32::mul_add` explicitly, so the baseline
/// build (libm `fmaf`) gives identical results, only slower.
macro_rules! avx2_fma_dispatch {
    ($(#[$m:meta])* $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? = $imp:path) => {
        $(#[$m])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn avx2($($arg: $ty),*) $(-> $ret)? {
                    $imp($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                    // SAFETY: the CPU supports AVX2 and FMA (checked at runtime).
                    return unsafe { avx2($($arg),*) };
                }
            }
            $imp($($arg),*)
        }
    };
}
pub(crate) use avx2_fma_dispatch;

pub mod blur;
pub mod clahe;
pub mod color;
pub mod flip;
pub mod hsv;
pub mod noise;
pub mod normalize;
pub mod remap;
pub mod resize;
pub mod simd;
pub mod warp;

pub use blur::{gaussian_blur, gaussian_kernel_1d};
pub use flip::{crop, hflip, pad, rot90, transpose, vflip};
pub use normalize::normalize;
pub use resize::{resize, resize_u8};
pub use warp::{warp_affine, warp_affine_u8};
