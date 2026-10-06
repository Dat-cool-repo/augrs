//! Hand-written AVX2 kernels for the hot u8 RGB paths (x86-64 only), selected
//! at runtime. Every kernel here is bit-identical to its scalar counterpart
//! (same integer arithmetic, same f32 operation order, explicit FMA where the
//! scalar code uses `mul_add`), so results never depend on the CPU. The unit
//! tests compare both paths on random data.

use std::sync::atomic::{AtomicBool, Ordering};

static FORCE_SCALAR: AtomicBool = AtomicBool::new(false);

/// Disable the SIMD kernels (for testing / benchmarking the scalar fallback).
#[doc(hidden)]
pub fn set_force_scalar(v: bool) {
    FORCE_SCALAR.store(v, Ordering::Relaxed);
}

/// True when the AVX2 (+FMA) kernels can be used. Setting the environment variable
/// `AUGRS_FORCE_SCALAR=1` (read once) disables them, e.g. to test the portable fallback.
#[inline]
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ENV_SCALAR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let env =
            *ENV_SCALAR.get_or_init(|| std::env::var("AUGRS_FORCE_SCALAR").is_ok_and(|v| !v.is_empty() && v != "0"));
        !env && !FORCE_SCALAR.load(Ordering::Relaxed)
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use crate::ops::hsv::HsvEdit;
    use std::arch::x86_64::*;

    /// Load 8 interleaved RGB pixels (exactly 24 bytes) as three i32x8 vectors.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load_rgb8(p: *const u8) -> (__m256i, __m256i, __m256i) {
        unsafe {
            let lo = _mm_loadu_si128(p as *const __m128i); // bytes 0..16
            let hi = _mm_loadl_epi64(p.add(16) as *const __m128i); // bytes 16..24
            let l1 = _mm_alignr_epi8(hi, lo, 12); // bytes 12..24
            let v = _mm256_set_m128i(l1, lo);
            // per 128-bit lane: pixels at byte offsets 0, 3, 6, 9
            let mr = _mm256_setr_epi8(
                0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1, 0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1,
                -1, 9, -1, -1, -1,
            );
            let mg = _mm256_add_epi8(
                mr,
                _mm256_and_si256(_mm256_set1_epi32(1), _mm256_cmpgt_epi8(mr, _mm256_set1_epi8(-1))),
            );
            let mb = _mm256_add_epi8(
                mg,
                _mm256_and_si256(_mm256_set1_epi32(1), _mm256_cmpgt_epi8(mr, _mm256_set1_epi8(-1))),
            );
            (
                _mm256_shuffle_epi8(v, mr),
                _mm256_shuffle_epi8(v, mg),
                _mm256_shuffle_epi8(v, mb),
            )
        }
    }

    /// Store three i32x8 vectors (values 0..=255) as 8 interleaved RGB pixels (exactly 24 bytes).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn store_rgb8(p: *mut u8, r: __m256i, g: __m256i, b: __m256i) {
        unsafe {
            let packed = _mm256_or_si256(r, _mm256_or_si256(_mm256_slli_epi32(g, 8), _mm256_slli_epi32(b, 16)));
            let m = _mm256_setr_epi8(
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1,
                -1, -1,
            );
            let c = _mm256_shuffle_epi8(packed, m);
            let c0 = _mm256_castsi256_si128(c);
            let c1 = _mm256_extracti128_si256(c, 1);
            let first = _mm_or_si128(c0, _mm_slli_si128(c1, 12));
            _mm_storeu_si128(p as *mut __m128i, first);
            _mm_storel_epi64(p.add(16) as *mut __m128i, _mm_srli_si128(c1, 4));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn clamp255(x: __m256i) -> __m256i {
        _mm256_min_epi32(_mm256_max_epi32(x, _mm256_setzero_si256()), _mm256_set1_epi32(255))
    }

    /// OpenCV-exact RGB -> HSV -> edit -> RGB on `px.len() / 24` groups of 8 pixels
    /// (see `ops::hsv`; this is the vector form of `rgb_to_hsv` / `hsv_to_rgb_simd`).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn hsv_edit_rgb(px: &mut [u8], edit: &HsvEdit, lut32: Option<&[i32; 256]>) {
        unsafe {
            let zero = _mm256_setzero_si256();
            let one_f = _mm256_set1_ps(1.0);
            let c255 = _mm256_set1_epi32(255);
            let round = _mm256_set1_epi32(1 << 11);
            let sat_add = _mm256_set1_epi32(edit.sat_add);
            let val_add = _mm256_set1_epi32(edit.val_add);
            let hscale = _mm256_set1_ps(6.0 / 180.0);
            let inv255 = _mm256_set1_ps(1.0 / 255.0);
            let sixth = _mm256_set1_ps(1.0 / 6.0);
            let six = _mm256_set1_ps(6.0);
            let f255 = _mm256_set1_ps(255.0);
            let (f1, f2, f3, f4) = (
                _mm256_set1_ps(1.0),
                _mm256_set1_ps(2.0),
                _mm256_set1_ps(3.0),
                _mm256_set1_ps(4.0),
            );
            for chunk in px.chunks_exact_mut(24) {
                let p = chunk.as_mut_ptr();
                let (r, g, b) = load_rgb8(p);
                let v = _mm256_max_epi32(_mm256_max_epi32(r, g), b);
                let vmin = _mm256_min_epi32(_mm256_min_epi32(r, g), b);
                let diff = _mm256_sub_epi32(v, vmin);
                let vz = _mm256_cmpeq_epi32(v, zero);
                let dz = _mm256_cmpeq_epi32(diff, zero);
                let sdiv = _mm256_cvtps_epi32(_mm256_div_ps(
                    _mm256_set1_ps(1_044_480.0),
                    _mm256_cvtepi32_ps(_mm256_max_epi32(v, _mm256_set1_epi32(1))),
                ));
                let sdiv = _mm256_andnot_si256(vz, sdiv);
                let hdiv = _mm256_cvtps_epi32(_mm256_div_ps(
                    _mm256_set1_ps(122_880.0),
                    _mm256_cvtepi32_ps(_mm256_max_epi32(diff, _mm256_set1_epi32(1))),
                ));
                let hdiv = _mm256_andnot_si256(dz, hdiv);
                let mut s = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(diff, sdiv), round), 12);
                let vr = _mm256_cmpeq_epi32(v, r);
                let vg = _mm256_cmpeq_epi32(v, g);
                let h_r = _mm256_sub_epi32(g, b);
                let h_g = _mm256_add_epi32(_mm256_sub_epi32(b, r), _mm256_slli_epi32(diff, 1));
                let h_b = _mm256_add_epi32(_mm256_sub_epi32(r, g), _mm256_slli_epi32(diff, 2));
                let h0 = _mm256_blendv_epi8(_mm256_blendv_epi8(h_b, h_g, vg), h_r, vr);
                let mut h = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(h0, hdiv), round), 12);
                h = _mm256_add_epi32(h, _mm256_and_si256(_mm256_cmpgt_epi32(zero, h), _mm256_set1_epi32(180)));
                let mut v = v;
                // edit
                if let Some(l) = lut32 {
                    h = _mm256_i32gather_epi32(l.as_ptr(), _mm256_and_si256(h, c255), 4);
                }
                if edit.sat_add != 0 {
                    let s2 = clamp255(_mm256_add_epi32(s, sat_add));
                    s = if edit.keep_gray_sat {
                        _mm256_andnot_si256(_mm256_cmpeq_epi32(s, zero), s2)
                    } else {
                        s2
                    };
                }
                if edit.val_add != 0 {
                    v = clamp255(_mm256_add_epi32(v, val_add));
                }
                // HSV -> RGB (OpenCV AVX2 kernel)
                let hf = _mm256_mul_ps(_mm256_cvtepi32_ps(h), hscale);
                let pre = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(hf));
                let fr = _mm256_sub_ps(hf, pre);
                let sf = _mm256_mul_ps(_mm256_cvtepi32_ps(s), inv255);
                let vf = _mm256_mul_ps(_mm256_cvtepi32_ps(v), inv255);
                let tab0 = vf;
                let tab1 = _mm256_mul_ps(vf, _mm256_sub_ps(one_f, sf));
                let tab2 = _mm256_mul_ps(vf, _mm256_fnmadd_ps(sf, fr, one_f));
                let tab3 = _mm256_mul_ps(vf, _mm256_fnmadd_ps(sf, _mm256_sub_ps(one_f, fr), one_f));
                let sec6 = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(_mm256_mul_ps(pre, sixth)));
                let sector = _mm256_sub_ps(pre, _mm256_mul_ps(sec6, six));
                let lt1 = _mm256_cmp_ps(sector, f1, _CMP_LT_OQ);
                let lt2 = _mm256_cmp_ps(sector, f2, _CMP_LT_OQ);
                let eq1 = _mm256_cmp_ps(sector, f1, _CMP_EQ_OQ);
                let eq2 = _mm256_cmp_ps(sector, f2, _CMP_EQ_OQ);
                let eq3 = _mm256_cmp_ps(sector, f3, _CMP_EQ_OQ);
                let eq4 = _mm256_cmp_ps(sector, f4, _CMP_EQ_OQ);
                let le2 = _mm256_cmp_ps(sector, f2, _CMP_LE_OQ);
                let le3 = _mm256_cmp_ps(sector, f3, _CMP_LE_OQ);
                let le4 = _mm256_cmp_ps(sector, f4, _CMP_LE_OQ);
                let mut bb = _mm256_blendv_ps(tab2, tab0, le4);
                bb = _mm256_blendv_ps(bb, tab3, eq2);
                bb = _mm256_blendv_ps(bb, tab1, lt2);
                let mut gg = _mm256_blendv_ps(tab1, tab2, eq3);
                gg = _mm256_blendv_ps(gg, tab0, le2);
                gg = _mm256_blendv_ps(gg, tab3, lt1);
                let mut rr = _mm256_blendv_ps(tab0, tab3, eq4);
                rr = _mm256_blendv_ps(rr, tab1, le3);
                rr = _mm256_blendv_ps(rr, tab2, eq1);
                rr = _mm256_blendv_ps(rr, tab0, lt1);
                store_rgb8(p, to_u8(rr, f255, c255), to_u8(gg, f255, c255), to_u8(bb, f255, c255));
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn to_u8(x: __m256, f255: __m256, c255: __m256i) -> __m256i {
        _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(x, f255)), c255)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sat_ch(c: __m256i, fqv: __m256i, gt: __m256i) -> __m256i {
        unsafe { clamp255(_mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(c, fqv), gt), 12)) }
    }

    /// `saturation_u8` for 8-pixel groups: blend with OpenCV's 15-bit gray in 12-bit fixed point.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn saturation_rgb(px: &mut [u8], fq: i32) {
        unsafe {
            let gq = _mm256_set1_epi32(4096 - fq);
            let fqv = _mm256_set1_epi32(fq);
            for chunk in px.chunks_exact_mut(24) {
                let p = chunk.as_mut_ptr();
                let (r, g, b) = load_rgb8(p);
                let gray = gray15(r, g, b);
                let gt = _mm256_add_epi32(_mm256_mullo_epi32(gray, gq), _mm256_set1_epi32(2048));
                store_rgb8(p, sat_ch(r, fqv, gt), sat_ch(g, fqv, gt), sat_ch(b, fqv, gt));
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gray15(r: __m256i, g: __m256i, b: __m256i) -> __m256i {
        let s = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(r, _mm256_set1_epi32(9798)),
                _mm256_mullo_epi32(g, _mm256_set1_epi32(19235)),
            ),
            _mm256_add_epi32(
                _mm256_mullo_epi32(b, _mm256_set1_epi32(3735)),
                _mm256_set1_epi32(1 << 14),
            ),
        );
        _mm256_srli_epi32(s, 15)
    }

    /// Sum of OpenCV 8-bit gray values over `px.len() / 24` groups of 8 RGB pixels.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gray_sum_rgb(px: &[u8]) -> u64 {
        unsafe {
            let mut total = 0u64;
            for block in px.chunks(24 * 4096) {
                let mut acc = _mm256_setzero_si256();
                for chunk in block.chunks_exact(24) {
                    let (r, g, b) = load_rgb8(chunk.as_ptr());
                    acc = _mm256_add_epi32(acc, gray15(r, g, b));
                }
                let mut lanes = [0i32; 8];
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
                total += lanes.iter().map(|&v| v as u64).sum::<u64>();
            }
            total
        }
    }

    /// `out[i] = src[i] as f32 * sc[i % 12] + of[i % 12]` for `src.len() / 24` blocks of 24 (mul, then add).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn affine_u8_rgb(src: &[u8], out: &mut [f32], sc: &[f32; 12], of: &[f32; 12]) {
        unsafe {
            let pat = |a: &[f32; 12], k: usize| {
                let mut v = [0f32; 8];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = a[(8 * k + j) % 12];
                }
                _mm256_loadu_ps(v.as_ptr())
            };
            let (s0, s1, s2) = (pat(sc, 0), pat(sc, 1), pat(sc, 2));
            let (o0, o1, o2) = (pat(of, 0), pat(of, 1), pat(of, 2));
            for (i, o) in src.chunks_exact(24).zip(out.chunks_exact_mut(24)) {
                let ip = i.as_ptr();
                let op = o.as_mut_ptr();
                for (k, (s, f)) in [(s0, o0), (s1, o1), (s2, o2)].into_iter().enumerate() {
                    let x = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(ip.add(8 * k) as *const __m128i)));
                    _mm256_storeu_ps(op.add(8 * k), _mm256_add_ps(_mm256_mul_ps(x, s), f));
                }
            }
        }
    }

    /// Bilinear u8 RGB warp for 8 output pixels whose 8-bit sub-pixel source positions are
    /// `x8`/`y8` (with the `OFF` bias of `ops::warp`). Returns false (writing nothing) unless all 4 taps of
    /// all 8 pixels are inside the image, so the caller can fall back to the scalar border code.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn warp_bilinear_rgb8(src: &[u8], w: usize, h: usize, x8: &[i32], y8: &[i32], out: *mut u8) -> bool {
        unsafe {
            let stride = (w * 3) as i32;
            let xv = _mm256_loadu_si256(x8.as_ptr() as *const __m256i);
            let yv = _mm256_loadu_si256(y8.as_ptr() as *const __m256i);
            let off = _mm256_set1_epi32(crate::ops::warp::OFF_I as i32);
            let x0 = _mm256_sub_epi32(_mm256_srai_epi32(xv, 8), off);
            let y0 = _mm256_sub_epi32(_mm256_srai_epi32(yv, 8), off);
            let m1 = _mm256_set1_epi32(-1);
            let inside = _mm256_and_si256(
                _mm256_and_si256(
                    _mm256_cmpgt_epi32(x0, m1),
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(w as i32 - 1), x0),
                ),
                _mm256_and_si256(
                    _mm256_cmpgt_epi32(y0, m1),
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(h as i32 - 1), y0),
                ),
            );
            let i00 = _mm256_add_epi32(
                _mm256_mullo_epi32(y0, _mm256_set1_epi32(stride)),
                _mm256_mullo_epi32(x0, _mm256_set1_epi32(3)),
            );
            // the 4-byte gather at i00 + stride + 3 must stay inside the buffer
            let lim = src.len() as i64 - stride as i64 - 7;
            if lim < 0 {
                return false;
            }
            let safe = _mm256_cmpgt_epi32(_mm256_set1_epi32(lim.min(i32::MAX as i64) as i32), i00);
            if _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(inside, safe))) != 0xff {
                return false;
            }
            let base = src.as_ptr() as *const i32;
            let p00 = _mm256_i32gather_epi32(base, i00, 1);
            let p01 = _mm256_i32gather_epi32(base, _mm256_add_epi32(i00, _mm256_set1_epi32(3)), 1);
            let i10 = _mm256_add_epi32(i00, _mm256_set1_epi32(stride));
            let p10 = _mm256_i32gather_epi32(base, i10, 1);
            let p11 = _mm256_i32gather_epi32(base, _mm256_add_epi32(i10, _mm256_set1_epi32(3)), 1);
            let c256 = _mm256_set1_epi32(256);
            let fx = _mm256_and_si256(xv, _mm256_set1_epi32(255));
            let fy = _mm256_and_si256(yv, _mm256_set1_epi32(255));
            let (gx, gy) = (_mm256_sub_epi32(c256, fx), _mm256_sub_epi32(c256, fy));
            let w00 = _mm256_mullo_epi32(gx, gy);
            let w01 = _mm256_mullo_epi32(fx, gy);
            let w10 = _mm256_mullo_epi32(gx, fy);
            let w11 = _mm256_mullo_epi32(fx, fy);
            let p = [p00, p01, p10, p11];
            let wv = [w00, w01, w10, w11];
            store_rgb8(out, warp_ch(&p, &wv, 0), warp_ch(&p, &wv, 8), warp_ch(&p, &wv, 16));
            true
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn warp_ch(p: &[__m256i; 4], w: &[__m256i; 4], sh: i32) -> __m256i {
        let s = _mm_cvtsi32_si128(sh);
        let m = _mm256_set1_epi32(255);
        let c0 = _mm256_and_si256(_mm256_srl_epi32(p[0], s), m);
        let c1 = _mm256_and_si256(_mm256_srl_epi32(p[1], s), m);
        let c2 = _mm256_and_si256(_mm256_srl_epi32(p[2], s), m);
        let c3 = _mm256_and_si256(_mm256_srl_epi32(p[3], s), m);
        let acc = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(c0, w[0]), _mm256_mullo_epi32(c1, w[1])),
            _mm256_add_epi32(_mm256_mullo_epi32(c2, w[2]), _mm256_mullo_epi32(c3, w[3])),
        );
        _mm256_srli_epi32(_mm256_add_epi32(acc, _mm256_set1_epi32(1 << 15)), 16)
    }

    /// Horizontal linear pass of the RGB resize for 8 outputs: planar i32 results
    /// `row[o0] * w0 + row[o1] * w1` per channel. `o0`/`o1` are byte offsets into `src`.
    /// Returns false when a 4-byte read could leave `src`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn hres_rgb8(src: &[u8], o0: &[i32], o1: &[i32], w0: &[i32], w1: &[i32], out: [&mut [i32]; 3]) -> bool {
        unsafe {
            let a = _mm256_loadu_si256(o0.as_ptr() as *const __m256i);
            let b = _mm256_loadu_si256(o1.as_ptr() as *const __m256i);
            let lim = src.len() as i64 - 4;
            let mx = o1[7].max(o0[7]).max(o1[0]).max(o0[0]) as i64;
            if mx > lim {
                return false;
            }
            let base = src.as_ptr() as *const i32;
            let p0 = _mm256_i32gather_epi32(base, a, 1);
            let p1 = _mm256_i32gather_epi32(base, b, 1);
            let wa = _mm256_loadu_si256(w0.as_ptr() as *const __m256i);
            let wb = _mm256_loadu_si256(w1.as_ptr() as *const __m256i);
            let m = _mm256_set1_epi32(255);
            let [orow, og, ob] = out;
            for (k, o) in [orow, og, ob].into_iter().enumerate() {
                let s = _mm_cvtsi32_si128(8 * k as i32);
                let c0 = _mm256_and_si256(_mm256_srl_epi32(p0, s), m);
                let c1 = _mm256_and_si256(_mm256_srl_epi32(p1, s), m);
                let v = _mm256_add_epi32(_mm256_mullo_epi32(c0, wa), _mm256_mullo_epi32(c1, wb));
                _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, v);
            }
            true
        }
    }

    /// Vertical pass of the RGB resize (OpenCV `VResizeLinearVec_32s8u` arithmetic) on planar
    /// rows, interleaving into `out` (8 pixels per step; `out.len() == 3 * n`, `n % 8 == 0`).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vres_rgb(a: [&[i32]; 3], bb: [&[i32]; 3], b0: i32, b1: i32, out: &mut [u8]) {
        unsafe {
            let n = out.len() / 3;
            let (vb0, vb1) = (_mm256_set1_epi32(b0), _mm256_set1_epi32(b1));
            let two = _mm256_set1_epi32(2);
            let mut i = 0;
            while i + 8 <= n {
                let mut c = [_mm256_setzero_si256(); 3];
                for k in 0..3 {
                    let p0 = _mm256_srai_epi32(_mm256_loadu_si256(a[k].as_ptr().add(i) as *const __m256i), 4);
                    let p1 = _mm256_srai_epi32(_mm256_loadu_si256(bb[k].as_ptr().add(i) as *const __m256i), 4);
                    let t0 = _mm256_srai_epi32(_mm256_mullo_epi32(vb0, p0), 16);
                    let t1 = _mm256_srai_epi32(_mm256_mullo_epi32(vb1, p1), 16);
                    c[k] = clamp255(_mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(t0, t1), two), 2));
                }
                store_rgb8(out.as_mut_ptr().add(3 * i), c[0], c[1], c[2]);
                i += 8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::BorderMode;
    use crate::geometry::Affine2;
    use crate::ops::resize::{Interp, Rect};
    use crate::rng::Rng;
    use ndarray::Array3;
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());

    fn rand_img(rng: &mut Rng, h: usize, w: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, 3), |_| rng.int_inclusive(0, 255) as u8)
    }

    /// Run `f` with SIMD on and off; both results must be identical.
    fn both<T: PartialEq + std::fmt::Debug>(f: impl Fn() -> T) {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !avx2_available() {
            return; // nothing to compare on this CPU
        }
        let a = f();
        set_force_scalar(true);
        let b = f();
        set_force_scalar(false);
        assert_eq!(a, b);
    }

    #[test]
    fn simd_matches_scalar() {
        let mut rng = Rng::seed_from_u64(5);
        for (h, w) in [(7, 70), (33, 64), (5, 131), (1, 8), (16, 37)] {
            let img = rand_img(&mut rng, h, w);
            for (hs, sa, va) in [(0.0, 0, 0), (13.7, -20, 9), (-170.2, 40, -60), (180.0, 0, 0)] {
                both(|| {
                    let mut a = img.clone();
                    let e = crate::ops::hsv::HsvEdit {
                        hue_lut: if hs != 0.0 {
                            Some(crate::ops::hsv::hue_lut(hs))
                        } else {
                            None
                        },
                        sat_add: sa,
                        val_add: va,
                        keep_gray_sat: true,
                    };
                    crate::ops::hsv::hsv_edit_u8(&mut a, &e);
                    a
                });
            }
            for f in [0.0, 0.3, 1.7] {
                both(|| {
                    let mut a = img.clone();
                    crate::ops::color::saturation_u8(&mut a, f);
                    crate::ops::color::contrast_u8(&mut a, f + 0.2);
                    a
                });
            }
            both(|| {
                crate::ops::normalize(&crate::Buf::U8(img.clone()), &[0.4, 0.5, 0.6], &[0.2, 0.3, 0.25], 255.0).unwrap()
            });
            for (oh, ow) in [(h * 2 + 3, w * 3 / 2 + 1), (h.max(2) / 2, w / 3 + 1), (64, 64)] {
                both(|| crate::ops::resize_u8(&img, Rect::full(h, w), oh, ow, Interp::Linear));
                if h > 4 && w > 4 {
                    both(|| {
                        crate::ops::resize_u8(
                            &img,
                            Rect {
                                x0: 1,
                                y0: 2,
                                x1: w - 1,
                                y1: h - 1,
                            },
                            oh,
                            ow,
                            Interp::Linear,
                        )
                    });
                }
            }
            for (ang, sc) in [(17.0, 1.1), (-63.0, 0.7), (90.0, 1.0), (3.0, 2.5)] {
                let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
                let m = Affine2::translate(-cx, -cy)
                    .then(&Affine2::scale(sc, sc * 0.9))
                    .then(&Affine2::rotate_deg(ang))
                    .then(&Affine2::translate(cx + 1.3, cy - 0.7));
                let inv = m.inverse().unwrap();
                for mode in [BorderMode::Constant, BorderMode::Reflect101, BorderMode::Wrap] {
                    both(|| {
                        crate::ops::warp_affine_u8(&img, &inv, h + 3, w + 5, Interp::Linear, mode, &[3.0, 4.0, 5.0])
                    });
                }
            }
        }
    }
}
