//! Affine warping by inverse mapping, with border modes.
//!
//! u8 images use 8-bit sub-pixel fixed-point bilinear weights (OpenCV uses 5
//! bits); other element types use f32 weights. Coordinates are floored with an
//! offset trick instead of `f64::floor` (which is a libm call on baseline x86-64).

use crate::border::{BorderMode, border_index};
use crate::buffer::{Element, data, from_vec};
use crate::geometry::Affine2;
use crate::ops::resize::Interp;
use ndarray::Array3;

const OFF: f64 = 65536.0;
const OFF_I: i64 = 65536;
/// coordinates are clamped to this range before flooring (keeps `x + OFF > 0`)
const LIM: f64 = 60000.0;
const WB: i64 = 8;
const WS: i64 = 1 << WB;

fn setup(
    dims: (usize, usize, usize),
    inv: &Affine2,
    oh: usize,
    ow: usize,
    mode: BorderMode,
    fill: &[f64],
) -> (WarpArgs, Vec<f32>) {
    let (h, w, c) = dims;
    let fillv: Vec<f32> = (0..c)
        .map(|k| *fill.get(k).or(fill.last()).unwrap_or(&0.0) as f32)
        .collect();
    // index-space coefficients: src_idx = inv(u + .5, v + .5) - .5
    let k = [
        inv.a,
        inv.b,
        inv.a * 0.5 + inv.b * 0.5 + inv.c - 0.5,
        inv.d,
        inv.e,
        inv.d * 0.5 + inv.e * 0.5 + inv.f - 0.5,
    ];
    (WarpArgs { h, w, k, oh, ow, mode }, fillv)
}

super::avx2_dispatch!(
    /// u8 warp: integer bilinear kernel (or nearest). See [`warp_affine`].
    pub fn warp_affine_u8(
        a: &Array3<u8>,
        inv: &Affine2,
        oh: usize,
        ow: usize,
        interp: Interp,
        mode: BorderMode,
        fill: &[f64],
    ) -> Array3<u8> = warp_affine_u8_impl
);

#[inline(always)]
fn warp_affine_u8_impl(
    a: &Array3<u8>,
    inv: &Affine2,
    oh: usize,
    ow: usize,
    interp: Interp,
    mode: BorderMode,
    fill: &[f64],
) -> Array3<u8> {
    let c = a.dim().2;
    let (args, fillv) = setup(a.dim(), inv, oh, ow, mode, fill);
    let src = data(a);
    if interp == Interp::Nearest {
        let out = match c {
            1 => warp_nearest::<u8, 1>(src, c, &args, &fillv),
            3 => warp_nearest::<u8, 3>(src, c, &args, &fillv),
            _ => warp_nearest::<u8, 0>(src, c, &args, &fillv),
        };
        return from_vec(oh, ow, c, out);
    }
    let fill_u8: Vec<u8> = fillv.iter().map(|&v| u8::from_f32(v)).collect();
    let out = match c {
        1 => warp_bilinear_u8::<1>(src, c, &args, &fill_u8),
        3 => warp_bilinear_u8::<3>(src, c, &args, &fill_u8),
        4 => warp_bilinear_u8::<4>(src, c, &args, &fill_u8),
        _ => warp_bilinear_u8::<0>(src, c, &args, &fill_u8),
    };
    from_vec(oh, ow, c, out)
}

/// Warp `a` into an `oh x ow` image. `inv` maps *output* continuous coordinates
/// to *input* continuous coordinates (pixel centres at `i + 0.5`).
/// `Interp::Nearest` samples the nearest pixel centre; every other value uses
/// bilinear interpolation (f32 weights; see [`warp_affine_u8`] for the faster
/// u8 kernel). `fill` gives the constant-border value per channel (a single
/// value is broadcast).
pub fn warp_affine<T: Element>(
    a: &Array3<T>,
    inv: &Affine2,
    oh: usize,
    ow: usize,
    interp: Interp,
    mode: BorderMode,
    fill: &[f64],
) -> Array3<T> {
    let c = a.dim().2;
    let (args, fillv) = setup(a.dim(), inv, oh, ow, mode, fill);
    let nearest = interp == Interp::Nearest;
    let src = data(a);
    let out = match (nearest, c) {
        (true, 1) => warp_nearest::<T, 1>(src, c, &args, &fillv),
        (true, 3) => warp_nearest::<T, 3>(src, c, &args, &fillv),
        (true, _) => warp_nearest::<T, 0>(src, c, &args, &fillv),
        (false, 1) => warp_bilinear_f::<T, 1>(src, c, &args, &fillv),
        (false, 3) => warp_bilinear_f::<T, 3>(src, c, &args, &fillv),
        (false, _) => warp_bilinear_f::<T, 0>(src, c, &args, &fillv),
    };
    from_vec(oh, ow, c, out)
}

struct WarpArgs {
    h: usize,
    w: usize,
    k: [f64; 6],
    oh: usize,
    ow: usize,
    mode: BorderMode,
}

/// floor(x) for |x| < LIM without a libm call.
#[inline(always)]
fn ifloor(x: f64) -> i64 {
    (x.clamp(-LIM, LIM) + OFF) as i64 - OFF_I
}

/// 32.32 fixed point helpers for integer coordinate stepping along a row.
const FB: u32 = 32;

#[inline(always)]
fn to_fix(x: f64) -> i64 {
    (x * (1u64 << FB) as f64).round() as i64
}

/// Per-row fixed-point start/step for x and y (with the `OFF` bias added).
struct RowFix {
    x0: i64,
    dx: i64,
    y0: i64,
    dy: i64,
    lo: i64,
    hi: i64,
}

impl RowFix {
    #[inline(always)]
    fn new(k: &[f64; 6], v: usize, bias: f64) -> RowFix {
        let vf = v as f64;
        RowFix {
            x0: to_fix(k[1] * vf + k[2] + OFF + bias),
            dx: to_fix(k[0]),
            y0: to_fix(k[4] * vf + k[5] + OFF + bias),
            dy: to_fix(k[3]),
            lo: to_fix(OFF - LIM),
            hi: to_fix(OFF + LIM),
        }
    }
    #[inline(always)]
    fn at(&self, u: usize) -> (i64, i64) {
        let u = u as i64;
        (
            (self.x0 + u * self.dx).clamp(self.lo, self.hi),
            (self.y0 + u * self.dy).clamp(self.lo, self.hi),
        )
    }
}

#[inline(always)]
fn warp_nearest<T: Element, const C: usize>(src: &[T], cdyn: usize, a: &WarpArgs, fillv: &[f32]) -> Vec<T> {
    let c = if C == 0 { cdyn } else { C };
    let (h, w) = (a.h, a.w);
    let mut out = vec![T::default(); a.oh * a.ow * c];
    let fill_t: Vec<T> = fillv.iter().map(|&v| T::from_f32(v)).collect();
    for v in 0..a.oh {
        // +0.5 then floor == nearest pixel centre
        let rf = RowFix::new(&a.k, v, 0.5);
        let orow = &mut out[v * a.ow * c..(v + 1) * a.ow * c];
        for (u, o) in orow.chunks_exact_mut(c).enumerate() {
            let (xf, yf) = rf.at(u);
            let xi = (xf >> FB) - OFF_I;
            let yi = (yf >> FB) - OFF_I;
            if (xi as u64) < w as u64 && (yi as u64) < h as u64 {
                let i = (yi as usize * w + xi as usize) * c;
                o.copy_from_slice(&src[i..i + c]);
            } else {
                match (
                    border_index(xi as isize, w, a.mode),
                    border_index(yi as isize, h, a.mode),
                ) {
                    (Some(x), Some(y)) => {
                        let i = (y * w + x) * c;
                        o.copy_from_slice(&src[i..i + c]);
                    }
                    _ => o.copy_from_slice(&fill_t),
                }
            }
        }
    }
    out
}

/// One bilinear u8 output pixel from its 8-bit sub-pixel source position (`x8`, `y8`
/// include the `OFF` bias). Shared by the scalar loop and the SIMD fallback.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn bilinear_px_u8(
    src: &[u8],
    w: usize,
    h: usize,
    c: usize,
    x8: i64,
    y8: i64,
    mode: BorderMode,
    fill: &[u8],
    o: &mut [u8],
) {
    let stride = w * c;
    let x0 = (x8 >> WB) - OFF_I;
    let y0 = (y8 >> WB) - OFF_I;
    let fx = (x8 & (WS - 1)) as i32;
    let fy = (y8 & (WS - 1)) as i32;
    let ws_i = WS as i32;
    let w00 = (ws_i - fx) * (ws_i - fy);
    let w01 = fx * (ws_i - fy);
    let w10 = (ws_i - fx) * fy;
    let w11 = fx * fy;
    if (x0 as u64) < w as u64 - 1 && (y0 as u64) < h as u64 - 1 {
        let i00 = y0 as usize * stride + x0 as usize * c;
        for ch in 0..c {
            let s = src[i00 + ch] as i32 * w00
                + src[i00 + c + ch] as i32 * w01
                + src[i00 + stride + ch] as i32 * w10
                + src[i00 + stride + c + ch] as i32 * w11;
            o[ch] = ((s + (1 << 15)) >> 16) as u8;
        }
        return;
    }
    if mode == BorderMode::Constant && (x0 < -1 || y0 < -1 || x0 >= w as i64 || y0 >= h as i64) {
        o.copy_from_slice(fill);
        return;
    }
    let xs = [
        border_index(x0 as isize, w, mode),
        border_index(x0 as isize + 1, w, mode),
    ];
    let ys = [
        border_index(y0 as isize, h, mode),
        border_index(y0 as isize + 1, h, mode),
    ];
    for ch in 0..c {
        let get = |yy: Option<usize>, xx: Option<usize>| match (yy, xx) {
            (Some(y), Some(x)) => src[y * stride + x * c + ch] as i32,
            _ => fill[ch] as i32,
        };
        let s = get(ys[0], xs[0]) * w00 + get(ys[0], xs[1]) * w01 + get(ys[1], xs[0]) * w10 + get(ys[1], xs[1]) * w11;
        o[ch] = ((s + (1 << 15)) >> 16) as u8;
    }
}

#[inline(always)]
fn warp_bilinear_u8<const C: usize>(src: &[u8], cdyn: usize, a: &WarpArgs, fill: &[u8]) -> Vec<u8> {
    let c = if C == 0 { cdyn } else { C };
    let (h, w) = (a.h, a.w);
    let mut out = vec![0u8; a.oh * a.ow * c];
    const ROUND8: i64 = 1 << (FB as i64 - WB - 1);
    #[cfg(target_arch = "x86_64")]
    let simd = c == 3 && h >= 2 && w >= 2 && super::simd::avx2_available();
    #[cfg(target_arch = "x86_64")]
    let (mut x32, mut y32) = (vec![0i32; a.ow], vec![0i32; a.ow]);
    for v in 0..a.oh {
        let rf = RowFix::new(&a.k, v, 0.0);
        let orow = &mut out[v * a.ow * c..(v + 1) * a.ow * c];
        #[cfg(target_arch = "x86_64")]
        if simd {
            for (u, (x, y)) in x32.iter_mut().zip(y32.iter_mut()).enumerate() {
                let (xf, yf) = rf.at(u);
                // round to the nearest 1/256 px (fits in i32: |coord| <= 60000 px plus the bias)
                *x = ((xf + ROUND8) >> (FB as i64 - WB)) as i32;
                *y = ((yf + ROUND8) >> (FB as i64 - WB)) as i32;
            }
            let mut u = 0;
            while u < a.ow {
                let n = (a.ow - u).min(8);
                // SAFETY: AVX2 is available; the kernel only gathers inside `src` and writes 24 bytes.
                let ok = n == 8
                    && unsafe {
                        super::simd::x86::warp_bilinear_rgb8(
                            src,
                            w,
                            h,
                            &x32[u..u + 8],
                            &y32[u..u + 8],
                            orow.as_mut_ptr().add(u * 3),
                        )
                    };
                if !ok {
                    for uu in u..u + n {
                        let o = &mut orow[uu * 3..uu * 3 + 3];
                        bilinear_px_u8(src, w, h, 3, x32[uu] as i64, y32[uu] as i64, a.mode, fill, o);
                    }
                }
                u += n;
            }
            continue;
        }
        for (u, o) in orow.chunks_exact_mut(c).enumerate() {
            let (xf, yf) = rf.at(u);
            // round to the nearest 1/256 px
            let x8 = (xf + ROUND8) >> (FB as i64 - WB);
            let y8 = (yf + ROUND8) >> (FB as i64 - WB);
            bilinear_px_u8(src, w, h, c, x8, y8, a.mode, fill, o);
        }
    }
    out
}

fn warp_bilinear_f<T: Element, const C: usize>(src: &[T], cdyn: usize, a: &WarpArgs, fillv: &[f32]) -> Vec<T> {
    let c = if C == 0 { cdyn } else { C };
    let (h, w, k) = (a.h, a.w, a.k);
    let (wi, hi) = (w as i64, h as i64);
    let mut out = vec![T::default(); a.oh * a.ow * c];
    let fill_t: Vec<T> = fillv.iter().map(|&v| T::from_f32(v)).collect();
    for v in 0..a.oh {
        let vf = v as f64;
        let bx = k[1] * vf + k[2];
        let by = k[4] * vf + k[5];
        let orow = &mut out[v * a.ow * c..(v + 1) * a.ow * c];
        for (u, o) in orow.chunks_exact_mut(c).enumerate() {
            let uf = u as f64;
            let sx = k[0] * uf + bx;
            let sy = k[3] * uf + by;
            let x0 = ifloor(sx);
            let y0 = ifloor(sy);
            let fx = (sx.clamp(-LIM, LIM) - x0 as f64) as f32;
            let fy = (sy.clamp(-LIM, LIM) - y0 as f64) as f32;
            if x0 >= 0 && y0 >= 0 && x0 + 1 < wi && y0 + 1 < hi {
                let i00 = (y0 as usize * w + x0 as usize) * c;
                let i10 = i00 + w * c;
                for ch in 0..c {
                    let p00 = src[i00 + ch].to_f32();
                    let p01 = src[i00 + c + ch].to_f32();
                    let p10 = src[i10 + ch].to_f32();
                    let p11 = src[i10 + c + ch].to_f32();
                    let top = p00 + (p01 - p00) * fx;
                    let bot = p10 + (p11 - p10) * fx;
                    o[ch] = T::from_f32(top + (bot - top) * fy);
                }
            } else {
                if a.mode == BorderMode::Constant && (x0 + 1 < 0 || y0 + 1 < 0 || x0 >= wi || y0 >= hi) {
                    o.copy_from_slice(&fill_t);
                    continue;
                }
                let xs = [
                    border_index(x0 as isize, w, a.mode),
                    border_index(x0 as isize + 1, w, a.mode),
                ];
                let ys = [
                    border_index(y0 as isize, h, a.mode),
                    border_index(y0 as isize + 1, h, a.mode),
                ];
                for ch in 0..c {
                    let get = |yy: Option<usize>, xx: Option<usize>| match (yy, xx) {
                        (Some(y), Some(x)) => src[(y * w + x) * c + ch].to_f32(),
                        _ => fillv[ch],
                    };
                    let p00 = get(ys[0], xs[0]);
                    let p01 = get(ys[0], xs[1]);
                    let p10 = get(ys[1], xs[0]);
                    let p11 = get(ys[1], xs[1]);
                    let top = p00 + (p01 - p00) * fx;
                    let bot = p10 + (p11 - p10) * fx;
                    o[ch] = T::from_f32(top + (bot - top) * fy);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_warp_is_exact() {
        let a = Array3::from_shape_fn((9, 11, 3), |(y, x, k)| (y * 17 + x * 5 + k) as u8);
        for interp in [Interp::Nearest, Interp::Linear] {
            let o = warp_affine_u8(&a, &Affine2::IDENTITY, 9, 11, interp, BorderMode::Constant, &[0.0]);
            assert_eq!(o, a);
            let f = a.mapv(|v| v as f32);
            let of = warp_affine(&f, &Affine2::IDENTITY, 9, 11, interp, BorderMode::Constant, &[0.0]);
            assert_eq!(of, f);
        }
    }

    #[test]
    fn hflip_via_warp_matches_hflip() {
        let a = Array3::from_shape_fn((6, 10, 3), |(y, x, k)| (y * 17 + x * 5 + k) as u8);
        let m = Affine2::new(-1.0, 0.0, 10.0, 0.0, 1.0, 0.0);
        let inv = m.inverse().unwrap();
        let o = warp_affine_u8(&a, &inv, 6, 10, Interp::Linear, BorderMode::Constant, &[0.0]);
        assert_eq!(o, crate::ops::hflip(&a));
        let f = a.mapv(|v| v as f32);
        let of = warp_affine(&f, &inv, 6, 10, Interp::Linear, BorderMode::Constant, &[0.0]);
        assert_eq!(of, crate::ops::hflip(&f));
    }

    #[test]
    fn rotate_180_matches_double_flip() {
        let a = Array3::from_shape_fn((6, 10, 1), |(y, x, _)| (y * 17 + x * 5) as u8);
        let m = Affine2::translate(-5.0, -3.0)
            .then(&Affine2::rotate_deg(180.0))
            .then(&Affine2::translate(5.0, 3.0));
        let o = warp_affine_u8(
            &a,
            &m.inverse().unwrap(),
            6,
            10,
            Interp::Linear,
            BorderMode::Constant,
            &[0.0],
        );
        assert_eq!(o, crate::ops::vflip(&crate::ops::hflip(&a)));
    }

    #[test]
    fn translate_with_constant_fill() {
        let a = Array3::from_elem((4, 4, 1), 100u8);
        let inv = Affine2::translate(2.0, 0.0).inverse().unwrap();
        let o = warp_affine_u8(&a, &inv, 4, 4, Interp::Linear, BorderMode::Constant, &[7.0]);
        assert_eq!(o[[0, 0, 0]], 7);
        assert_eq!(o[[0, 1, 0]], 7);
        assert_eq!(o[[0, 2, 0]], 100);
        let r = warp_affine_u8(&a, &inv, 4, 4, Interp::Linear, BorderMode::Replicate, &[7.0]);
        assert!(r.iter().all(|&v| v == 100));
    }

    #[test]
    fn u8_and_f32_paths_agree() {
        let a = Array3::from_shape_fn((20, 30, 3), |(y, x, k)| ((y * 37 + x * 11 + k * 80) % 256) as u8);
        let m = Affine2::translate(-15.0, -10.0)
            .then(&Affine2::rotate_deg(23.0))
            .then(&Affine2::scale(1.2, 0.9))
            .then(&Affine2::translate(14.0, 11.0));
        let inv = m.inverse().unwrap();
        for mode in [BorderMode::Constant, BorderMode::Reflect101, BorderMode::Wrap] {
            let o8 = warp_affine_u8(&a, &inv, 20, 30, Interp::Linear, mode, &[0.0]);
            let of = warp_affine(&a.mapv(|v| v as f32), &inv, 20, 30, Interp::Linear, mode, &[0.0]);
            for (x, y) in o8.iter().zip(of.iter()) {
                assert!((*x as f32 - y).abs() <= 1.01, "{x} vs {y}");
            }
        }
    }
}
