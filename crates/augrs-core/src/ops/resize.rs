//! Resizing (optionally from a crop window, so crop+resize needs no copy).
//!
//! * `Nearest` picks the pixel whose centre is closest (OpenCV `INTER_NEAREST_EXACT`).
//! * `Linear` is 2-tap bilinear with half-pixel centres and *no* antialiasing,
//!   i.e. the same sampling as OpenCV `INTER_LINEAR` (u8 uses the same 11-bit
//!   fixed-point weights, so results match OpenCV within +-1).
//! * `Cubic`, `Area`, `Lanczos` on u8 images with 1-4 channels use
//!   `fast_image_resize` convolution (SIMD, antialiased). Other cases fall back
//!   to `Linear`.

use crate::buffer::{Element, data, from_vec};
use fast_image_resize as fr;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Interp {
    Nearest,
    #[default]
    Linear,
    Cubic,
    Area,
    Lanczos,
}

/// A crop window `[x0, x1) x [y0, y1)` in source pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Rect {
    pub fn full(h: usize, w: usize) -> Rect {
        Rect {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        }
    }
    pub fn w(&self) -> usize {
        self.x1 - self.x0
    }
    pub fn h(&self) -> usize {
        self.y1 - self.y0
    }
}

const COEF_BITS: i32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32;

/// Linear sampling positions along one axis: (index0, index1, weight1) with
/// indices relative to the start of the window.
fn lin_axis(src_len: usize, dst_len: usize) -> Vec<(usize, usize, f32)> {
    // computed exactly like OpenCV (1 / inv_scale) so sample positions match bit-for-bit
    let scale = 1.0 / (dst_len as f64 / src_len as f64);
    (0..dst_len)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let mut s = f.floor();
            let mut fx = f - s;
            if s < 0.0 {
                s = 0.0;
                fx = 0.0;
            }
            if s >= (src_len - 1) as f32 {
                s = (src_len - 1) as f32;
                fx = 0.0;
            }
            let i0 = s as usize;
            let i1 = (i0 + 1).min(src_len - 1);
            (i0, i1, fx)
        })
        .collect()
}

fn nearest_axis(src_len: usize, dst_len: usize) -> Vec<usize> {
    let scale = src_len as f64 / dst_len as f64;
    (0..dst_len)
        .map(|d| (((d as f64 + 0.5) * scale).floor() as usize).min(src_len - 1))
        .collect()
}

/// Generic resize (any element type) for the `Nearest`/`Linear` kernels;
/// other kernels fall back to `Linear`.
pub fn resize<T: Element>(a: &Array3<T>, rect: Rect, dh: usize, dw: usize, interp: Interp) -> Array3<T> {
    match interp {
        Interp::Nearest => resize_nearest(a, rect, dh, dw),
        _ => resize_linear_generic(a, rect, dh, dw),
    }
}

super::avx2_dispatch!(
    /// u8 resize, with fast paths.
    pub fn resize_u8(a: &Array3<u8>, rect: Rect, dh: usize, dw: usize, interp: Interp) -> Array3<u8> = resize_u8_impl
);

#[inline(always)]
fn resize_u8_impl(a: &Array3<u8>, rect: Rect, dh: usize, dw: usize, interp: Interp) -> Array3<u8> {
    let c = a.dim().2;
    match interp {
        Interp::Nearest => resize_nearest(a, rect, dh, dw),
        Interp::Linear => match c {
            1 => resize_linear_u8::<1>(a, rect, dh, dw),
            // planar layout for the AVX2 kernels; the interleaved loop is faster in scalar code
            // (both compute the same values)
            3 if super::simd::avx2_available() => resize_linear_rgb(a, rect, dh, dw),
            3 => resize_linear_u8::<3>(a, rect, dh, dw),
            4 => resize_linear_u8::<4>(a, rect, dh, dw),
            _ => resize_linear_u8::<0>(a, rect, dh, dw),
        },
        Interp::Cubic | Interp::Area | Interp::Lanczos => {
            resize_fir(a, rect, dh, dw, interp).unwrap_or_else(|| resize_linear_u8::<0>(a, rect, dh, dw))
        }
    }
}

#[inline(always)]
fn resize_nearest<T: Element>(a: &Array3<T>, rect: Rect, dh: usize, dw: usize) -> Array3<T> {
    let (_, w, c) = a.dim();
    let src = data(a);
    let xs: Vec<usize> = nearest_axis(rect.w(), dw)
        .into_iter()
        .map(|x| (x + rect.x0) * c)
        .collect();
    let ys = nearest_axis(rect.h(), dh);
    let mut out = Vec::with_capacity(dh * dw * c);
    for &sy in &ys {
        let row = &src[(sy + rect.y0) * w * c..(sy + rect.y0 + 1) * w * c];
        if c == 1 {
            out.extend(xs.iter().map(|&x| row[x]));
        } else {
            for &x in &xs {
                out.extend_from_slice(&row[x..x + c]);
            }
        }
    }
    from_vec(dh, dw, c, out)
}

#[inline(always)]
fn resize_linear_u8<const C: usize>(a: &Array3<u8>, rect: Rect, dh: usize, dw: usize) -> Array3<u8> {
    let (_, w, cdyn) = a.dim();
    let c = if C == 0 { cdyn } else { C };
    let src = data(a);
    let xa = lin_axis(rect.w(), dw);
    let ya = lin_axis(rect.h(), dh);
    let mut xo0 = Vec::with_capacity(dw);
    let mut xo1 = Vec::with_capacity(dw);
    let mut xw0 = Vec::with_capacity(dw);
    let mut xw1 = Vec::with_capacity(dw);
    for &(i0, i1, fx) in &xa {
        xo0.push((i0 + rect.x0) * c);
        xo1.push((i1 + rect.x0) * c);
        xw0.push(((1.0 - fx) * COEF_SCALE).round_ties_even() as i32);
        xw1.push((fx * COEF_SCALE).round_ties_even() as i32);
    }
    let rowlen = dw * c;
    let hres = |sy: usize, buf: &mut [i32]| {
        let row = &src[sy * w * c..(sy + 1) * w * c];
        for dx in 0..dw {
            let (o0, o1, w0, w1) = (xo0[dx], xo1[dx], xw0[dx], xw1[dx]);
            let ob = &mut buf[dx * c..dx * c + c];
            for k in 0..c {
                ob[k] = row[o0 + k] as i32 * w0 + row[o1 + k] as i32 * w1;
            }
        }
    };
    let mut ba = vec![0i32; rowlen];
    let mut bb = vec![0i32; rowlen];
    let (mut ra, mut rb) = (usize::MAX, usize::MAX);
    let mut out = vec![0u8; dh * rowlen];
    for (dy, &(i0, i1, fy)) in ya.iter().enumerate() {
        let (r0, r1) = (i0 + rect.y0, i1 + rect.y0);
        if ra != r0 {
            if rb == r0 {
                std::mem::swap(&mut ba, &mut bb);
                std::mem::swap(&mut ra, &mut rb);
            } else {
                hres(r0, &mut ba);
                ra = r0;
            }
        }
        if rb != r1 {
            hres(r1, &mut bb);
            rb = r1;
        }
        let b0 = ((1.0 - fy) * COEF_SCALE).round_ties_even() as i32;
        let b1 = (fy * COEF_SCALE).round_ties_even() as i32;
        let orow = &mut out[dy * rowlen..(dy + 1) * rowlen];
        // Same arithmetic as OpenCV's SIMD vertical pass (VResizeLinearVec_32s8u):
        // ((b0 * (S0 >> 4)) >> 16) + ((b1 * (S1 >> 4)) >> 16) + 2) >> 2
        for ((o, &p0), &p1) in orow.iter_mut().zip(ba.iter()).zip(bb.iter()) {
            let v = (((b0 * (p0 >> 4)) >> 16) + ((b1 * (p1 >> 4)) >> 16) + 2) >> 2;
            *o = v.clamp(0, 255) as u8;
        }
    }
    from_vec(dh, dw, c, out)
}

/// Same arithmetic as [`resize_linear_u8`] for 3-channel images, with planar
/// intermediate rows so both passes vectorise (AVX2 kernels in `ops::simd`).
#[inline(always)]
fn resize_linear_rgb(a: &Array3<u8>, rect: Rect, dh: usize, dw: usize) -> Array3<u8> {
    let (_, w, _) = a.dim();
    let c = 3;
    let src = data(a);
    let xa = lin_axis(rect.w(), dw);
    let ya = lin_axis(rect.h(), dh);
    let mut xo0 = Vec::with_capacity(dw);
    let mut xo1 = Vec::with_capacity(dw);
    let mut xw0 = Vec::with_capacity(dw);
    let mut xw1 = Vec::with_capacity(dw);
    for &(i0, i1, fx) in &xa {
        xo0.push(((i0 + rect.x0) * c) as i32);
        xo1.push(((i1 + rect.x0) * c) as i32);
        xw0.push(((1.0 - fx) * COEF_SCALE).round_ties_even() as i32);
        xw1.push((fx * COEF_SCALE).round_ties_even() as i32);
    }
    #[cfg(target_arch = "x86_64")]
    let simd = super::simd::avx2_available();
    // planar row buffers: [R | G | B], each `dw` long
    let hres = |sy: usize, buf: &mut [i32]| {
        let row_off = (sy * w * c) as i32;
        let (br, rest) = buf.split_at_mut(dw);
        let (bg, bb) = rest.split_at_mut(dw);
        #[allow(unused_mut)]
        let mut dx = 0;
        #[cfg(target_arch = "x86_64")]
        if simd {
            let (mut o0, mut o1) = ([0i32; 8], [0i32; 8]);
            while dx + 8 <= dw {
                for k in 0..8 {
                    o0[k] = row_off + xo0[dx + k];
                    o1[k] = row_off + xo1[dx + k];
                }
                // SAFETY: AVX2 is available; the kernel checks that every 4-byte read stays inside `src`.
                let ok = unsafe {
                    super::simd::x86::hres_rgb8(
                        src,
                        &o0,
                        &o1,
                        &xw0[dx..dx + 8],
                        &xw1[dx..dx + 8],
                        [&mut br[dx..dx + 8], &mut bg[dx..dx + 8], &mut bb[dx..dx + 8]],
                    )
                };
                if !ok {
                    break;
                }
                dx += 8;
            }
        }
        let row = &src[sy * w * c..(sy + 1) * w * c];
        for x in dx..dw {
            let (o0, o1, w0, w1) = (xo0[x] as usize, xo1[x] as usize, xw0[x], xw1[x]);
            br[x] = row[o0] as i32 * w0 + row[o1] as i32 * w1;
            bg[x] = row[o0 + 1] as i32 * w0 + row[o1 + 1] as i32 * w1;
            bb[x] = row[o0 + 2] as i32 * w0 + row[o1 + 2] as i32 * w1;
        }
    };
    let mut ba = vec![0i32; 3 * dw];
    let mut bb = vec![0i32; 3 * dw];
    let (mut ra, mut rb) = (usize::MAX, usize::MAX);
    let rowlen = dw * c;
    let mut out = vec![0u8; dh * rowlen];
    for (dy, &(i0, i1, fy)) in ya.iter().enumerate() {
        let (r0, r1) = (i0 + rect.y0, i1 + rect.y0);
        if ra != r0 {
            if rb == r0 {
                std::mem::swap(&mut ba, &mut bb);
                std::mem::swap(&mut ra, &mut rb);
            } else {
                hres(r0, &mut ba);
                ra = r0;
            }
        }
        if rb != r1 {
            hres(r1, &mut bb);
            rb = r1;
        }
        let b0 = ((1.0 - fy) * COEF_SCALE).round_ties_even() as i32;
        let b1 = (fy * COEF_SCALE).round_ties_even() as i32;
        let orow = &mut out[dy * rowlen..(dy + 1) * rowlen];
        let pa = [&ba[..dw], &ba[dw..2 * dw], &ba[2 * dw..]];
        let pb = [&bb[..dw], &bb[dw..2 * dw], &bb[2 * dw..]];
        #[allow(unused_mut)]
        let mut x0 = 0;
        #[cfg(target_arch = "x86_64")]
        if simd {
            x0 = dw / 8 * 8;
            // SAFETY: AVX2 is available; all planes hold `dw >= x0` values and `orow` 3 * x0 bytes.
            unsafe { super::simd::x86::vres_rgb(pa, pb, b0, b1, &mut orow[..3 * x0]) };
        }
        // Same arithmetic as OpenCV's SIMD vertical pass (VResizeLinearVec_32s8u)
        for x in x0..dw {
            for k in 0..3 {
                let v = (((b0 * (pa[k][x] >> 4)) >> 16) + ((b1 * (pb[k][x] >> 4)) >> 16) + 2) >> 2;
                orow[x * 3 + k] = v.clamp(0, 255) as u8;
            }
        }
    }
    from_vec(dh, dw, c, out)
}

fn resize_linear_generic<T: Element>(a: &Array3<T>, rect: Rect, dh: usize, dw: usize) -> Array3<T> {
    let (_, w, c) = a.dim();
    let src = data(a);
    let xa: Vec<(usize, usize, f32)> = lin_axis(rect.w(), dw)
        .into_iter()
        .map(|(i0, i1, f)| ((i0 + rect.x0) * c, (i1 + rect.x0) * c, f))
        .collect();
    let ya = lin_axis(rect.h(), dh);
    let rowlen = dw * c;
    let hres = |sy: usize, buf: &mut [f32]| {
        let row = &src[sy * w * c..(sy + 1) * w * c];
        for (dx, &(o0, o1, f)) in xa.iter().enumerate() {
            for k in 0..c {
                let p0 = row[o0 + k].to_f32();
                let p1 = row[o1 + k].to_f32();
                buf[dx * c + k] = p0 + (p1 - p0) * f;
            }
        }
    };
    let mut ba = vec![0f32; rowlen];
    let mut bb = vec![0f32; rowlen];
    let mut out = Vec::with_capacity(dh * rowlen);
    for &(i0, i1, fy) in &ya {
        hres(i0 + rect.y0, &mut ba);
        hres(i1 + rect.y0, &mut bb);
        out.extend(
            ba.iter()
                .zip(bb.iter())
                .map(|(&p0, &p1)| T::from_f32(p0 + (p1 - p0) * fy)),
        );
    }
    from_vec(dh, dw, c, out)
}

fn resize_fir(a: &Array3<u8>, rect: Rect, dh: usize, dw: usize, interp: Interp) -> Option<Array3<u8>> {
    let (h, w, c) = a.dim();
    let pt = match c {
        1 => fr::PixelType::U8,
        2 => fr::PixelType::U8x2,
        3 => fr::PixelType::U8x3,
        4 => fr::PixelType::U8x4,
        _ => return None,
    };
    let src = fr::images::ImageRef::new(w as u32, h as u32, data(a), pt).ok()?;
    let mut dst = fr::images::Image::new(dw as u32, dh as u32, pt);
    let filter = match interp {
        Interp::Cubic => fr::FilterType::CatmullRom,
        Interp::Area => fr::FilterType::Box,
        Interp::Lanczos => fr::FilterType::Lanczos3,
        _ => fr::FilterType::Bilinear,
    };
    let opts = fr::ResizeOptions::new()
        .resize_alg(fr::ResizeAlg::Convolution(filter))
        .use_alpha(false)
        .crop(rect.x0 as f64, rect.y0 as f64, rect.w() as f64, rect.h() as f64);
    let mut resizer = fr::Resizer::new();
    resizer.resize(&src, &mut dst, &opts).ok()?;
    Some(from_vec(dh, dw, c, dst.into_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize() {
        let a = Array3::from_shape_fn((7, 9, 3), |(y, x, k)| (y * 20 + x * 3 + k) as u8);
        let r = Rect::full(7, 9);
        assert_eq!(resize_u8(&a, r, 7, 9, Interp::Linear), a);
        assert_eq!(resize_u8(&a, r, 7, 9, Interp::Nearest), a);
        let f = a.mapv(|v| v as f32);
        assert_eq!(resize(&f, r, 7, 9, Interp::Linear), f);
    }

    #[test]
    fn constant_image_stays_constant() {
        let a = Array3::from_elem((13, 17, 3), 77u8);
        for (h, w) in [(5, 6), (30, 41), (13, 3)] {
            for i in [
                Interp::Nearest,
                Interp::Linear,
                Interp::Cubic,
                Interp::Area,
                Interp::Lanczos,
            ] {
                let o = resize_u8(&a, Rect::full(13, 17), h, w, i);
                assert_eq!(o.dim(), (h, w, 3));
                assert!(o.iter().all(|&v| v == 77), "{i:?}");
            }
        }
    }

    #[test]
    fn crop_window_equals_crop_then_resize() {
        let a = Array3::from_shape_fn((20, 30, 3), |(y, x, k)| ((y * 13 + x * 5 + k * 50) % 256) as u8);
        let rect = Rect {
            x0: 4,
            y0: 3,
            x1: 24,
            y1: 15,
        };
        let cropped = crate::ops::crop(&a, 4, 3, 24, 15);
        let a1 = resize_u8(&a, rect, 25, 33, Interp::Linear);
        let a2 = resize_u8(&cropped, Rect::full(12, 20), 25, 33, Interp::Linear);
        assert_eq!(a1, a2);
    }
}
