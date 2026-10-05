//! `(x - mean * max_pixel_value) / (std * max_pixel_value)` -> float32.

use crate::buffer::{Buf, Element, data, from_vec};
use crate::error::{Result, param};
use ndarray::Array3;

fn per_channel(v: &[f64], c: usize, what: &str) -> Result<Vec<f64>> {
    match v.len() {
        1 => Ok(vec![v[0]; c]),
        n if n == c => Ok(v.to_vec()),
        n => param(format!(
            "Normalize: {what} has {n} values but the image has {c} channels"
        )),
    }
}

pub fn normalize(img: &Buf, mean: &[f64], std: &[f64], max_pixel_value: f64) -> Result<Array3<f32>> {
    let (h, w, c) = img.dims();
    let mean = per_channel(mean, c, "mean")?;
    let std = per_channel(std, c, "std")?;
    let scale: Vec<f64> = std.iter().map(|s| 1.0 / (s * max_pixel_value)).collect();
    let shift: Vec<f64> = mean.iter().map(|m| m * max_pixel_value).collect();
    let out = match img {
        Buf::U8(a) if c == 3 => {
            // x * scale + offset with a 12-lane repeating channel pattern (lcm of 3 and 4),
            // which the compiler vectorises; faster than a LUT gather.
            let src = data(a);
            let mut sc = [0f32; 12];
            let mut of = [0f32; 12];
            for i in 0..12 {
                sc[i] = scale[i % 3] as f32;
                of[i] = (-shift[i % 3] * scale[i % 3]) as f32;
            }
            affine_u8x12(src, &sc, &of)
        }
        Buf::U8(a) => {
            // per-channel LUT: 256 * c entries
            let mut lut = vec![0f32; 256 * c];
            for k in 0..c {
                for v in 0..256 {
                    lut[k * 256 + v] = ((v as f64 - shift[k]) * scale[k]) as f32;
                }
            }
            let src = data(a);
            let mut out = Vec::with_capacity(src.len());
            {
                for p in src.chunks_exact(c) {
                    for (k, &v) in p.iter().enumerate() {
                        out.push(lut[k * 256 + v as usize]);
                    }
                }
            }
            out
        }
        other => {
            fn go<T: Element>(a: &Array3<T>, c: usize, shift: &[f64], scale: &[f64]) -> Vec<f32> {
                data(a)
                    .chunks_exact(c)
                    .flat_map(|p| {
                        p.iter()
                            .enumerate()
                            .map(|(k, v)| ((v.to_f32() as f64 - shift[k]) * scale[k]) as f32)
                    })
                    .collect()
            }
            crate::with_buf!(other, a => go(a, c, &shift, &scale))
        }
    };
    Ok(from_vec(h, w, c, out))
}

super::avx2_dispatch!(
    fn affine_u8x12(src: &[u8], sc: &[f32; 12], of: &[f32; 12]) -> Vec<f32> = affine_u8x12_impl
);

#[inline(always)]
fn affine_u8x12_impl(src: &[u8], sc: &[f32; 12], of: &[f32; 12]) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if super::simd::avx2_available() {
        done = src.len() / 24 * 24;
        // SAFETY: AVX2 is available; both slices hold `done` (a multiple of 24) elements.
        unsafe { super::simd::x86::affine_u8_rgb(&src[..done], &mut out[..done], sc, of) };
    }
    let (src, out_rest) = (&src[done..], &mut out[done..]);
    let mut oc = out_rest.chunks_exact_mut(12);
    let mut ic = src.chunks_exact(12);
    for (o, i) in (&mut oc).zip(&mut ic) {
        for k in 0..12 {
            o[k] = i[k] as f32 * sc[k] + of[k];
        }
    }
    for (k, (o, i)) in oc.into_remainder().iter_mut().zip(ic.remainder()).enumerate() {
        *o = *i as f32 * sc[k] + of[k];
    }
    out
}
