//! Photometric ops (ColorJitter components). u8 images use LUTs where the op
//! is per-value; float images are assumed to be in `[0, 1]` and are clipped.

use ndarray::Array3;

#[inline]
fn gray_f(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

pub fn lut_apply(a: &mut Array3<u8>, lut: &[u8; 256]) {
    for v in a.iter_mut() {
        *v = lut[*v as usize];
    }
}

/// Mean of the grayscale image (or of the single channel).
fn gray_mean_u8(a: &Array3<u8>) -> f64 {
    let (h, w, c) = a.dim();
    let s = a.as_slice().unwrap();
    if h * w == 0 {
        return 0.0;
    }
    if c < 3 {
        let sum: u64 = s.iter().step_by(c.max(1)).map(|&v| v as u64).sum();
        return sum as f64 / (h * w) as f64;
    }
    // same rounded gray as OpenCV's cvtColor(RGB2GRAY), summed in integers
    let mut sum: u64 = 0;
    #[allow(unused_mut)]
    let mut s = s;
    #[cfg(target_arch = "x86_64")]
    if c == 3 && super::simd::avx2_available() {
        let n8 = s.len() / 24 * 24;
        // SAFETY: AVX2 is available; the slice is a whole number of 8-pixel groups.
        sum += unsafe { super::simd::x86::gray_sum_rgb(&s[..n8]) };
        s = &s[n8..];
    }
    for p in s.chunks_exact(c) {
        sum += gray_u8(p[0] as i32, p[1] as i32, p[2] as i32) as u64;
    }
    sum as f64 / (h * w) as f64
}

fn gray_mean_f32(a: &Array3<f32>) -> f64 {
    let (h, w, c) = a.dim();
    let s = a.as_slice().unwrap();
    if h * w == 0 {
        return 0.0;
    }
    let sum: f64 = if c < 3 {
        s.iter().step_by(c.max(1)).map(|&v| v as f64).sum()
    } else {
        s.chunks_exact(c).map(|p| gray_f(p[0], p[1], p[2]) as f64).sum()
    };
    sum / (h * w) as f64
}

pub fn brightness_u8(a: &mut Array3<u8>, f: f64) {
    if f == 1.0 {
        return;
    }
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = (i as f64 * f + 0.5).clamp(0.0, 255.0) as u8;
    }
    lut_apply(a, &lut);
}

pub fn contrast_u8(a: &mut Array3<u8>, f: f64) {
    if f == 1.0 {
        return;
    }
    let mean = gray_mean_u8(a);
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = (i as f64 * f + mean * (1.0 - f) + 0.5).clamp(0.0, 255.0) as u8;
    }
    lut_apply(a, &lut);
}

/// OpenCV's fixed-point RGB->gray for 8-bit images (`RY15/GY15/BY15`, 15-bit), rounded.
#[inline(always)]
pub(crate) fn gray_u8(r: i32, g: i32, b: i32) -> i32 {
    (r * 9798 + g * 19235 + b * 3735 + (1 << 14)) >> 15
}

/// Pixels per SoA block: deinterleave, process with straight-line (vectorisable) code, re-interleave.
const BLK: usize = 64;

avx2_dispatch!(
    /// Blend with the grayscale image: `c * f + gray * (1 - f)` (12-bit fixed point).
    pub fn saturation_u8(a: &mut Array3<u8>, f: f64) = saturation_u8_impl
);

#[inline(always)]
fn saturation_u8_impl(a: &mut Array3<u8>, f: f64) {
    let c = a.dim().2;
    if f == 1.0 || c < 3 {
        return;
    }
    let fq = (f * 4096.0).round() as i32;
    let gq = 4096 - fq;
    let (mut r, mut g, mut b) = ([0i32; BLK], [0i32; BLK], [0i32; BLK]);
    #[allow(unused_mut)]
    let mut data = a.as_slice_mut().unwrap();
    #[cfg(target_arch = "x86_64")]
    if c == 3 && super::simd::avx2_available() {
        let n8 = data.len() / 24 * 24;
        let (head, rest) = data.split_at_mut(n8);
        // SAFETY: AVX2 is available; `head` is a whole number of 8-pixel groups.
        unsafe { super::simd::x86::saturation_rgb(head, fq) };
        data = rest;
    }
    for block in data.chunks_mut(BLK * c) {
        let n = block.len() / c;
        for (i, p) in block.chunks_exact(c).enumerate() {
            r[i] = p[0] as i32;
            g[i] = p[1] as i32;
            b[i] = p[2] as i32;
        }
        for i in 0..BLK {
            let gt = gray_u8(r[i], g[i], b[i]) * gq + 2048;
            r[i] = ((r[i] * fq + gt) >> 12).clamp(0, 255);
            g[i] = ((g[i] * fq + gt) >> 12).clamp(0, 255);
            b[i] = ((b[i] * fq + gt) >> 12).clamp(0, 255);
        }
        for (i, p) in block.chunks_exact_mut(c).enumerate().take(n) {
            p[0] = r[i] as u8;
            p[1] = g[i] as u8;
            p[2] = b[i] as u8;
        }
    }
}

/// `floor(x)` for `0 <= x < 2^22` without libm / SSE4.1: round-to-nearest via the
/// 1.5 * 2^23 trick applied to `x - 0.5` (ties may land on either neighbour,
/// which is harmless for the continuous, periodic hue formula below).
#[inline(always)]
fn floor_pos(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    (x - 0.5 + MAGIC) - MAGIC
}

/// Rotate hue of an RGB triple by `shift6` sixths of a turn.
///
/// Branch-free (data-dependent branches mispredict on every other pixel):
/// RGB -> (V, delta, H) and back via `f(n) = V - delta * clamp(min(k, 4 - k), 0, 1)`,
/// `k = (n + H) mod 6` for `n = 5, 3, 1`. Gray pixels (`delta = 0`) are unchanged.
#[inline(always)]
fn hue_rotate(r: f32, g: f32, b: f32, shift6: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let inv = 1.0 / delta.max(1e-6);
    let hr = (g - b) * inv;
    let hg = (b - r) * inv + 2.0;
    let hb = (r - g) * inv + 4.0;
    let h = if max == r {
        hr
    } else if max == g {
        hg
    } else {
        hb
    };
    // h in [-1, 6), shift6 in [-3, 3]: +12 keeps everything positive
    let h = h + shift6 + 12.0;
    #[inline(always)]
    fn chan(n: f32, h: f32, max: f32, delta: f32) -> f32 {
        let k = n + h;
        let k = k - 6.0 * floor_pos(k * (1.0 / 6.0));
        max - delta * k.min(4.0 - k).clamp(0.0, 1.0)
    }
    (
        chan(5.0, h, max, delta),
        chan(3.0, h, max, delta),
        chan(1.0, h, max, delta),
    )
}

/// Shift hue by `shift` turns (`[-0.5, 0.5]`) through 8-bit HSV, exactly like
/// Albumentations/OpenCV (`H' = (H + 180 * shift) mod 180`, 2-degree steps).
pub fn hue_u8(a: &mut Array3<u8>, shift: f64) {
    if shift == 0.0 || a.dim().2 < 3 {
        return;
    }
    let edit = super::hsv::HsvEdit {
        hue_lut: Some(super::hsv::hue_lut(180.0 * shift)),
        sat_add: 0,
        val_add: 0,
        keep_gray_sat: true,
    };
    super::hsv::hsv_edit_u8(a, &edit);
}

// ---- RandomBrightnessContrast / RandomGamma / ToGray ----------------------

/// `img * alpha + beta` with Albumentations' uint8 LUT arithmetic
/// (float32 multiply then add, clipped, truncated).
pub fn multiply_add_lut(alpha: f64, beta: f64) -> [u8; 256] {
    let (a, b) = (alpha as f32, beta as f32);
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        let v = i as f32 * a + b;
        *l = v.clamp(0.0, 255.0) as u8;
    }
    lut
}

pub fn multiply_add_f32(a: &mut Array3<f32>, alpha: f64, beta: f64) {
    a.mapv_inplace(|v| ((v as f64) * alpha + beta).clamp(0.0, 1.0) as f32);
}

/// Albumentations' gamma table: `trunc((i * (1/255)) ** gamma * 255)`.
pub fn gamma_lut(gamma: f64) -> [u8; 256] {
    let step = 1.0 / 255.0;
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = ((i as f64 * step).powf(gamma) * 255.0).clamp(0.0, 255.0) as u8;
    }
    lut
}

pub fn gamma_f32(a: &mut Array3<f32>, gamma: f64) {
    let g = gamma as f32;
    a.mapv_inplace(|v| v.max(0.0).powf(g));
}

/// Mean over all values (Albumentations' `np.mean(image)`).
pub fn mean_all(b: &crate::buffer::Buf) -> f64 {
    use crate::buffer::Buf;
    match b {
        Buf::U8(a) => {
            let s: u64 = a.iter().map(|&v| v as u64).sum();
            s as f64 / a.len().max(1) as f64
        }
        Buf::F32(a) => a.iter().map(|&v| v as f64).sum::<f64>() / a.len().max(1) as f64,
        other => {
            crate::with_buf!(other, a => a.iter().map(|v| crate::buffer::Element::to_f32(*v) as f64).sum::<f64>() / a.len().max(1) as f64)
        }
    }
}

/// Grayscale conversion methods (Albumentations `ToGray.method`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GrayMethod {
    /// OpenCV `RGB2GRAY` (0.299 R + 0.587 G + 0.114 B, fixed point).
    #[default]
    WeightedAverage,
    /// `(max + min) / 2`.
    Desaturation,
    /// Mean of the channels.
    Average,
    /// Max of the channels.
    Max,
}

/// Convert to gray with `out_channels` identical channels.
pub fn to_gray_u8(a: &Array3<u8>, method: GrayMethod, out_channels: usize) -> Array3<u8> {
    let (h, w, c) = a.dim();
    let src = a.as_slice().unwrap();
    let mut out = Vec::with_capacity(h * w * out_channels);
    for p in src.chunks_exact(c) {
        let g = match method {
            GrayMethod::WeightedAverage if c >= 3 => gray_u8(p[0] as i32, p[1] as i32, p[2] as i32) as u8,
            GrayMethod::WeightedAverage => p[0],
            GrayMethod::Desaturation => {
                let (mx, mn) = p.iter().fold((0u8, 255u8), |(mx, mn), &v| (mx.max(v), mn.min(v)));
                ((mx as f32 + mn as f32) / 2.0) as u8
            }
            // float64 mean then truncation, like `np.mean(axis=-1).astype(uint8)`
            GrayMethod::Average => (p.iter().map(|&v| v as f64).sum::<f64>() / c as f64) as u8,
            GrayMethod::Max => *p.iter().max().unwrap(),
        };
        for _ in 0..out_channels {
            out.push(g);
        }
    }
    Array3::from_shape_vec((h, w, out_channels), out).unwrap()
}

pub fn to_gray_f32(a: &Array3<f32>, method: GrayMethod, out_channels: usize) -> Array3<f32> {
    let (h, w, c) = a.dim();
    let src = a.as_slice().unwrap();
    let mut out = Vec::with_capacity(h * w * out_channels);
    for p in src.chunks_exact(c) {
        let g = match method {
            GrayMethod::WeightedAverage if c >= 3 => gray_f(p[0], p[1], p[2]),
            GrayMethod::WeightedAverage => p[0],
            GrayMethod::Desaturation => {
                let (mx, mn) = p
                    .iter()
                    .fold((f32::MIN, f32::MAX), |(mx, mn), &v| (mx.max(v), mn.min(v)));
                ((mx + mn) / 2.0).clamp(0.0, 1.0)
            }
            GrayMethod::Average => p.iter().sum::<f32>() / c as f32,
            GrayMethod::Max => p.iter().cloned().fold(f32::MIN, f32::max),
        };
        for _ in 0..out_channels {
            out.push(g);
        }
    }
    Array3::from_shape_vec((h, w, out_channels), out).unwrap()
}

pub fn brightness_f32(a: &mut Array3<f32>, f: f64) {
    let f = f as f32;
    a.mapv_inplace(|v| (v * f).clamp(0.0, 1.0));
}

pub fn contrast_f32(a: &mut Array3<f32>, f: f64) {
    let mean = gray_mean_f32(a) as f32;
    let f = f as f32;
    a.mapv_inplace(|v| (v * f + mean * (1.0 - f)).clamp(0.0, 1.0));
}

pub fn saturation_f32(a: &mut Array3<f32>, f: f64) {
    let c = a.dim().2;
    if c < 3 {
        return;
    }
    let f = f as f32;
    for p in a.as_slice_mut().unwrap().chunks_exact_mut(c) {
        let gray = gray_f(p[0], p[1], p[2]) * (1.0 - f);
        for v in &mut p[..3] {
            *v = (*v * f + gray).clamp(0.0, 1.0);
        }
    }
}

pub fn hue_f32(a: &mut Array3<f32>, shift: f64) {
    let c = a.dim().2;
    if shift == 0.0 || c < 3 {
        return;
    }
    let s6 = (shift * 6.0) as f32;
    for p in a.as_slice_mut().unwrap().chunks_exact_mut(c) {
        let (r, g, b) = hue_rotate(p[0], p[1], p[2], s6);
        p[0] = r;
        p[1] = g;
        p[2] = b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hue_full_turn_is_identity() {
        let mut a = Array3::from_shape_fn((4, 5, 3), |(y, x, k)| (y * 50 + x * 20 + k * 70) as f32 / 255.0);
        let orig = a.clone();
        hue_f32(&mut a, 1.0);
        for (x, y) in a.iter().zip(orig.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn luts() {
        assert_eq!(multiply_add_lut(1.0, 0.0)[200], 200);
        assert_eq!(multiply_add_lut(1.5, -10.0)[100], 140);
        assert_eq!(multiply_add_lut(2.0, 0.0)[200], 255);
        let g = gamma_lut(1.0);
        assert!((g[128] as i32 - 128).abs() <= 1);
        assert_eq!(gamma_lut(2.0)[0], 0);
    }

    #[test]
    fn hue_known_values() {
        // red rotated by 1/3 turn becomes green
        let (r, g, b) = hue_rotate(255.0, 0.0, 0.0, 2.0);
        assert_eq!((r, g, b), (0.0, 255.0, 0.0));
        let (r, g, b) = hue_rotate(255.0, 0.0, 0.0, -2.0);
        assert_eq!((r, g, b), (0.0, 0.0, 255.0));
    }

    #[test]
    fn saturation_zero_is_gray() {
        let mut a = Array3::from_shape_fn((2, 2, 3), |(y, x, k)| (y * 90 + x * 40 + k * 30) as u8);
        saturation_u8(&mut a, 0.0);
        for p in a.as_slice().unwrap().chunks(3) {
            assert!(p[0] == p[1] && p[1] == p[2]);
        }
    }

    #[test]
    fn brightness_contrast_identity() {
        let mut a = Array3::from_shape_fn((3, 3, 3), |(y, x, k)| (y * 90 + x * 40 + k * 30) as u8);
        let o = a.clone();
        brightness_u8(&mut a, 1.0);
        contrast_u8(&mut a, 1.0);
        saturation_u8(&mut a, 1.0);
        assert_eq!(a, o);
    }
}
