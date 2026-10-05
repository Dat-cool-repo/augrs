//! 8-bit HSV round trip with OpenCV's exact arithmetic (`COLOR_RGB2HSV` /
//! `COLOR_HSV2RGB` on `CV_8U`, hue range 180), so `HueSaturationValue` and the
//! `ColorJitter` hue shift reproduce Albumentations bit for bit.
//!
//! * RGB -> HSV is OpenCV's integer algorithm (12-bit reciprocal tables). The
//!   tables are recomputed per pixel with an f32 division that rounds to the
//!   same integers (checked exhaustively in the tests), which lets the whole
//!   loop vectorise without gathers.
//! * HSV -> RGB follows OpenCV's SIMD kernel (f32 math, results *truncated*).
//!   OpenCV's AVX2 build is compiled with FMA contraction, so `1 - s * h` is a
//!   fused multiply-add there; augrs uses `mul_add` for exactly those terms
//!   (verified against cv2 on all 180 * 256 * 256 inputs).
//!   OpenCV processes each image row in blocks of 32 pixels (AVX2 build) and
//!   finishes the row with a scalar loop that *rounds* instead; that tail is
//!   reproduced too (`TAIL_BLOCK`).
//!
//! Everything runs on SoA blocks so the compiler can vectorise it; the
//! function is compiled for AVX2 and baseline x86-64 (no FMA: same results).

use ndarray::Array3;

/// OpenCV's HSV2RGB SIMD body handles `4 * f32 lanes` pixels per iteration (32 with AVX2).
const TAIL_BLOCK: usize = 32;
const BLK: usize = 32;

/// `x` rounded half to even, for `0 <= x < 2^22` (f32, no SSE4.1 needed).
#[inline(always)]
fn rne(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23
    (x + MAGIC) - MAGIC
}

/// `round((255 << 12) / v)` (OpenCV `sdiv_table`), `0` for `v == 0`.
#[inline(always)]
fn sdiv(v: i32) -> i32 {
    let q = rne(1_044_480.0 / (v.max(1) as f32)) as i32;
    if v == 0 { 0 } else { q }
}

/// `round((180 << 12) / (6 * d))` (OpenCV `hdiv_table180`), `0` for `d == 0`.
#[inline(always)]
fn hdiv(d: i32) -> i32 {
    let q = rne(122_880.0 / (d.max(1) as f32)) as i32;
    if d == 0 { 0 } else { q }
}

/// OpenCV RGB -> HSV (8-bit, H in `[0, 180]`).
#[inline(always)]
pub fn rgb_to_hsv(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let v = r.max(g).max(b);
    let vmin = r.min(g).min(b);
    let diff = v - vmin;
    let s = (diff * sdiv(v) + (1 << 11)) >> 12;
    let h0 = if v == r {
        g - b
    } else if v == g {
        b - r + 2 * diff
    } else {
        r - g + 4 * diff
    };
    let mut h = (h0 * hdiv(diff) + (1 << 11)) >> 12;
    if h < 0 {
        h += 180;
    }
    (h, s, v)
}

const HSCALE: f32 = 6.0 / 180.0;
const INV255: f32 = 1.0 / 255.0;

/// OpenCV HSV -> RGB, SIMD formula (truncating). Inputs are 8-bit HSV values.
#[inline(always)]
fn hsv_to_rgb_simd(h: i32, s: i32, v: i32) -> (i32, i32, i32) {
    let hf = h as f32 * HSCALE;
    let pre = (hf as i32) as f32; // trunc (hf >= 0)
    let fr = hf - pre;
    let sf = s as f32 * INV255;
    let vf = v as f32 * INV255;
    let tab0 = vf;
    let tab1 = vf * (1.0 - sf);
    let tab2 = vf * (-sf).mul_add(fr, 1.0);
    let tab3 = vf * (-sf).mul_add(1.0 - fr, 1.0);
    let sector = pre - ((pre * (1.0 / 6.0)) as i32) as f32 * 6.0;
    let b = if sector < 2.0 {
        tab1
    } else if sector == 2.0 {
        tab3
    } else if sector <= 4.0 {
        tab0
    } else {
        tab2
    };
    let g = if sector < 1.0 {
        tab3
    } else if sector <= 2.0 {
        tab0
    } else if sector == 3.0 {
        tab2
    } else {
        tab1
    };
    let r = if sector < 1.0 {
        tab0
    } else if sector == 1.0 {
        tab2
    } else if sector <= 3.0 {
        tab1
    } else if sector == 4.0 {
        tab3
    } else {
        tab0
    };
    // v_trunc then saturating packs; values are >= 0
    (
        ((r * 255.0) as i32).min(255),
        ((g * 255.0) as i32).min(255),
        ((b * 255.0) as i32).min(255),
    )
}

/// OpenCV HSV -> RGB, scalar formula (rounding half to even), used for row tails.
fn hsv_to_rgb_scalar(h: i32, s: i32, v: i32) -> (i32, i32, i32) {
    let sf = s as f32 * INV255;
    let vf = v as f32 * INV255;
    let round = |x: f32| (x * 255.0).round_ties_even().clamp(0.0, 255.0) as i32;
    if sf == 0.0 {
        let c = round(vf);
        return (c, c, c);
    }
    const SECTOR: [[usize; 3]; 6] = [[1, 3, 0], [1, 0, 2], [3, 0, 1], [0, 2, 1], [0, 1, 3], [2, 1, 0]];
    let mut hf = h as f32 * HSCALE;
    let mut sector = hf.floor() as i32;
    hf -= sector as f32;
    sector %= 6;
    if sector < 0 {
        sector += 6;
    }
    let tab = [
        vf,
        vf * (1.0 - sf),
        vf * (-sf).mul_add(hf, 1.0),
        vf * (-sf).mul_add(1.0 - hf, 1.0),
    ];
    let d = SECTOR[sector as usize];
    // sector_data is (b, g, r)
    (round(tab[d[2]]), round(tab[d[1]]), round(tab[d[0]]))
}

/// Per-pixel HSV edit: hue through a 256-entry LUT, saturation and value by an
/// integer offset (saturating). OpenCV/Albumentations semantics: pixels with
/// `s == 0` keep `s == 0` when `keep_gray_sat` is set.
#[derive(Clone, Debug)]
pub struct HsvEdit {
    pub hue_lut: Option<[u8; 256]>,
    pub sat_add: i32,
    pub val_add: i32,
    pub keep_gray_sat: bool,
}

impl HsvEdit {
    #[inline(always)]
    fn apply(&self, h: &mut [i32; BLK], s: &mut [i32; BLK], v: &mut [i32; BLK]) {
        if let Some(lut) = &self.hue_lut {
            for x in h.iter_mut() {
                *x = lut[(*x & 255) as usize] as i32;
            }
        }
        if self.sat_add != 0 {
            for x in s.iter_mut() {
                let n = (*x + self.sat_add).clamp(0, 255);
                *x = if self.keep_gray_sat && *x == 0 { 0 } else { n };
            }
        }
        if self.val_add != 0 {
            for x in v.iter_mut() {
                *x = (*x + self.val_add).clamp(0, 255);
            }
        }
    }
}

/// Hue LUT `trunc(mod(i + shift, 180))` computed like NumPy (float64, floored mod).
pub fn hue_lut(shift: f64) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        let a = i as f64 + shift;
        let mut m = a % 180.0;
        if m != 0.0 && m < 0.0 {
            m += 180.0;
        }
        *l = m as u8;
    }
    lut
}

super::avx2_fma_dispatch!(
    /// Convert each RGB pixel to 8-bit HSV, apply `edit`, convert back (in place).
    /// Images with fewer than 3 channels are treated as gray (`h = s = 0`).
    pub fn hsv_edit_u8(a: &mut Array3<u8>, edit: &HsvEdit) = hsv_edit_u8_impl
);

#[inline(always)]
fn hsv_edit_u8_impl(a: &mut Array3<u8>, edit: &HsvEdit) {
    let (_, w, c) = a.dim();
    if w == 0 {
        return;
    }
    let simd_w = w - w % TAIL_BLOCK;
    let data = a.as_slice_mut().expect("standard layout");
    #[cfg(target_arch = "x86_64")]
    let avx2 = c == 3 && super::simd::avx2_available();
    #[cfg(target_arch = "x86_64")]
    let lut32: Option<[i32; 256]> = edit.hue_lut.map(|l| l.map(|v| v as i32));
    let (mut r, mut g, mut b) = ([0i32; BLK], [0i32; BLK], [0i32; BLK]);
    let (mut h, mut s, mut v) = ([0i32; BLK], [0i32; BLK], [0i32; BLK]);
    for row in data.chunks_exact_mut(w * c) {
        let (body, tail) = row.split_at_mut(simd_w * c);
        #[cfg(target_arch = "x86_64")]
        let body: &mut [u8] = if avx2 {
            // SAFETY: AVX2 and FMA are available (checked above); body is a whole number of 8-pixel groups.
            unsafe { super::simd::x86::hsv_edit_rgb(body, edit, lut32.as_ref()) };
            &mut []
        } else {
            body
        };
        for block in body.chunks_exact_mut(BLK * c) {
            if c >= 3 {
                for (i, p) in block.chunks_exact(c).enumerate() {
                    r[i] = p[0] as i32;
                    g[i] = p[1] as i32;
                    b[i] = p[2] as i32;
                }
                for i in 0..BLK {
                    let (hh, ss, vv) = rgb_to_hsv(r[i], g[i], b[i]);
                    h[i] = hh;
                    s[i] = ss;
                    v[i] = vv;
                }
            } else {
                for (i, p) in block.chunks_exact(c).enumerate() {
                    h[i] = 0;
                    s[i] = 0;
                    v[i] = p[0] as i32;
                }
            }
            edit.apply(&mut h, &mut s, &mut v);
            for i in 0..BLK {
                let (rr, gg, bb) = hsv_to_rgb_simd(h[i], s[i], v[i]);
                r[i] = rr;
                g[i] = gg;
                b[i] = bb;
            }
            if c >= 3 {
                for (i, p) in block.chunks_exact_mut(c).enumerate() {
                    p[0] = r[i] as u8;
                    p[1] = g[i] as u8;
                    p[2] = b[i] as u8;
                }
            } else {
                // gray -> RGB -> gray of equal channels is the identity
                for (i, p) in block.chunks_exact_mut(c).enumerate() {
                    p[0] = r[i] as u8;
                }
            }
        }
        for p in tail.chunks_exact_mut(c) {
            let (hh, ss, vv) = if c >= 3 {
                rgb_to_hsv(p[0] as i32, p[1] as i32, p[2] as i32)
            } else {
                (0, 0, p[0] as i32)
            };
            let (mut hb, mut sb, mut vb) = ([0i32; BLK], [0i32; BLK], [0i32; BLK]);
            hb[0] = hh;
            sb[0] = ss;
            vb[0] = vv;
            edit.apply(&mut hb, &mut sb, &mut vb);
            let (rr, gg, bb) = hsv_to_rgb_scalar(hb[0], sb[0], vb[0]);
            p[0] = rr as u8;
            if c >= 3 {
                p[1] = gg as u8;
                p[2] = bb as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_match_opencv_double_precision() {
        for i in 1..256 {
            let s = ((255i64 << 12) as f64 / i as f64).round_ties_even() as i32;
            let h = ((180i64 << 12) as f64 / (6.0 * i as f64)).round_ties_even() as i32;
            assert_eq!(sdiv(i), s, "sdiv {i}");
            assert_eq!(hdiv(i), h, "hdiv {i}");
        }
        assert_eq!((sdiv(0), hdiv(0)), (0, 0));
    }

    #[test]
    fn known_hsv_values() {
        assert_eq!(rgb_to_hsv(255, 0, 0), (0, 255, 255));
        assert_eq!(rgb_to_hsv(0, 255, 0), (60, 255, 255));
        assert_eq!(rgb_to_hsv(0, 0, 255), (120, 255, 255));
        assert_eq!(rgb_to_hsv(17, 17, 17), (0, 0, 17));
        assert_eq!(hsv_to_rgb_scalar(60, 255, 255), (0, 255, 0));
    }

    #[test]
    fn identity_edit_is_close_and_gray_is_stable() {
        let mut a = Array3::from_shape_fn((3, 70, 3), |(y, x, k)| ((y * 50 + x * 7 + k * 70) % 256) as u8);
        let orig = a.clone();
        hsv_edit_u8(
            &mut a,
            &HsvEdit {
                hue_lut: None,
                sat_add: 0,
                val_add: 0,
                keep_gray_sat: true,
            },
        );
        // the 8-bit HSV round trip is lossy (like OpenCV), but only slightly
        for (x, y) in a.iter().zip(orig.iter()) {
            assert!((*x as i32 - *y as i32).abs() <= 8);
        }
    }
}
