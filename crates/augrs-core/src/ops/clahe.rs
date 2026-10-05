//! CLAHE (contrast-limited adaptive histogram equalisation).
//!
//! The single-channel algorithm is OpenCV's (`cv::createCLAHE(...).apply` on
//! 8-bit images): reflect-101 padding to a multiple of the tile grid, per-tile
//! clipped histograms with the same integer redistribution, rounded LUTs and
//! the same f32 bilinear blend between tile LUTs, so gray results are exact.
//!
//! RGB images are equalised on the L channel of CIE Lab (sRGB, D65), as in
//! Albumentations. OpenCV's 8-bit Lab conversion uses its own fixed-point
//! tables; augrs converts in float, so RGB results are close but not exact.

use ndarray::Array3;
use std::sync::OnceLock;

fn border101(i: isize, n: usize) -> usize {
    crate::border::border_index(i, n, crate::border::BorderMode::Reflect101).unwrap_or(0)
}

/// OpenCV CLAHE on one 8-bit channel stored as `src[y * w + x]`.
pub fn clahe_channel(src: &[u8], h: usize, w: usize, clip_limit: f64, tiles_x: usize, tiles_y: usize) -> Vec<u8> {
    let (tx_n, ty_n) = (tiles_x.max(1), tiles_y.max(1));
    // OpenCV pads bottom/right by `tiles - (size % tiles)` (a full tile when divisible
    // in one direction but not the other).
    let (ph, pw) = if w % tx_n == 0 && h % ty_n == 0 {
        (h, w)
    } else {
        (h + ty_n - h % ty_n, w + tx_n - w % tx_n)
    };
    let (tw, th) = (pw / tx_n, ph / ty_n);
    let area = (tw * th) as i64;
    const HIST: usize = 256;
    let lut_scale = 255.0f32 / area as f32;
    let clip = if clip_limit > 0.0 {
        ((clip_limit * area as f64 / HIST as f64) as i64).max(1)
    } else {
        0
    };
    let mut luts = vec![0u8; tx_n * ty_n * HIST];
    let row_map: Vec<usize> = (0..ph).map(|y| border101(y as isize, h)).collect();
    let col_map: Vec<usize> = (0..pw).map(|x| border101(x as isize, w)).collect();
    let mut hist = [0i64; HIST];
    for ty in 0..ty_n {
        for tx in 0..tx_n {
            hist.fill(0);
            for y in ty * th..(ty + 1) * th {
                let row = &src[row_map[y] * w..row_map[y] * w + w];
                if (tx + 1) * tw <= w {
                    for &v in &row[tx * tw..(tx + 1) * tw] {
                        hist[v as usize] += 1;
                    }
                } else {
                    for &x in &col_map[tx * tw..(tx + 1) * tw] {
                        hist[row[x] as usize] += 1;
                    }
                }
            }
            if clip > 0 {
                let mut clipped = 0i64;
                for v in hist.iter_mut() {
                    if *v > clip {
                        clipped += *v - clip;
                        *v = clip;
                    }
                }
                let batch = clipped / HIST as i64;
                let mut residual = clipped - batch * HIST as i64;
                for v in hist.iter_mut() {
                    *v += batch;
                }
                if residual != 0 {
                    let step = (HIST as i64 / residual).max(1) as usize;
                    let mut i = 0;
                    while i < HIST && residual > 0 {
                        hist[i] += 1;
                        i += step;
                        residual -= 1;
                    }
                }
            }
            let lut = &mut luts[(ty * tx_n + tx) * HIST..(ty * tx_n + tx + 1) * HIST];
            let mut sum = 0i64;
            for (l, &v) in lut.iter_mut().zip(hist.iter()) {
                sum += v;
                *l = ((sum as f32) * lut_scale).round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }
    // bilinear blend of the 4 neighbouring tile LUTs
    let inv_tw = 1.0f32 / tw as f32;
    let inv_th = 1.0f32 / th as f32;
    let mut ind1 = vec![0usize; w];
    let mut ind2 = vec![0usize; w];
    let mut xa = vec![0f32; w];
    let mut xa1 = vec![0f32; w];
    for x in 0..w {
        let txf = x as f32 * inv_tw - 0.5;
        let tx1 = txf.floor() as i64;
        xa[x] = txf - tx1 as f32;
        xa1[x] = 1.0 - xa[x];
        ind1[x] = tx1.max(0) as usize * HIST;
        ind2[x] = ((tx1 + 1).min(tx_n as i64 - 1)) as usize * HIST;
    }
    let mut out = vec![0u8; h * w];
    for y in 0..h {
        let tyf = y as f32 * inv_th - 0.5;
        let ty1 = tyf.floor() as i64;
        let ya = tyf - ty1 as f32;
        let ya1 = 1.0 - ya;
        let p1 = &luts[ty1.max(0) as usize * tx_n * HIST..];
        let p2 = &luts[((ty1 + 1).min(ty_n as i64 - 1)) as usize * tx_n * HIST..];
        let srow = &src[y * w..(y + 1) * w];
        let orow = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            let v = srow[x] as usize;
            let (i1, i2) = (ind1[x] + v, ind2[x] + v);
            let res = (p1[i1] as f32 * xa1[x] + p1[i2] as f32 * xa[x]) * ya1
                + (p2[i1] as f32 * xa1[x] + p2[i2] as f32 * xa[x]) * ya;
            orow[x] = res.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ---- sRGB <-> Lab (float, D65) ------------------------------------------------

const XN: f32 = 0.950456;
const ZN: f32 = 1.088754;
// sRGB -> XYZ (D65), rows X, Y, Z; X and Z pre-divided by the white point
const M: [[f32; 3]; 3] = [
    [0.412453 / XN, 0.357580 / XN, 0.180423 / XN],
    [0.212671, 0.715160, 0.072169],
    [0.019334 / ZN, 0.119193 / ZN, 0.950227 / ZN],
];
// XYZ -> sRGB (inverse of the unnormalised matrix)
const MI: [[f32; 3]; 3] = [
    [3.240479, -1.53715, -0.498535],
    [-0.969256, 1.875991, 0.041556],
    [0.055648, -0.204043, 1.057311],
];
const CBRT_N: usize = 4096;

struct Tabs {
    to_lin: [f32; 256],
    /// Lab `f(t)` on `[0, 1]`, `CBRT_N + 2` samples
    f: Vec<f32>,
    /// linear -> sRGB in `[0, 255]` on `[0, 1]`, `CBRT_N + 2` samples
    to_srgb: Vec<f32>,
}

fn tabs() -> &'static Tabs {
    static T: OnceLock<Tabs> = OnceLock::new();
    T.get_or_init(|| {
        let mut to_lin = [0f32; 256];
        for (i, v) in to_lin.iter_mut().enumerate() {
            let c = i as f64 / 255.0;
            *v = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            } as f32;
        }
        let f = (0..CBRT_N + 2)
            .map(|i| {
                let t = i as f64 / CBRT_N as f64;
                (if t > 0.008856 {
                    t.cbrt()
                } else {
                    7.787 * t + 16.0 / 116.0
                }) as f32
            })
            .collect();
        let to_srgb = (0..CBRT_N + 2)
            .map(|i| {
                let l = (i as f64 / CBRT_N as f64).min(1.0);
                let c = if l <= 0.0031308 {
                    12.92 * l
                } else {
                    1.055 * l.powf(1.0 / 2.4) - 0.055
                };
                (c * 255.0) as f32
            })
            .collect();
        Tabs { to_lin, f, to_srgb }
    })
}

#[inline(always)]
fn lerp_tab(t: &[f32], x: f32) -> f32 {
    let p = x.clamp(0.0, 1.0) * CBRT_N as f32;
    let i = p as usize;
    let fr = p - i as f32;
    t[i] + (t[i + 1] - t[i]) * fr
}

#[inline(always)]
fn finv(f: f32) -> f32 {
    if f > 0.206_893 {
        f * f * f
    } else {
        (f - 16.0 / 116.0) / 7.787
    }
}

/// CLAHE on the L channel of an RGB image (channels beyond 3 are kept).
pub fn clahe_rgb(a: &Array3<u8>, clip_limit: f64, tiles_x: usize, tiles_y: usize) -> Array3<u8> {
    let (h, w, c) = a.dim();
    let src = a.as_slice().unwrap();
    let t = tabs();
    let n = h * w;
    let mut l8 = vec![0u8; n];
    let mut ab = vec![(0f32, 0f32); n];
    for (i, p) in src.chunks_exact(c).enumerate() {
        let (r, g, b) = (
            t.to_lin[p[0] as usize],
            t.to_lin[p[1] as usize],
            t.to_lin[p[2] as usize],
        );
        let x = M[0][0] * r + M[0][1] * g + M[0][2] * b;
        let y = M[1][0] * r + M[1][1] * g + M[1][2] * b;
        let z = M[2][0] * r + M[2][1] * g + M[2][2] * b;
        let (fx, fy, fz) = (lerp_tab(&t.f, x), lerp_tab(&t.f, y), lerp_tab(&t.f, z));
        let l = 116.0 * fy - 16.0;
        l8[i] = (l * (255.0 / 100.0)).round().clamp(0.0, 255.0) as u8;
        ab[i] = (fx - fy, fy - fz);
    }
    let eq = clahe_channel(&l8, h, w, clip_limit, tiles_x, tiles_y);
    let mut out = src.to_vec();
    for (i, p) in out.chunks_exact_mut(c).enumerate() {
        if eq[i] == l8[i] {
            continue; // unchanged L: keep the exact input pixel
        }
        let fy = (eq[i] as f32 * (100.0 / 255.0) + 16.0) / 116.0;
        let (dxy, dyz) = ab[i];
        let x = finv(fy + dxy) * XN;
        let y = finv(fy);
        let z = finv(fy - dyz) * ZN;
        let rgb = [
            MI[0][0] * x + MI[0][1] * y + MI[0][2] * z,
            MI[1][0] * x + MI[1][1] * y + MI[1][2] * z,
            MI[2][0] * x + MI[2][1] * y + MI[2][2] * z,
        ];
        for k in 0..3 {
            p[k] = (lerp_tab(&t.to_srgb, rgb[k]) + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    Array3::from_shape_vec((h, w, c), out).unwrap()
}

/// CLAHE for 1-channel (gray) or >= 3-channel (RGB) u8 images.
pub fn clahe_u8(a: &Array3<u8>, clip_limit: f64, tiles_x: usize, tiles_y: usize) -> Array3<u8> {
    let (h, w, c) = a.dim();
    if c >= 3 {
        return clahe_rgb(a, clip_limit, tiles_x, tiles_y);
    }
    let src = a.as_slice().unwrap();
    if c == 1 {
        return Array3::from_shape_vec((h, w, 1), clahe_channel(src, h, w, clip_limit, tiles_x, tiles_y)).unwrap();
    }
    // 2 channels: equalise each independently
    let mut out = src.to_vec();
    for k in 0..c {
        let ch: Vec<u8> = src.iter().skip(k).step_by(c).copied().collect();
        let eq = clahe_channel(&ch, h, w, clip_limit, tiles_x, tiles_y);
        for (o, v) in out.iter_mut().skip(k).step_by(c).zip(eq) {
            *o = v;
        }
    }
    Array3::from_shape_vec((h, w, c), out).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_stays_flat_and_rgb_gray_consistent() {
        let a = Array3::from_elem((20, 30, 1), 100u8);
        let o = clahe_u8(&a, 2.0, 4, 4);
        let v = o[[0, 0, 0]];
        assert!(o.iter().all(|&x| x == v));
        let g = Array3::from_shape_fn((16, 16, 3), |(y, x, _)| (y * 16 + x) as u8);
        let o = clahe_u8(&g, 2.0, 2, 2);
        for p in o.as_slice().unwrap().chunks(3) {
            assert!((p[0] as i32 - p[1] as i32).abs() <= 1 && (p[1] as i32 - p[2] as i32).abs() <= 1);
        }
    }
}
