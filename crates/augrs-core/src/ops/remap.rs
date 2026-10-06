//! Generic inverse-mapping sampler: the source position of every output pixel
//! comes from a per-row coordinate generator. Used by perspective warps and
//! displacement fields (elastic). Coordinates are in pixel-index space (the
//! centre of pixel `i` is at `i`); non-finite coordinates count as outside.
//!
//! Linear sampling uses the same 8-bit sub-pixel fixed point as the affine
//! u8 kernel for `u8` and f32 weights otherwise.

use crate::border::{BorderMode, border_index};
use crate::buffer::{Element, data, from_vec};
use crate::ops::resize::Interp;
use ndarray::Array3;

/// Larger than `MAX_SIDE`: every in-image position is represented exactly.
const LIM: f64 = 2_000_000.0;

#[inline(always)]
fn sane(x: f64) -> f64 {
    if x.is_nan() { -LIM } else { x.clamp(-LIM, LIM) }
}

/// Sample `a` into an `oh x ow` image. `coords(v, xs, ys)` fills the source
/// positions of output row `v` (both slices have length `ow`).
pub fn remap<T: Element, F>(
    a: &Array3<T>,
    oh: usize,
    ow: usize,
    interp: Interp,
    mode: BorderMode,
    fill: &[f64],
    coords: F,
) -> Array3<T>
where
    F: Fn(usize, &mut [f64], &mut [f64]),
{
    let (h, w, c) = a.dim();
    let src = data(a);
    let fillv: Vec<T> = (0..c)
        .map(|k| T::from_f64(*fill.get(k).or(fill.last()).unwrap_or(&0.0)))
        .collect();
    let fillf: Vec<f32> = fillv.iter().map(|v| v.to_f32()).collect();
    let mut out = vec![T::default(); oh * ow * c];
    let mut xs = vec![0f64; ow];
    let mut ys = vec![0f64; ow];
    let is_u8 = T::NAME == "uint8";
    let stride = w * c;
    for v in 0..oh {
        coords(v, &mut xs, &mut ys);
        let orow = &mut out[v * ow * c..(v + 1) * ow * c];
        for (u, o) in orow.chunks_exact_mut(c).enumerate() {
            let (sx, sy) = (sane(xs[u]), sane(ys[u]));
            if interp == Interp::Nearest {
                let xi = (sx + 0.5).floor() as isize;
                let yi = (sy + 0.5).floor() as isize;
                match (border_index(xi, w, mode), border_index(yi, h, mode)) {
                    (Some(x), Some(y)) => o.copy_from_slice(&src[(y * w + x) * c..(y * w + x) * c + c]),
                    _ => o.copy_from_slice(&fillv),
                }
                continue;
            }
            let (x0, y0, fx, fy) = if is_u8 {
                // 8-bit sub-pixel position, like the affine u8 kernel
                let x8 = (sx * 256.0).round() as i64;
                let y8 = (sy * 256.0).round() as i64;
                (x8 >> 8, y8 >> 8, (x8 & 255) as f32 / 256.0, (y8 & 255) as f32 / 256.0)
            } else {
                let (fx0, fy0) = (sx.floor(), sy.floor());
                (fx0 as i64, fy0 as i64, (sx - fx0) as f32, (sy - fy0) as f32)
            };
            let inside = x0 >= 0 && y0 >= 0 && x0 + 1 < w as i64 && y0 + 1 < h as i64;
            if !inside && mode == BorderMode::Constant && (x0 < -1 || y0 < -1 || x0 >= w as i64 || y0 >= h as i64) {
                o.copy_from_slice(&fillv);
                continue;
            }
            let xs2 = [
                border_index(x0 as isize, w, mode),
                border_index(x0 as isize + 1, w, mode),
            ];
            let ys2 = [
                border_index(y0 as isize, h, mode),
                border_index(y0 as isize + 1, h, mode),
            ];
            for ch in 0..c {
                let get = |yy: Option<usize>, xx: Option<usize>| match (yy, xx) {
                    (Some(y), Some(x)) => src[y * stride + x * c + ch].to_f32(),
                    _ => fillf[ch],
                };
                let (p00, p01, p10, p11) = (
                    get(ys2[0], xs2[0]),
                    get(ys2[0], xs2[1]),
                    get(ys2[1], xs2[0]),
                    get(ys2[1], xs2[1]),
                );
                o[ch] = if is_u8 {
                    // integer blend with 8-bit weights (exact, same rounding as warp_affine_u8)
                    let (ix, iy) = ((fx * 256.0) as i32, (fy * 256.0) as i32);
                    let s = p00 as i32 * (256 - ix) * (256 - iy)
                        + p01 as i32 * ix * (256 - iy)
                        + p10 as i32 * (256 - ix) * iy
                        + p11 as i32 * ix * iy;
                    T::from_f32(((s + (1 << 15)) >> 16) as f32)
                } else {
                    let top = p00 + (p01 - p00) * fx;
                    let bot = p10 + (p11 - p10) * fx;
                    T::from_f32(top + (bot - top) * fy)
                };
            }
        }
    }
    from_vec(oh, ow, c, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Affine2;

    #[test]
    fn remap_matches_affine_warp() {
        let a = Array3::from_shape_fn((20, 30, 3), |(y, x, k)| ((y * 37 + x * 11 + k * 80) % 256) as u8);
        let m = Affine2::translate(-15.0, -10.0)
            .then(&Affine2::rotate_deg(23.0))
            .then(&Affine2::scale(1.2, 0.9))
            .then(&Affine2::translate(14.0, 11.0));
        let inv = m.inverse().unwrap();
        for mode in [BorderMode::Constant, BorderMode::Reflect101] {
            for interp in [Interp::Linear, Interp::Nearest] {
                let w = crate::ops::warp_affine_u8(&a, &inv, 20, 30, interp, mode, &[0.0]);
                let r = remap(&a, 20, 30, interp, mode, &[0.0], |v, xs, ys| {
                    for u in 0..xs.len() {
                        let (x, y) = inv.apply(u as f64 + 0.5, v as f64 + 0.5);
                        xs[u] = x - 0.5;
                        ys[u] = y - 0.5;
                    }
                });
                let maxd = w
                    .iter()
                    .zip(r.iter())
                    .map(|(p, q)| (*p as i32 - *q as i32).abs())
                    .max()
                    .unwrap();
                assert!(maxd <= 1, "{mode:?} {interp:?} {maxd}");
            }
        }
    }
}
