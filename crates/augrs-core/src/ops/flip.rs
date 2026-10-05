//! Exact pixel-permutation ops: flips, crops and padding.

use crate::border::{BorderMode, border_index};
use crate::buffer::{Element, data, from_vec};
use ndarray::Array3;

/// Mirror left-right.
pub fn hflip<T: Element>(a: &Array3<T>) -> Array3<T> {
    let (h, w, c) = a.dim();
    let src = data(a);
    let mut out = vec![T::default(); h * w * c];
    let row = w * c;
    if row == 0 {
        return from_vec(h, w, c, out);
    }
    for (orow, irow) in out.chunks_exact_mut(row).zip(src.chunks_exact(row)) {
        match c {
            1 => {
                for (o, i) in orow.iter_mut().zip(irow.iter().rev()) {
                    *o = *i;
                }
            }
            3 => {
                for (o, i) in orow.chunks_exact_mut(3).zip(irow.chunks_exact(3).rev()) {
                    o.copy_from_slice(i);
                }
            }
            _ => {
                for (o, i) in orow.chunks_exact_mut(c).zip(irow.chunks_exact(c).rev()) {
                    o.copy_from_slice(i);
                }
            }
        }
    }
    from_vec(h, w, c, out)
}

/// Mirror top-bottom.
pub fn vflip<T: Element>(a: &Array3<T>) -> Array3<T> {
    let (h, w, c) = a.dim();
    let src = data(a);
    let row = w * c;
    let mut out = Vec::with_capacity(h * row);
    if row > 0 {
        for irow in src.chunks_exact(row).rev() {
            out.extend_from_slice(irow);
        }
    }
    from_vec(h, w, c, out)
}

/// Copy the window `[x0, x1) x [y0, y1)` (must lie inside the image).
pub fn crop<T: Element>(a: &Array3<T>, x0: usize, y0: usize, x1: usize, y1: usize) -> Array3<T> {
    let (_, w, c) = a.dim();
    let src = data(a);
    let (ow, oh) = (x1 - x0, y1 - y0);
    let mut out = Vec::with_capacity(oh * ow * c);
    for y in y0..y1 {
        let s = (y * w + x0) * c;
        out.extend_from_slice(&src[s..s + ow * c]);
    }
    from_vec(oh, ow, c, out)
}

/// Pad with the given border mode. `fill` holds one value per channel
/// (or a single value broadcast to all channels).
pub fn pad<T: Element>(
    a: &Array3<T>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    mode: BorderMode,
    fill: &[f64],
) -> Array3<T> {
    let (h, w, c) = a.dim();
    let src = data(a);
    let (oh, ow) = (h + top + bottom, w + left + right);
    let fillv: Vec<T> = (0..c)
        .map(|k| T::from_f64(*fill.get(k).or(fill.last()).unwrap_or(&0.0)))
        .collect();
    let mut out = Vec::with_capacity(oh * ow * c);
    // column map, computed once
    let cols: Vec<Option<usize>> = (0..ow)
        .map(|x| border_index(x as isize - left as isize, w, mode))
        .collect();
    for y in 0..oh {
        match border_index(y as isize - top as isize, h, mode) {
            None => {
                for _ in 0..ow {
                    out.extend_from_slice(&fillv);
                }
            }
            Some(sy) => {
                let row = &src[sy * w * c..(sy + 1) * w * c];
                for (x, col) in cols.iter().enumerate() {
                    if x == left && w > 0 {
                        // fast path for the interior
                        out.extend_from_slice(row);
                        continue;
                    }
                    if x > left && x < left + w {
                        continue;
                    }
                    match col {
                        Some(sx) => out.extend_from_slice(&row[sx * c..sx * c + c]),
                        None => out.extend_from_slice(&fillv),
                    }
                }
            }
        }
    }
    from_vec(oh, ow, c, out)
}

/// Swap rows and columns: `out[x][y] = in[y][x]` (output is `w x h`).
pub fn transpose<T: Element>(a: &Array3<T>) -> Array3<T> {
    rot_generic(a, |x, y, _h, _w| (y, x), true)
}

/// Rotate by `k * 90` degrees counter-clockwise (like `np.rot90(img, k)`).
pub fn rot90<T: Element>(a: &Array3<T>, k: u8) -> Array3<T> {
    match k % 4 {
        0 => a.clone(),
        // out[i, j] = in[j, W-1-i]  <=>  input (x, y) goes to (y, W-1-x)
        1 => rot_generic(a, |x, y, _h, w| (y, w - 1 - x), true),
        2 => vflip(&hflip(a)),
        // out[i, j] = in[H-1-j, i]  <=>  input (x, y) goes to (H-1-y, x)
        _ => rot_generic(a, |x, y, h, _w| (h - 1 - y, x), true),
    }
}

/// Generic pixel permutation: input pixel `(x, y)` goes to `f(x, y, h, w)`.
/// Tiled so that both the reads and the scattered writes stay in cache.
fn rot_generic<T: Element, F: Fn(usize, usize, usize, usize) -> (usize, usize)>(
    a: &Array3<T>,
    f: F,
    swap: bool,
) -> Array3<T> {
    let (h, w, c) = a.dim();
    let (oh, ow) = if swap { (w, h) } else { (h, w) };
    let src = data(a);
    let mut out = vec![T::default(); h * w * c];
    const TILE: usize = 32;
    for ty in (0..h).step_by(TILE) {
        for tx in (0..w).step_by(TILE) {
            for y in ty..(ty + TILE).min(h) {
                for x in tx..(tx + TILE).min(w) {
                    let (ox, oy) = f(x, y, h, w);
                    let i = (y * w + x) * c;
                    let o = (oy * ow + ox) * c;
                    out[o..o + c].copy_from_slice(&src[i..i + c]);
                }
            }
        }
    }
    from_vec(oh, ow, c, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(h: usize, w: usize, c: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, c), |(y, x, k)| (y * 31 + x * 7 + k * 3) as u8)
    }

    #[test]
    fn flips_are_involutions() {
        let a = img(5, 7, 3);
        assert_eq!(hflip(&hflip(&a)), a);
        assert_eq!(vflip(&vflip(&a)), a);
        let f = hflip(&a);
        assert_eq!(f[[2, 0, 1]], a[[2, 6, 1]]);
        let v = vflip(&a);
        assert_eq!(v[[0, 3, 2]], a[[4, 3, 2]]);
    }

    #[test]
    fn rot90_and_transpose_match_numpy_semantics() {
        let a = img(3, 5, 2);
        let r1 = rot90(&a, 1);
        assert_eq!(r1.dim(), (5, 3, 2));
        // np.rot90: out[i, j] = in[j, W-1-i]
        assert_eq!(r1[[0, 0, 1]], a[[0, 4, 1]]);
        assert_eq!(r1[[4, 2, 0]], a[[2, 0, 0]]);
        assert_eq!(rot90(&rot90(&a, 1), 3), a);
        assert_eq!(rot90(&rot90(&a, 2), 2), a);
        assert_eq!(rot90(&a, 2), vflip(&hflip(&a)));
        let t = transpose(&a);
        assert_eq!(t[[4, 2, 1]], a[[2, 4, 1]]);
        assert_eq!(transpose(&t), a);
    }

    #[test]
    fn crop_and_pad() {
        let a = img(6, 8, 2);
        let c = crop(&a, 2, 1, 5, 4);
        assert_eq!(c.dim(), (3, 3, 2));
        assert_eq!(c[[0, 0, 1]], a[[1, 2, 1]]);
        let p = pad(&a, 1, 2, 3, 4, BorderMode::Constant, &[9.0]);
        assert_eq!(p.dim(), (9, 15, 2));
        assert_eq!(p[[0, 0, 0]], 9);
        assert_eq!(p[[1, 3, 1]], a[[0, 0, 1]]);
        assert_eq!(p[[6, 10, 0]], a[[5, 7, 0]]);
        let r = pad(&a, 1, 0, 1, 0, BorderMode::Reflect101, &[0.0]);
        assert_eq!(r[[0, 0, 0]], a[[1, 1, 0]]);
        assert_eq!(crop(&r, 1, 1, 9, 7), a);
    }
}
