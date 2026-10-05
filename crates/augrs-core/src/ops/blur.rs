//! Separable Gaussian blur with reflect-101 borders (OpenCV default).

use crate::border::{BorderMode, border_index};
use crate::buffer::{Element, data, from_vec};
use ndarray::Array3;

/// 1-D Gaussian kernel. If `ksize == 0` the size is `int(sigma * 3.5) * 2 + 1`
/// (Albumentations/PIL rule). If `sigma <= 0` it is derived from `ksize` with
/// OpenCV's rule `0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`.
pub fn gaussian_kernel_1d(sigma: f64, ksize: usize) -> Vec<f32> {
    let mut size = if ksize == 0 {
        (sigma.max(0.0) * 3.5) as usize * 2 + 1
    } else {
        ksize
    };
    if size % 2 == 0 {
        size += 1;
    }
    let sigma = if sigma <= 0.0 {
        0.3 * ((size as f64 - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };
    let r = (size / 2) as isize;
    let k: Vec<f64> = (-r..=r).map(|x| (-0.5 * (x as f64 / sigma).powi(2)).exp()).collect();
    let s: f64 = k.iter().sum();
    k.iter().map(|v| (v / s) as f32).collect()
}

/// Convolve rows then columns with `kernel` (odd length).
pub fn gaussian_blur<T: Element>(a: &Array3<T>, kernel: &[f32]) -> Array3<T> {
    let (h, w, c) = a.dim();
    if kernel.len() <= 1 || h == 0 || w == 0 {
        return a.clone();
    }
    let src = data(a);
    let r = kernel.len() / 2;
    let rowlen = w * c;
    // horizontal pass into f32 buffer
    let mut tmp = vec![0f32; h * rowlen];
    let mut padded = vec![0f32; (w + 2 * r) * c];
    let cols: Vec<usize> = (0..w + 2 * r)
        .map(|x| border_index(x as isize - r as isize, w, BorderMode::Reflect101).unwrap())
        .collect();
    for y in 0..h {
        let row = &src[y * rowlen..(y + 1) * rowlen];
        for (x, &sx) in cols.iter().enumerate() {
            for k in 0..c {
                padded[x * c + k] = row[sx * c + k].to_f32();
            }
        }
        let o = &mut tmp[y * rowlen..(y + 1) * rowlen];
        for (j, &kj) in kernel.iter().enumerate() {
            let s = &padded[j * c..j * c + rowlen];
            for (ov, &sv) in o.iter_mut().zip(s.iter()) {
                *ov += kj * sv;
            }
        }
    }
    // vertical pass
    let mut out = Vec::with_capacity(h * rowlen);
    let mut acc = vec![0f32; rowlen];
    for y in 0..h {
        acc.iter_mut().for_each(|v| *v = 0.0);
        for (j, &kj) in kernel.iter().enumerate() {
            let sy = border_index(y as isize + j as isize - r as isize, h, BorderMode::Reflect101).unwrap();
            let s = &tmp[sy * rowlen..(sy + 1) * rowlen];
            for (av, &sv) in acc.iter_mut().zip(s.iter()) {
                *av += kj * sv;
            }
        }
        out.extend(acc.iter().map(|&v| T::from_f32(v)));
    }
    from_vec(h, w, c, out)
}

super::avx2_dispatch!(
    /// u8 Gaussian blur in fixed point (kernel quantised to 8 fractional bits,
    /// u16 row buffer, u32 column accumulation), like OpenCV's bit-exact 8U path.
    pub fn gaussian_blur_u8(a: &Array3<u8>, kernel: &[f32]) -> Array3<u8> = gaussian_blur_u8_impl
);

#[inline(always)]
fn gaussian_blur_u8_impl(a: &Array3<u8>, kernel: &[f32]) -> Array3<u8> {
    let (h, w, c) = a.dim();
    if kernel.len() <= 1 || h == 0 || w == 0 {
        return a.clone();
    }
    let r = kernel.len() / 2;
    // quantise so the taps sum to exactly 256
    let mut q: Vec<u32> = kernel.iter().map(|&k| (k * 256.0).round().max(0.0) as u32).collect();
    let s: i64 = q.iter().map(|&v| v as i64).sum();
    q[r] = (q[r] as i64 + 256 - s).max(0) as u32;
    let src = data(a);
    let rowlen = w * c;
    let mut tmp = vec![0u16; h * rowlen];
    let mut padded = vec![0u16; (w + 2 * r) * c];
    let cols: Vec<usize> = (0..w + 2 * r)
        .map(|x| border_index(x as isize - r as isize, w, BorderMode::Reflect101).unwrap())
        .collect();
    let mut acc = vec![0u32; rowlen];
    for y in 0..h {
        let row = &src[y * rowlen..(y + 1) * rowlen];
        // interior is a straight copy; only the 2r border columns need the index map
        for (x, &sx) in cols[..r].iter().enumerate() {
            for k in 0..c {
                padded[x * c + k] = row[sx * c + k] as u16;
            }
        }
        for (d, &v) in padded[r * c..r * c + rowlen].iter_mut().zip(row.iter()) {
            *d = v as u16;
        }
        for (x, &sx) in cols.iter().enumerate().skip(r + w) {
            for k in 0..c {
                padded[x * c + k] = row[sx * c + k] as u16;
            }
        }
        acc.iter_mut().for_each(|v| *v = 0);
        for (j, &qj) in q.iter().enumerate() {
            let s = &padded[j * c..j * c + rowlen];
            for (av, &sv) in acc.iter_mut().zip(s.iter()) {
                *av += qj * sv as u32;
            }
        }
        for (t, &v) in tmp[y * rowlen..(y + 1) * rowlen].iter_mut().zip(acc.iter()) {
            *t = v as u16; // <= 255 * 256
        }
    }
    let mut out = vec![0u8; h * rowlen];
    for y in 0..h {
        acc.iter_mut().for_each(|v| *v = 0);
        for (j, &qj) in q.iter().enumerate() {
            let sy = border_index(y as isize + j as isize - r as isize, h, BorderMode::Reflect101).unwrap();
            let s = &tmp[sy * rowlen..(sy + 1) * rowlen];
            for (av, &sv) in acc.iter_mut().zip(s.iter()) {
                *av += qj * sv as u32;
            }
        }
        for (o, &v) in out[y * rowlen..(y + 1) * rowlen].iter_mut().zip(acc.iter()) {
            *o = ((v + (1 << 15)) >> 16) as u8;
        }
    }
    from_vec(h, w, c, out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn u8_fixed_point_close_to_float() {
        let a = ndarray::Array3::from_shape_fn((23, 31, 3), |(y, x, k)| ((y * 37 + x * 11 + k * 80) % 256) as u8);
        for (s, k) in [(1.0, 0), (2.0, 7), (0.8, 3)] {
            let kern = gaussian_kernel_1d(s, k);
            let f = gaussian_blur(&a, &kern);
            let q = gaussian_blur_u8(&a, &kern);
            for (x, y) in f.iter().zip(q.iter()) {
                assert!((*x as i32 - *y as i32).abs() <= 1);
            }
        }
    }

    use super::*;

    #[test]
    fn kernel_sums_to_one() {
        for (s, k) in [(1.0, 0), (0.0, 5), (2.5, 7), (0.6, 0)] {
            let kern = gaussian_kernel_1d(s, k);
            assert_eq!(kern.len() % 2, 1);
            assert!((kern.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert_eq!(gaussian_kernel_1d(1.0, 0).len(), 7);
    }

    #[test]
    fn blur_constant_is_constant() {
        let a = Array3::from_elem((10, 12, 3), 123u8);
        let b = gaussian_blur(&a, &gaussian_kernel_1d(2.0, 0));
        assert!(b.iter().all(|&v| v == 123));
    }
}
