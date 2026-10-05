//! Fast Gaussian noise: each sample is one 16-bit uniform mapped through a
//! 65536-entry table of the inverse normal CDF (bin midpoints), so a single
//! `next_u64` gives four samples. The distribution is a 65536-level
//! discretisation of N(0, 1), truncated at about +-4.17 sigma: fine for
//! augmentation and far faster than Box-Muller.

use crate::rng::Rng;
use std::sync::OnceLock;

/// Acklam's rational approximation of the inverse normal CDF (rel. error < 1.2e-9).
#[allow(clippy::excessive_precision)]
pub fn inv_norm_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let pl = 0.02425;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -inv_norm_cdf(1.0 - p)
    }
}

fn table() -> &'static [f32] {
    static T: OnceLock<Vec<f32>> = OnceLock::new();
    T.get_or_init(|| {
        (0..65536)
            .map(|i| inv_norm_cdf((i as f64 + 0.5) / 65536.0) as f32)
            .collect()
    })
}

/// Fill `out` with `mean + std * N(0, 1)` samples.
pub fn fill_normal(rng: &mut Rng, out: &mut [f32], mean: f32, std: f32) {
    let t = table();
    let mut chunks = out.chunks_exact_mut(4);
    for c in &mut chunks {
        let r = rng.next_u64();
        c[0] = mean + std * t[(r & 0xffff) as usize];
        c[1] = mean + std * t[((r >> 16) & 0xffff) as usize];
        c[2] = mean + std * t[((r >> 32) & 0xffff) as usize];
        c[3] = mean + std * t[(r >> 48) as usize];
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
        let mut r = rng.next_u64();
        for v in rem {
            *v = mean + std * t[(r & 0xffff) as usize];
            r >>= 16;
        }
    }
}

/// Fill `out` with uniform samples in `[lo, hi)`.
pub fn fill_uniform(rng: &mut Rng, out: &mut [f32], lo: f32, hi: f32) {
    for v in out.iter_mut() {
        *v = lo + (hi - lo) * ((rng.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments() {
        let mut rng = Rng::seed_from_u64(1);
        let mut v = vec![0f32; 200_001];
        fill_normal(&mut rng, &mut v, 2.0, 3.0);
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!((mean - 2.0).abs() < 0.03, "{mean}");
        assert!((var.sqrt() - 3.0).abs() < 0.03, "{var}");
        assert!((inv_norm_cdf(0.975) - 1.959964).abs() < 1e-5);
    }
}
