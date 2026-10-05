//! Small, dependency-free, deterministic RNG (xoshiro256++ seeded via splitmix64).
//!
//! The algorithm is fixed forever so that a given seed reproduces the same
//! augmentations across versions of dependencies and platforms.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Mix a base seed with an index into a new, well-separated seed.
#[inline]
pub fn derive_seed(base: u64, index: u64) -> u64 {
    let mut s = base ^ 0xA076_1D64_78BD_642F;
    let a = splitmix64(&mut s);
    let mut t = index.wrapping_mul(0xE703_7ED1_A0B4_28DB) ^ a;
    splitmix64(&mut t)
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut st = seed;
        let s = [
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
        ];
        Rng { s }
    }

    /// Seed from process entropy (non-deterministic).
    pub fn from_entropy() -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let mut h = RandomState::new().build_hasher();
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        h.write_u64(t);
        h.write_usize(&h as *const _ as usize);
        Rng::seed_from_u64(h.finish())
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)` (returns `lo` when `lo == hi`).
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = self.f64();
        if lo == hi { lo } else { lo + (hi - lo) * u }
    }

    /// Uniform integer in the inclusive range `[lo, hi]`.
    #[inline]
    pub fn int_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            // still consume a draw so stream structure does not depend on params
            let _ = self.next_u64();
            return lo;
        }
        let span = (hi - lo) as u64 + 1;
        // Lemire's multiply-shift (bias is negligible for our spans)
        let r = ((self.next_u64() as u128 * span as u128) >> 64) as u64;
        lo + r as i64
    }

    /// Standard normal sample (Box-Muller; one sample per call).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.f64(); // (0, 1]
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// `true` with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.int_inclusive(0, i as i64) as usize;
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::seed_from_u64(42);
        let mut b = Rng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Rng::seed_from_u64(43);
        assert_ne!(Rng::seed_from_u64(42).next_u64(), c.next_u64());
    }

    #[test]
    fn ranges() {
        let mut r = Rng::seed_from_u64(1);
        for _ in 0..10_000 {
            let x = r.uniform(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&x));
            let i = r.int_inclusive(3, 7);
            assert!((3..=7).contains(&i));
        }
    }

    #[test]
    fn int_covers_range() {
        let mut r = Rng::seed_from_u64(7);
        let mut seen = [false; 5];
        for _ in 0..1000 {
            seen[r.int_inclusive(0, 4) as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
