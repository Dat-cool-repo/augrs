//! Border extrapolation modes (same semantics as OpenCV's `BORDER_*`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BorderMode {
    /// `iiiiii|abcdefgh|iiiiiii` with a fill value.
    #[default]
    Constant,
    /// `aaaaaa|abcdefgh|hhhhhhh`
    Replicate,
    /// `fedcba|abcdefgh|hgfedcb`
    Reflect,
    /// `cdefgh|abcdefgh|abcdefg`
    Wrap,
    /// `gfedcb|abcdefgh|gfedcba` (OpenCV's default)
    #[serde(alias = "reflect_101")]
    Reflect101,
}

/// Map a possibly out-of-range index into `[0, n)`. Returns `None` for
/// [`BorderMode::Constant`] when the index is outside.
#[inline]
pub fn border_index(i: isize, n: usize, mode: BorderMode) -> Option<usize> {
    let n_i = n as isize;
    if i >= 0 && i < n_i {
        return Some(i as usize);
    }
    if n == 0 {
        return None;
    }
    match mode {
        BorderMode::Constant => None,
        BorderMode::Replicate => Some(i.clamp(0, n_i - 1) as usize),
        BorderMode::Wrap => Some(i.rem_euclid(n_i) as usize),
        BorderMode::Reflect => {
            let p = 2 * n_i;
            let m = i.rem_euclid(p);
            Some(if m >= n_i { p - 1 - m } else { m } as usize)
        }
        BorderMode::Reflect101 => {
            if n == 1 {
                return Some(0);
            }
            let p = 2 * n_i - 2;
            let m = i.rem_euclid(p);
            Some(if m >= n_i { p - m } else { m } as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(mode: BorderMode, n: usize) -> Vec<Option<usize>> {
        (-3..(n as isize + 3)).map(|i| border_index(i, n, mode)).collect()
    }

    #[test]
    fn modes_match_opencv() {
        // n = 4: abcd
        let s = |v: &[usize]| v.iter().map(|&x| Some(x)).collect::<Vec<_>>();
        assert_eq!(seq(BorderMode::Replicate, 4), s(&[0, 0, 0, 0, 1, 2, 3, 3, 3, 3]));
        assert_eq!(seq(BorderMode::Reflect, 4), s(&[2, 1, 0, 0, 1, 2, 3, 3, 2, 1]));
        assert_eq!(seq(BorderMode::Reflect101, 4), s(&[3, 2, 1, 0, 1, 2, 3, 2, 1, 0]));
        assert_eq!(seq(BorderMode::Wrap, 4), s(&[1, 2, 3, 0, 1, 2, 3, 0, 1, 2]));
        assert_eq!(border_index(-1, 4, BorderMode::Constant), None);
        assert_eq!(border_index(5, 1, BorderMode::Reflect101), Some(0));
    }
}
