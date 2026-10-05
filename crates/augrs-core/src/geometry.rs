//! 2-D affine maps in continuous pixel coordinates (y axis pointing down).

/// `x' = a*x + b*y + c`, `y' = d*x + e*y + f`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine2 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Affine2 {
    pub const IDENTITY: Affine2 = Affine2 {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
        e: 1.0,
        f: 0.0,
    };

    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Affine2 { a, b, c, d, e, f }
    }

    pub fn translate(tx: f64, ty: f64) -> Self {
        Affine2::new(1.0, 0.0, tx, 0.0, 1.0, ty)
    }

    pub fn scale(sx: f64, sy: f64) -> Self {
        Affine2::new(sx, 0.0, 0.0, 0.0, sy, 0.0)
    }

    /// Rotation by `deg` degrees, counter-clockwise *as seen on screen*
    /// (same sign convention as OpenCV's `getRotationMatrix2D`).
    pub fn rotate_deg(deg: f64) -> Self {
        let (s, c) = deg.to_radians().sin_cos();
        Affine2::new(c, s, 0.0, -s, c, 0.0)
    }

    /// Shear: `x' = x + tan(sx)*y`, `y' = y + tan(sy)*x` (angles in degrees).
    pub fn shear_deg(sx: f64, sy: f64) -> Self {
        Affine2::new(1.0, sx.to_radians().tan(), 0.0, sy.to_radians().tan(), 1.0, 0.0)
    }

    /// Composition: first `self`, then `next`.
    pub fn then(&self, next: &Affine2) -> Affine2 {
        let n = next;
        Affine2 {
            a: n.a * self.a + n.b * self.d,
            b: n.a * self.b + n.b * self.e,
            c: n.a * self.c + n.b * self.f + n.c,
            d: n.d * self.a + n.e * self.d,
            e: n.d * self.b + n.e * self.e,
            f: n.d * self.c + n.e * self.f + n.f,
        }
    }

    pub fn det(&self) -> f64 {
        self.a * self.e - self.b * self.d
    }

    pub fn inverse(&self) -> Option<Affine2> {
        let det = self.det();
        if det.abs() < 1e-12 || !det.is_finite() {
            return None;
        }
        let ia = self.e / det;
        let ib = -self.b / det;
        let id = -self.d / det;
        let ie = self.a / det;
        Some(Affine2 {
            a: ia,
            b: ib,
            c: -(ia * self.c + ib * self.f),
            d: id,
            e: ie,
            f: -(id * self.c + ie * self.f),
        })
    }

    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f)
    }

    /// Apply only the linear part (for direction vectors).
    #[inline]
    pub fn apply_vec(&self, x: f64, y: f64) -> (f64, f64) {
        (self.a * x + self.b * y, self.d * x + self.e * y)
    }

    /// True when the map keeps axes aligned (no rotation/shear): boxes map to boxes exactly.
    pub fn is_axis_aligned(&self) -> bool {
        self.b.abs() < 1e-12 && self.d.abs() < 1e-12
    }

    /// Scale factor applied to keypoint `scale` (max column norm, like Albumentations' max(sx, sy)).
    pub fn scale_factor(&self) -> f64 {
        self.a.hypot(self.d).max(self.b.hypot(self.e))
    }
}

/// A projective map `(x, y) -> ((a x + b y + c) / w, (d x + e y + f) / w)`,
/// `w = g x + h y + i`, in continuous pixel coordinates. Row-major 3x3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography {
    pub m: [f64; 9],
}

impl Homography {
    pub const IDENTITY: Homography = Homography {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };

    pub fn from_affine(a: &Affine2) -> Self {
        Homography {
            m: [a.a, a.b, a.c, a.d, a.e, a.f, 0.0, 0.0, 1.0],
        }
    }

    /// Composition: first `self`, then `next`.
    pub fn then(&self, next: &Homography) -> Homography {
        let (a, b) = (&next.m, &self.m);
        let mut m = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                m[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
            }
        }
        Homography { m }
    }

    pub fn inverse(&self) -> Option<Homography> {
        let m = &self.m;
        let c00 = m[4] * m[8] - m[5] * m[7];
        let c01 = m[5] * m[6] - m[3] * m[8];
        let c02 = m[3] * m[7] - m[4] * m[6];
        let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
        if det.abs() < 1e-15 || !det.is_finite() {
            return None;
        }
        let inv = [
            c00,
            m[2] * m[7] - m[1] * m[8],
            m[1] * m[5] - m[2] * m[4],
            c01,
            m[0] * m[8] - m[2] * m[6],
            m[2] * m[3] - m[0] * m[5],
            c02,
            m[1] * m[6] - m[0] * m[7],
            m[0] * m[4] - m[1] * m[3],
        ];
        Some(Homography {
            m: inv.map(|v| v / det),
        })
    }

    /// Map a point; `None` when it lands on or behind the line at infinity.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = &self.m;
        let w = m[6] * x + m[7] * y + m[8];
        if w <= 1e-12 {
            return None;
        }
        Some(((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w))
    }

    /// Jacobian `[dx'/dx, dx'/dy, dy'/dx, dy'/dy]` at `(x, y)`.
    pub fn jacobian(&self, x: f64, y: f64) -> [f64; 4] {
        let m = &self.m;
        let w = m[6] * x + m[7] * y + m[8];
        let u = m[0] * x + m[1] * y + m[2];
        let v = m[3] * x + m[4] * y + m[5];
        let w2 = w * w;
        [
            (m[0] * w - u * m[6]) / w2,
            (m[1] * w - u * m[7]) / w2,
            (m[3] * w - v * m[6]) / w2,
            (m[4] * w - v * m[7]) / w2,
        ]
    }

    /// The map sending the 4 points `src` to `dst` (same as OpenCV's
    /// `getPerspectiveTransform`). `None` if the points are degenerate.
    #[allow(clippy::needless_range_loop)]
    pub fn from_quad(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Option<Homography> {
        // 8x8 linear system for (a, b, c, d, e, f, g, h) with i = 1
        let mut a = [[0.0f64; 9]; 8];
        for k in 0..4 {
            let (x, y) = src[k];
            let (u, v) = dst[k];
            a[2 * k] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u];
            a[2 * k + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v];
        }
        // Gaussian elimination with partial pivoting
        for col in 0..8 {
            let piv = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
            if a[piv][col].abs() < 1e-12 {
                return None;
            }
            a.swap(col, piv);
            for r in 0..8 {
                if r != col {
                    let f = a[r][col] / a[col][col];
                    if f != 0.0 {
                        for c in col..9 {
                            a[r][c] -= f * a[col][c];
                        }
                    }
                }
            }
        }
        let mut m = [0.0; 9];
        for i in 0..8 {
            m[i] = a[i][8] / a[i][i];
        }
        m[8] = 1.0;
        Some(Homography { m })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_roundtrip() {
        let m = Affine2::translate(-5.0, -3.0)
            .then(&Affine2::scale(1.3, 0.7))
            .then(&Affine2::shear_deg(10.0, -5.0))
            .then(&Affine2::rotate_deg(33.0))
            .then(&Affine2::translate(7.0, 2.0));
        let inv = m.inverse().unwrap();
        let (x, y) = m.apply(12.5, -4.25);
        let (bx, by) = inv.apply(x, y);
        assert!((bx - 12.5).abs() < 1e-9 && (by + 4.25).abs() < 1e-9);
    }

    #[test]
    fn homography_quad_and_inverse() {
        let src = [(3.0, 2.0), (95.0, 7.0), (90.0, 70.0), (5.0, 66.0)];
        let dst = [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)];
        let h = Homography::from_quad(src, dst).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            let (x, y) = h.apply(s.0, s.1).unwrap();
            assert!((x - d.0).abs() < 1e-9 && (y - d.1).abs() < 1e-9);
        }
        let inv = h.inverse().unwrap();
        let (x, y) = h.apply(40.0, 30.0).unwrap();
        let (bx, by) = inv.apply(x, y).unwrap();
        assert!((bx - 40.0).abs() < 1e-9 && (by - 30.0).abs() < 1e-9);
        // numeric jacobian
        let j = h.jacobian(40.0, 30.0);
        let e = 1e-6;
        let (x1, y1) = h.apply(40.0 + e, 30.0).unwrap();
        assert!(((x1 - x) / e - j[0]).abs() < 1e-4 && ((y1 - y) / e - j[2]).abs() < 1e-4);
        let a = Affine2::rotate_deg(30.0).then(&Affine2::translate(3.0, 4.0));
        let ha = Homography::from_affine(&a);
        let (px, py) = ha.apply(2.0, 5.0).unwrap();
        let (qx, qy) = a.apply(2.0, 5.0);
        assert!((px - qx).abs() < 1e-12 && (py - qy).abs() < 1e-12);
    }

    #[test]
    fn rotate_ccw_on_screen() {
        // y points down: a CCW rotation by 90 deg maps +x (right) to -y (up).
        let (x, y) = Affine2::rotate_deg(90.0).apply(1.0, 0.0);
        assert!(x.abs() < 1e-12 && (y + 1.0).abs() < 1e-12);
    }
}
