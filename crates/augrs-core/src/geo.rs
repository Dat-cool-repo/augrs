//! Geometric operations: one sampled operation is applied to every image, every
//! mask, the boxes and the keypoints through the *same* point map (an exact
//! affine map, a homography, or a smooth displacement field).

use crate::border::BorderMode;
use crate::buffer::Buf;
use crate::error::{Result, input};
use crate::geometry::{Affine2, Homography};
use crate::map_buf;
use crate::ops::remap::remap;
use crate::ops::resize::{Interp, Rect};
use crate::ops::warp::warp_affine_u8;
use crate::ops::{crop, hflip, pad, resize, resize_u8, rot90, transpose, vflip, warp_affine};
use crate::pipeline::Sample;
use crate::targets::{
    BboxMethod, transform_boxes, transform_boxes_with, transform_keypoints, transform_keypoints_with,
};

/// A dense displacement field: output pixel `(u, v)` samples the input at
/// `(u + dx, v + dy)` (pixel-index coordinates), stored row-major `h x w`.
#[derive(Clone, Debug)]
pub(crate) struct DispField {
    pub h: usize,
    pub w: usize,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
}

impl DispField {
    /// Displacement at a continuous point (bilinear between pixel centres, edge-clamped).
    fn at(&self, x: f64, y: f64) -> (f64, f64) {
        let fx = (x - 0.5).clamp(0.0, (self.w - 1) as f64);
        let fy = (y - 0.5).clamp(0.0, (self.h - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (ax, ay) = (fx - x0 as f64, fy - y0 as f64);
        let lerp = |f: &[f32]| {
            let g = |yy: usize, xx: usize| f[yy * self.w + xx] as f64;
            let top = g(y0, x0) + (g(y0, x1) - g(y0, x0)) * ax;
            let bot = g(y1, x0) + (g(y1, x1) - g(y1, x0)) * ax;
            top + (bot - top) * ay
        };
        (lerp(&self.dx), lerp(&self.dy))
    }

    /// Largest displacement magnitude (per axis) in the field.
    fn max_disp(&self) -> f64 {
        let m = |f: &[f32]| f.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
        m(&self.dx).max(m(&self.dy))
    }

    /// Fixed-point iteration `p <- q - d(p)` from `start`; returns the point and its residual.
    fn iterate(&self, qx: f64, qy: f64, start: (f64, f64)) -> ((f64, f64), f64) {
        let (mut px, mut py) = start;
        for _ in 0..32 {
            let (dx, dy) = self.at(px, py);
            let (nx, ny) = (qx - dx, qy - dy);
            let step = (nx - px).abs() + (ny - py).abs();
            px = nx;
            py = ny;
            if step < 1e-5 {
                break;
            }
        }
        let (dx, dy) = self.at(px, py);
        ((px, py), (px + dx - qx).hypot(py + dy - qy))
    }

    /// Forward map of an input point `q`: an output point `p` with `p + d(p) = q`.
    /// Fixed-point iteration converges for smooth fields; where the field folds
    /// (strong, small-scale deformations) the best pixel centre in a window of the
    /// maximum displacement seeds a second attempt.
    fn forward(&self, qx: f64, qy: f64, dmax: f64) -> (f64, f64) {
        let (p, res) = self.iterate(qx, qy, (qx, qy));
        if res < 0.05 {
            return p;
        }
        let r = dmax.ceil() as i64 + 1;
        let (cu, cv) = (qx.floor() as i64, qy.floor() as i64);
        let mut best = (f64::INFINITY, (qx, qy));
        for v in (cv - r).max(0)..(cv + r + 1).min(self.h as i64) {
            for u in (cu - r).max(0)..(cu + r + 1).min(self.w as i64) {
                let i = v as usize * self.w + u as usize;
                let (sx, sy) = (u as f64 + 0.5 + self.dx[i] as f64, v as f64 + 0.5 + self.dy[i] as f64);
                let d = (sx - qx).hypot(sy - qy);
                if d < best.0 {
                    best = (d, (u as f64 + 0.5, v as f64 + 0.5));
                }
            }
        }
        let (p2, res2) = self.iterate(qx, qy, best.1);
        if res2 < res { p2 } else { p }
    }

    /// Bounding box of the output pixels whose sample point lies inside the input box
    /// (exact for the image/mask sampling), unioned with the mapped box boundary.
    fn map_box(&self, b: &crate::targets::BBox, dmax: f64) -> (f64, f64, f64, f64) {
        let mut r = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        let pad = dmax.ceil() + 1.0;
        let u0 = ((b.x0 - pad).floor().max(0.0)) as usize;
        let u1 = ((b.x1 + pad).ceil().max(0.0) as usize).min(self.w);
        let v0 = ((b.y0 - pad).floor().max(0.0)) as usize;
        let v1 = ((b.y1 + pad).ceil().max(0.0) as usize).min(self.h);
        for v in v0..v1 {
            for u in u0..u1 {
                let i = v * self.w + u;
                let sx = u as f64 + 0.5 + self.dx[i] as f64;
                let sy = v as f64 + 0.5 + self.dy[i] as f64;
                if sx >= b.x0 && sx <= b.x1 && sy >= b.y0 && sy <= b.y1 {
                    let (uf, vf) = (u as f64, v as f64);
                    r = (r.0.min(uf), r.1.min(vf), r.2.max(uf + 1.0), r.3.max(vf + 1.0));
                }
            }
        }
        // the boundary keeps sub-pixel boxes (no sample point inside) alive
        let nx = ((b.x1 - b.x0).ceil() as usize).max(1);
        let ny = ((b.y1 - b.y0).ceil() as usize).max(1);
        let mut add = |x: f64, y: f64| {
            let (px, py) = self.forward(x, y, dmax);
            r = (r.0.min(px), r.1.min(py), r.2.max(px), r.3.max(py));
        };
        for i in 0..=nx {
            let x = b.x0 + (b.x1 - b.x0) * i as f64 / nx as f64;
            add(x, b.y0);
            add(x, b.y1);
        }
        for j in 1..ny {
            let y = b.y0 + (b.y1 - b.y0) * j as f64 / ny as f64;
            add(b.x0, y);
            add(b.x1, y);
        }
        r
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Geo {
    HFlip,
    VFlip,
    Transpose,
    /// Counter-clockwise rotation by `k * 90` degrees.
    Rot90(u8),
    /// Crop `rect` and resize it to `oh x ow` (a plain crop when sizes match).
    CropResize {
        rect: Rect,
        oh: usize,
        ow: usize,
        interp: Interp,
        mask_interp: Interp,
    },
    Pad {
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        mode: BorderMode,
        fill: Vec<f64>,
        fill_mask: Vec<f64>,
    },
    Affine {
        m: Affine2,
        oh: usize,
        ow: usize,
        interp: Interp,
        mask_interp: Interp,
        mode: BorderMode,
        fill: Vec<f64>,
        fill_mask: Vec<f64>,
        method: BboxMethod,
    },
    Perspective {
        hm: Homography,
        oh: usize,
        ow: usize,
        interp: Interp,
        mask_interp: Interp,
        mode: BorderMode,
        fill: Vec<f64>,
        fill_mask: Vec<f64>,
    },
    Field {
        field: DispField,
        interp: Interp,
        mask_interp: Interp,
        mode: BorderMode,
        fill: Vec<f64>,
        fill_mask: Vec<f64>,
    },
}

/// How points move under a [`Geo`].
enum PointMap<'a> {
    Affine(Affine2),
    Proj(Homography),
    Field(&'a DispField),
}

impl Geo {
    /// The point map and output size for an `h x w` input.
    fn map(&self, h: usize, w: usize) -> (PointMap<'_>, usize, usize) {
        let (hf, wf) = (h as f64, w as f64);
        let aff = |m: Affine2, oh, ow| (PointMap::Affine(m), oh, ow);
        match self {
            Geo::HFlip => aff(Affine2::new(-1.0, 0.0, wf, 0.0, 1.0, 0.0), h, w),
            Geo::VFlip => aff(Affine2::new(1.0, 0.0, 0.0, 0.0, -1.0, hf), h, w),
            Geo::Transpose => aff(Affine2::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0), w, h),
            Geo::Rot90(k) => match k % 4 {
                0 => aff(Affine2::IDENTITY, h, w),
                1 => aff(Affine2::new(0.0, 1.0, 0.0, -1.0, 0.0, wf), w, h),
                2 => aff(Affine2::new(-1.0, 0.0, wf, 0.0, -1.0, hf), h, w),
                _ => aff(Affine2::new(0.0, -1.0, hf, 1.0, 0.0, 0.0), w, h),
            },
            Geo::CropResize { rect, oh, ow, .. } => {
                let m = Affine2::translate(-(rect.x0 as f64), -(rect.y0 as f64)).then(&Affine2::scale(
                    *ow as f64 / rect.w() as f64,
                    *oh as f64 / rect.h() as f64,
                ));
                aff(m, *oh, *ow)
            }
            Geo::Pad {
                top,
                bottom,
                left,
                right,
                ..
            } => aff(
                Affine2::translate(*left as f64, *top as f64),
                h + top + bottom,
                w + left + right,
            ),
            Geo::Affine { m, oh, ow, .. } => aff(*m, *oh, *ow),
            Geo::Perspective { hm, oh, ow, .. } => (PointMap::Proj(*hm), *oh, *ow),
            Geo::Field { field, .. } => (PointMap::Field(field), h, w),
        }
    }

    fn bbox_method(&self) -> BboxMethod {
        match self {
            Geo::Affine { method, .. } => *method,
            _ => BboxMethod::LargestBox,
        }
    }
}

fn geo_buf(b: &Buf, g: &Geo, pm: &PointMap<'_>, oh: usize, ow: usize, is_mask: bool) -> Result<Buf> {
    let pick = |interp: &Interp, mask_interp: &Interp| if is_mask { *mask_interp } else { *interp };
    let pick_fill = |fill: &Vec<f64>, fill_mask: &Vec<f64>| if is_mask { fill_mask.clone() } else { fill.clone() };
    Ok(match g {
        Geo::HFlip => map_buf!(b, a => hflip(a)),
        Geo::VFlip => map_buf!(b, a => vflip(a)),
        Geo::Transpose => map_buf!(b, a => transpose(a)),
        Geo::Rot90(k) => map_buf!(b, a => rot90(a, *k)),
        Geo::CropResize {
            rect,
            oh,
            ow,
            interp,
            mask_interp,
        } => {
            if rect.w() == *ow && rect.h() == *oh {
                map_buf!(b, a => crop(a, rect.x0, rect.y0, rect.x1, rect.y1))
            } else {
                let interp = pick(interp, mask_interp);
                match b {
                    Buf::U8(a) => Buf::U8(resize_u8(a, *rect, *oh, *ow, interp)),
                    other => map_buf!(other, a => resize(a, *rect, *oh, *ow, interp)),
                }
            }
        }
        Geo::Pad {
            top,
            bottom,
            left,
            right,
            mode,
            fill,
            fill_mask,
        } => {
            let f = pick_fill(fill, fill_mask);
            map_buf!(b, a => pad(a, *top, *bottom, *left, *right, *mode, &f))
        }
        Geo::Affine {
            oh,
            ow,
            interp,
            mask_interp,
            mode,
            fill,
            fill_mask,
            ..
        } => {
            let PointMap::Affine(m) = pm else { unreachable!() };
            let Some(inv) = m.inverse() else {
                return input("affine transform is singular (zero scale?)");
            };
            let interp = pick(interp, mask_interp);
            let f = pick_fill(fill, fill_mask);
            match b {
                Buf::U8(a) => Buf::U8(warp_affine_u8(a, &inv, *oh, *ow, interp, *mode, &f)),
                other => map_buf!(other, a => warp_affine(a, &inv, *oh, *ow, interp, *mode, &f)),
            }
        }
        Geo::Perspective {
            hm,
            interp,
            mask_interp,
            mode,
            fill,
            fill_mask,
            ..
        } => {
            let Some(inv) = hm.inverse() else {
                return input("perspective transform is singular");
            };
            let interp = pick(interp, mask_interp);
            let f = pick_fill(fill, fill_mask);
            let m = inv.m;
            let coords = |v: usize, xs: &mut [f64], ys: &mut [f64]| {
                let y = v as f64 + 0.5;
                for (u, (xo, yo)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
                    let x = u as f64 + 0.5;
                    let wq = m[6] * x + m[7] * y + m[8];
                    if wq <= 1e-12 {
                        *xo = f64::NAN;
                        *yo = f64::NAN;
                    } else {
                        *xo = (m[0] * x + m[1] * y + m[2]) / wq - 0.5;
                        *yo = (m[3] * x + m[4] * y + m[5]) / wq - 0.5;
                    }
                }
            };
            map_buf!(b, a => remap(a, oh, ow, interp, *mode, &f, coords))
        }
        Geo::Field {
            field,
            interp,
            mask_interp,
            mode,
            fill,
            fill_mask,
        } => {
            let interp = pick(interp, mask_interp);
            let f = pick_fill(fill, fill_mask);
            let fw = field.w;
            let coords = |v: usize, xs: &mut [f64], ys: &mut [f64]| {
                let row = v * fw;
                for u in 0..xs.len() {
                    xs[u] = u as f64 + field.dx[row + u] as f64;
                    ys[u] = v as f64 + field.dy[row + u] as f64;
                }
            };
            map_buf!(b, a => remap(a, oh, ow, interp, *mode, &f, coords))
        }
    })
}

/// Apply one geometric op to every target of the sample.
pub(crate) fn apply_geo(s: &mut Sample, g: &Geo, remove_invisible_kps: bool) -> Result<()> {
    let (h, w, _) = s.image.dims();
    let (pm, oh, ow) = g.map(h, w);
    if oh == 0 || ow == 0 {
        return input(format!("transform would produce an empty {oh}x{ow} image"));
    }
    s.image = geo_buf(&s.image, g, &pm, oh, ow, false)?;
    for img in s.extra_images.iter_mut() {
        *img = geo_buf(img, g, &pm, oh, ow, false)?;
    }
    for mask in s.masks.iter_mut() {
        *mask = geo_buf(mask, g, &pm, oh, ow, true)?;
    }
    match pm {
        PointMap::Affine(m) => {
            if !s.bboxes.is_empty() {
                transform_boxes(&mut s.bboxes, &m, g.bbox_method(), oh, ow);
            }
            if !s.keypoints.is_empty() {
                transform_keypoints(&mut s.keypoints, &m, oh, ow, remove_invisible_kps);
            }
        }
        PointMap::Proj(hm) => {
            // boxes: bounding box of the 4 projected corners
            transform_boxes_with(&mut s.bboxes, oh, ow, |b| {
                let mut r = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for (x, y) in [(b.x0, b.y0), (b.x1, b.y0), (b.x0, b.y1), (b.x1, b.y1)] {
                    let (px, py) = hm.apply(x, y)?;
                    r = (r.0.min(px), r.1.min(py), r.2.max(px), r.3.max(py));
                }
                Some(r)
            });
            transform_keypoints_with(&mut s.keypoints, oh, ow, remove_invisible_kps, |x, y| {
                let p = hm.apply(x, y)?;
                Some((p, hm.jacobian(x, y)))
            });
        }
        PointMap::Field(field) => {
            let dmax = if s.bboxes.is_empty() && s.keypoints.is_empty() {
                0.0
            } else {
                field.max_disp()
            };
            transform_boxes_with(&mut s.bboxes, oh, ow, |b| Some(field.map_box(b, dmax)));
            transform_keypoints_with(&mut s.keypoints, oh, ow, remove_invisible_kps, |x, y| {
                let (px, py) = field.forward(x, y, dmax);
                let e = 0.25;
                let (ax, ay) = field.forward(x + e, y, dmax);
                let (bx, by) = field.forward(x, y + e, dmax);
                Some(((px, py), [(ax - px) / e, (bx - px) / e, (ay - py) / e, (by - py) / e]))
            });
        }
    }
    Ok(())
}
