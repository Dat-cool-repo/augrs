//! Bounding boxes and keypoints: formats, conversion, and transformation by an
//! affine map in continuous pixel coordinates.

use crate::error::{Result, input, param};
use crate::geometry::Affine2;
use serde::{Deserialize, Serialize};

/// Bounding-box formats (same names as Albumentations).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BboxFormat {
    /// `[x_min, y_min, x_max, y_max]` in pixels.
    PascalVoc,
    /// `[x_min, y_min, width, height]` in pixels.
    Coco,
    /// `[x_center, y_center, width, height]` normalised to `[0, 1]`.
    Yolo,
    /// `[x_min, y_min, x_max, y_max]` normalised to `[0, 1]`.
    Albumentations,
}

impl BboxFormat {
    /// Convert a box in this format to absolute `[x0, y0, x1, y1]` pixels.
    pub fn to_abs(self, b: [f64; 4], h: usize, w: usize) -> [f64; 4] {
        let (w, h) = (w as f64, h as f64);
        match self {
            BboxFormat::PascalVoc => b,
            BboxFormat::Coco => [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            BboxFormat::Yolo => {
                let (cx, cy, bw, bh) = (b[0] * w, b[1] * h, b[2] * w, b[3] * h);
                [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]
            }
            BboxFormat::Albumentations => [b[0] * w, b[1] * h, b[2] * w, b[3] * h],
        }
    }

    /// Convert absolute `[x0, y0, x1, y1]` pixels to this format.
    pub fn from_abs(self, b: [f64; 4], h: usize, w: usize) -> [f64; 4] {
        let (w, h) = (w as f64, h as f64);
        match self {
            BboxFormat::PascalVoc => b,
            BboxFormat::Coco => [b[0], b[1], b[2] - b[0], b[3] - b[1]],
            BboxFormat::Yolo => [
                (b[0] + b[2]) / 2.0 / w,
                (b[1] + b[3]) / 2.0 / h,
                (b[2] - b[0]) / w,
                (b[3] - b[1]) / h,
            ],
            BboxFormat::Albumentations => [b[0] / w, b[1] / h, b[2] / w, b[3] / h],
        }
    }
}

/// How an axis-aligned box is mapped through a rotation/shear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BboxMethod {
    /// Bounding box of the 4 transformed corners (always contains the object).
    #[default]
    LargestBox,
    /// Bounding box of the transformed inscribed ellipse (tighter, usually closer to the object).
    Ellipse,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BboxParams {
    pub format: BboxFormat,
    /// Drop boxes whose final area (in output pixels) is below this.
    #[serde(default)]
    pub min_area: f64,
    /// Drop boxes whose visible fraction (after all clipping) is below this.
    #[serde(default)]
    pub min_visibility: f64,
    #[serde(default)]
    pub min_width: f64,
    #[serde(default)]
    pub min_height: f64,
    /// Clip input boxes that extend outside the image (otherwise they are an error).
    #[serde(default = "default_true")]
    pub clip: bool,
    /// Drop boxes whose aspect ratio `max(w/h, h/w)` exceeds this.
    #[serde(default)]
    pub max_accept_ratio: Option<f64>,
}

fn default_true() -> bool {
    true
}

impl BboxParams {
    pub fn new(format: BboxFormat) -> Self {
        BboxParams {
            format,
            min_area: 0.0,
            min_visibility: 0.0,
            min_width: 0.0,
            min_height: 0.0,
            clip: true,
            max_accept_ratio: None,
        }
    }
}

/// A box in absolute continuous pixel coordinates, plus bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    /// Fraction of the original object that is still inside the image.
    pub visibility: f64,
    /// Index of the box in the caller's input list.
    pub id: usize,
}

impl BBox {
    pub fn area(&self) -> f64 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
}

/// Keypoint formats (same names as Albumentations).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeypointFormat {
    Xy,
    Yx,
    Xya,
    Xys,
    Xyas,
    Xysa,
}

impl KeypointFormat {
    pub fn ncols(self) -> usize {
        match self {
            KeypointFormat::Xy | KeypointFormat::Yx => 2,
            KeypointFormat::Xya | KeypointFormat::Xys => 3,
            KeypointFormat::Xyas | KeypointFormat::Xysa => 4,
        }
    }

    /// Parse a row into `(x, y, angle, scale)` (angle in the user's unit).
    pub fn parse(self, r: [f64; 4]) -> (f64, f64, f64, f64) {
        match self {
            KeypointFormat::Xy => (r[0], r[1], 0.0, 1.0),
            KeypointFormat::Yx => (r[1], r[0], 0.0, 1.0),
            KeypointFormat::Xya => (r[0], r[1], r[2], 1.0),
            KeypointFormat::Xys => (r[0], r[1], 0.0, r[2]),
            KeypointFormat::Xyas => (r[0], r[1], r[2], r[3]),
            KeypointFormat::Xysa => (r[0], r[1], r[3], r[2]),
        }
    }

    pub fn emit(self, x: f64, y: f64, a: f64, s: f64) -> [f64; 4] {
        match self {
            KeypointFormat::Xy => [x, y, 0.0, 0.0],
            KeypointFormat::Yx => [y, x, 0.0, 0.0],
            KeypointFormat::Xya => [x, y, a, 0.0],
            KeypointFormat::Xys => [x, y, s, 0.0],
            KeypointFormat::Xyas => [x, y, a, s],
            KeypointFormat::Xysa => [x, y, s, a],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeypointParams {
    pub format: KeypointFormat,
    /// Drop keypoints that leave the image.
    #[serde(default = "default_true")]
    pub remove_invisible: bool,
    #[serde(default = "default_true")]
    pub angle_in_degrees: bool,
    /// If true, input/output `x, y` use the *pixel-index* convention (pixel
    /// `i` is at coordinate `i`), instead of the continuous convention where
    /// pixel `i` is centred at `i + 0.5`. Conversion adds/subtracts 0.5.
    #[serde(default)]
    pub pixel_index_coords: bool,
}

impl KeypointParams {
    pub fn new(format: KeypointFormat) -> Self {
        KeypointParams {
            format,
            remove_invisible: true,
            angle_in_degrees: true,
            pixel_index_coords: false,
        }
    }
}

/// A keypoint in continuous pixel coordinates; `angle` in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keypoint {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub scale: f64,
    pub id: usize,
}

/// Convert user boxes to internal absolute boxes. Validates them.
pub(crate) fn boxes_in(raw: &[[f64; 4]], p: &BboxParams, h: usize, w: usize) -> Result<Vec<BBox>> {
    let mut out = Vec::with_capacity(raw.len());
    let (wf, hf) = (w as f64, h as f64);
    for (id, r) in raw.iter().enumerate() {
        if r.iter().any(|v| !v.is_finite()) {
            return input(format!("bbox {id} has non-finite values: {r:?}"));
        }
        let [x0, y0, x1, y1] = p.format.to_abs(*r, h, w);
        if x1 < x0 || y1 < y0 {
            return input(format!("bbox {id} {r:?} ({:?}) has negative width or height", p.format));
        }
        let eps = 1e-6 * wf.max(hf).max(1.0);
        let outside = x0 < -eps || y0 < -eps || x1 > wf + eps || y1 > hf + eps;
        if outside && !p.clip {
            return input(format!(
                "bbox {id} {r:?} lies outside the {w}x{h} image (set clip=True to clip it)"
            ));
        }
        let b = BBox {
            x0: x0.clamp(0.0, wf),
            y0: y0.clamp(0.0, hf),
            x1: x1.clamp(0.0, wf),
            y1: y1.clamp(0.0, hf),
            visibility: 1.0,
            id,
        };
        out.push(b);
    }
    Ok(out)
}

pub(crate) fn boxes_out(boxes: &[BBox], p: &BboxParams, h: usize, w: usize) -> (Vec<[f64; 4]>, Vec<usize>) {
    let mut out = Vec::with_capacity(boxes.len());
    let mut ids = Vec::with_capacity(boxes.len());
    for b in boxes {
        let bw = b.x1 - b.x0;
        let bh = b.y1 - b.y0;
        if bw <= 0.0 || bh <= 0.0 {
            continue;
        }
        if b.area() < p.min_area || b.visibility < p.min_visibility || bw < p.min_width || bh < p.min_height {
            continue;
        }
        if let Some(r) = p.max_accept_ratio {
            if (bw / bh).max(bh / bw) > r {
                continue;
            }
        }
        out.push(p.format.from_abs([b.x0, b.y0, b.x1, b.y1], h, w));
        ids.push(b.id);
    }
    (out, ids)
}

pub(crate) fn keypoints_in(raw: &[[f64; 4]], p: &KeypointParams) -> Result<Vec<Keypoint>> {
    let off = if p.pixel_index_coords { 0.5 } else { 0.0 };
    raw.iter()
        .enumerate()
        .map(|(id, r)| {
            let (x, y, a, s) = p.format.parse(*r);
            if !x.is_finite() || !y.is_finite() {
                return input(format!("keypoint {id} has non-finite coordinates"));
            }
            let a = if p.angle_in_degrees { a.to_radians() } else { a };
            Ok(Keypoint {
                x: x + off,
                y: y + off,
                angle: a,
                scale: s,
                id,
            })
        })
        .collect()
}

pub(crate) fn keypoints_out(kps: &[Keypoint], p: &KeypointParams) -> (Vec<[f64; 4]>, Vec<usize>) {
    let off = if p.pixel_index_coords { 0.5 } else { 0.0 };
    let mut out = Vec::with_capacity(kps.len());
    let mut ids = Vec::with_capacity(kps.len());
    for k in kps {
        let a = if p.angle_in_degrees {
            k.angle.to_degrees()
        } else {
            k.angle
        };
        out.push(p.format.emit(k.x - off, k.y - off, a, k.scale));
        ids.push(k.id);
    }
    (out, ids)
}

/// Map boxes through `m` into an `out_h x out_w` image, clip them, update the
/// visibility ratio and drop boxes that no longer overlap the image.
pub fn transform_boxes(boxes: &mut Vec<BBox>, m: &Affine2, method: BboxMethod, out_h: usize, out_w: usize) {
    let axis = m.is_axis_aligned();
    transform_boxes_with(boxes, out_h, out_w, |b| {
        Some(if axis {
            let (ax, ay) = m.apply(b.x0, b.y0);
            let (bx, by) = m.apply(b.x1, b.y1);
            (ax.min(bx), ay.min(by), ax.max(bx), ay.max(by))
        } else {
            match method {
                BboxMethod::LargestBox => {
                    let pts = [
                        m.apply(b.x0, b.y0),
                        m.apply(b.x1, b.y0),
                        m.apply(b.x0, b.y1),
                        m.apply(b.x1, b.y1),
                    ];
                    let mut r = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                    for (x, y) in pts {
                        r = (r.0.min(x), r.1.min(y), r.2.max(x), r.3.max(y));
                    }
                    r
                }
                BboxMethod::Ellipse => {
                    let (cx, cy) = m.apply((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0);
                    let rx = (b.x1 - b.x0) / 2.0;
                    let ry = (b.y1 - b.y0) / 2.0;
                    let hw = (m.a * rx).hypot(m.b * ry);
                    let hh = (m.d * rx).hypot(m.e * ry);
                    (cx - hw, cy - hh, cx + hw, cy + hh)
                }
            }
        })
    });
}

/// Like [`transform_boxes`] for any point map: `f` returns the unclipped new
/// extent `(x0, y0, x1, y1)` of a box, or `None` to drop it.
pub fn transform_boxes_with<F>(boxes: &mut Vec<BBox>, out_h: usize, out_w: usize, f: F)
where
    F: Fn(&BBox) -> Option<(f64, f64, f64, f64)>,
{
    let (wf, hf) = (out_w as f64, out_h as f64);
    boxes.retain_mut(|b| {
        let Some((nx0, ny0, nx1, ny1)) = f(b) else { return false };
        if !(nx0.is_finite() && ny0.is_finite() && nx1.is_finite() && ny1.is_finite()) {
            return false;
        }
        let full = (nx1 - nx0).max(0.0) * (ny1 - ny0).max(0.0);
        let cx0 = nx0.clamp(0.0, wf);
        let cy0 = ny0.clamp(0.0, hf);
        let cx1 = nx1.clamp(0.0, wf);
        let cy1 = ny1.clamp(0.0, hf);
        let clipped = (cx1 - cx0).max(0.0) * (cy1 - cy0).max(0.0);
        if full <= 0.0 || clipped <= 0.0 {
            return false;
        }
        b.visibility *= clipped / full;
        b.x0 = cx0;
        b.y0 = cy0;
        b.x1 = cx1;
        b.y1 = cy1;
        true
    });
}

/// Map keypoints with any point map: `f(x, y)` returns the new position and the
/// local Jacobian `[dx'/dx, dx'/dy, dy'/dx, dy'/dy]` (for angle and scale), or
/// `None` to drop the keypoint.
pub fn transform_keypoints_with<F>(kps: &mut Vec<Keypoint>, out_h: usize, out_w: usize, remove_invisible: bool, f: F)
where
    F: Fn(f64, f64) -> Option<((f64, f64), [f64; 4])>,
{
    let (wf, hf) = (out_w as f64, out_h as f64);
    kps.retain_mut(|k| {
        let Some(((x, y), j)) = f(k.x, k.y) else { return false };
        let (c, s) = (k.angle.cos(), k.angle.sin());
        let (dx, dy) = (j[0] * c + j[1] * s, j[2] * c + j[3] * s);
        k.x = x;
        k.y = y;
        k.angle = dy.atan2(dx).rem_euclid(std::f64::consts::TAU);
        k.scale *= j[0].hypot(j[2]).max(j[1].hypot(j[3]));
        !remove_invisible || (x >= 0.0 && x < wf && y >= 0.0 && y < hf)
    });
}

/// Map keypoints through `m`; optionally drop those that leave the output image.
pub fn transform_keypoints(kps: &mut Vec<Keypoint>, m: &Affine2, out_h: usize, out_w: usize, remove_invisible: bool) {
    let (wf, hf) = (out_w as f64, out_h as f64);
    let sf = m.scale_factor();
    kps.retain_mut(|k| {
        let (x, y) = m.apply(k.x, k.y);
        let (dx, dy) = m.apply_vec(k.angle.cos(), k.angle.sin());
        k.x = x;
        k.y = y;
        k.angle = dy.atan2(dx).rem_euclid(std::f64::consts::TAU);
        k.scale *= sf;
        !remove_invisible || (x >= 0.0 && x < wf && y >= 0.0 && y < hf)
    });
}

pub(crate) fn validate_bbox_params(p: &BboxParams) -> Result<()> {
    if !(0.0..=1.0).contains(&p.min_visibility) {
        return param("BboxParams.min_visibility must be in [0, 1]");
    }
    if p.min_area < 0.0 || p.min_width < 0.0 || p.min_height < 0.0 {
        return param("BboxParams.min_area/min_width/min_height must be >= 0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_roundtrip() {
        let (h, w) = (480, 640);
        let abs = [10.0, 20.0, 110.0, 220.0];
        for f in [
            BboxFormat::PascalVoc,
            BboxFormat::Coco,
            BboxFormat::Yolo,
            BboxFormat::Albumentations,
        ] {
            let r = f.from_abs(abs, h, w);
            let back = f.to_abs(r, h, w);
            for i in 0..4 {
                assert!((back[i] - abs[i]).abs() < 1e-9, "{f:?}");
            }
        }
        assert_eq!(BboxFormat::Coco.from_abs(abs, h, w), [10.0, 20.0, 100.0, 200.0]);
    }

    #[test]
    fn hflip_boxes() {
        let mut b = vec![BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 5.0,
            y1: 5.0,
            visibility: 1.0,
            id: 0,
        }];
        let m = Affine2::new(-1.0, 0.0, 20.0, 0.0, 1.0, 0.0);
        transform_boxes(&mut b, &m, BboxMethod::LargestBox, 10, 20);
        assert_eq!((b[0].x0, b[0].x1), (15.0, 20.0));
    }

    #[test]
    fn crop_visibility() {
        let mut b = vec![BBox {
            x0: 0.0,
            y0: 0.0,
            x1: 10.0,
            y1: 10.0,
            visibility: 1.0,
            id: 3,
        }];
        // crop starting at x=5: half the box remains
        transform_boxes(&mut b, &Affine2::translate(-5.0, 0.0), BboxMethod::LargestBox, 10, 10);
        assert!((b[0].visibility - 0.5).abs() < 1e-12);
        assert_eq!(b[0].id, 3);
        // crop fully outside: dropped
        transform_boxes(&mut b, &Affine2::translate(-50.0, 0.0), BboxMethod::LargestBox, 10, 10);
        assert!(b.is_empty());
    }
}
