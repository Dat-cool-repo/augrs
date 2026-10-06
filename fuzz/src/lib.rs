//! Structured input generation for the augrs fuzz targets.
//!
//! The fuzzer's bytes drive an `arbitrary::Unstructured` that builds a random pipeline
//! (any transform, nested `Compose`/`OneOf`/`SomeOf`/`Sequential`), random target
//! parameters and a random sample (image, masks, boxes, keypoints). Every value is
//! either "sane" (what a real config would contain) or, at a configurable rate, "wild":
//! zero / negative / huge sizes, NaN and infinities, inverted ranges, `p` outside
//! `[0, 1]`, empty fills, degenerate images (0x0, 1x1, 1xN) and degenerate boxes.
//!
//! The harness contract: `Pipeline::new` and `Pipeline::apply*` may return `Err`, but
//! must never panic, hang, or allocate absurd amounts of memory; and a successful
//! output must satisfy the invariants checked by [`check_output`].

use arbitrary::{Result, Unstructured};
use augrs_core::{
    BboxFormat, BboxMethod, BboxParams, BorderMode, Buf, DropoutBboxes, DropoutFill, DropoutKeypoints, Input, Interp,
    KeypointFormat, KeypointParams, NoiseDistribution, NormMode, Output, PadPosition, PipelineSpec, ToGrayMethod,
    Transform,
};
use ndarray::Array3;

/// Knobs for the generator.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    /// Out of 256: how often a value is drawn from the "wild" set.
    pub wild: u8,
    /// Maximum nesting depth of random compositions.
    pub max_depth: usize,
    /// Maximum number of elements (h * w * c) of a generated image.
    pub max_elems: usize,
}

pub struct G<'a, 'b> {
    pub u: &'b mut Unstructured<'a>,
    pub cfg: Cfg,
}

const WILD_F: [f64; 16] = [
    0.0,
    -0.0,
    -1.0,
    1.0,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1e9,
    -1e9,
    1e300,
    -1e300,
    f64::MIN_POSITIVE,
    1e-12,
    f64::MAX,
    0.999_999_999,
    1.000_000_001,
];

const WILD_USIZE: [usize; 10] = [
    0,
    1,
    2,
    65_535,
    100_000,
    1 << 20,
    1 << 31,
    1 << 40,
    usize::MAX / 2,
    usize::MAX,
];

impl<'a, 'b> G<'a, 'b> {
    pub fn new(u: &'b mut Unstructured<'a>, cfg: Cfg) -> Self {
        G { u, cfg }
    }

    fn wild(&mut self) -> Result<bool> {
        Ok(self.u.arbitrary::<u8>()? < self.cfg.wild)
    }

    fn bool(&mut self) -> Result<bool> {
        self.u.arbitrary()
    }

    fn pick<T: Copy>(&mut self, v: &[T]) -> Result<T> {
        Ok(*self.u.choose(v)?)
    }

    /// Any float: a wild one, a raw bit pattern, or a value in `[lo, hi]`.
    fn f(&mut self, lo: f64, hi: f64) -> Result<f64> {
        if self.wild()? {
            if self.bool()? {
                return self.pick(&WILD_F);
            }
            return Ok(f64::from_bits(self.u.arbitrary()?));
        }
        let t = self.u.int_in_range(0u32..=1000)? as f64 / 1000.0;
        Ok(lo + (hi - lo) * t)
    }

    /// A `(lo, hi)` range inside `[lo, hi]` (sorted unless wild).
    fn range(&mut self, lo: f64, hi: f64) -> Result<(f64, f64)> {
        let a = self.f(lo, hi)?;
        let b = self.f(lo, hi)?;
        if self.wild()? { Ok((a, b)) } else { Ok((a.min(b), a.max(b))) }
    }

    /// A range symmetric around `c`.
    fn sym(&mut self, c: f64, max: f64) -> Result<(f64, f64)> {
        if self.wild()? {
            return self.range(c - max, c + max);
        }
        let r = self.f(0.0, max)?;
        Ok((c - r, c + r))
    }

    fn p(&mut self) -> Result<f64> {
        if self.wild()? {
            return self.pick(&[-0.1, 1.5, f64::NAN, f64::INFINITY, -0.0, 2.0]);
        }
        Ok(match self.u.int_in_range(0u8..=5)? {
            0 => 0.0,
            1 | 2 => 1.0,
            3 => 0.5,
            _ => self.u.int_in_range(0u32..=100)? as f64 / 100.0,
        })
    }

    fn size(&mut self, lo: usize, hi: usize) -> Result<usize> {
        if self.wild()? {
            return self.pick(&WILD_USIZE);
        }
        self.u.int_in_range(lo..=hi)
    }

    fn fill(&mut self) -> Result<Vec<f64>> {
        if self.wild()? {
            let n = self.u.int_in_range(0usize..=5)?;
            return (0..n).map(|_| self.f(-1e9, 1e9)).collect();
        }
        let n = self.u.int_in_range(1usize..=4)?;
        (0..n).map(|_| Ok(self.u.int_in_range(0u32..=255)? as f64)).collect()
    }

    fn interp(&mut self) -> Result<Interp> {
        self.pick(&[Interp::Nearest, Interp::Linear, Interp::Cubic, Interp::Area, Interp::Lanczos])
    }

    fn border(&mut self) -> Result<BorderMode> {
        self.pick(&[
            BorderMode::Constant,
            BorderMode::Replicate,
            BorderMode::Reflect,
            BorderMode::Wrap,
            BorderMode::Reflect101,
        ])
    }

    fn pad_pos(&mut self) -> Result<PadPosition> {
        self.pick(&[
            PadPosition::Center,
            PadPosition::TopLeft,
            PadPosition::TopRight,
            PadPosition::BottomLeft,
            PadPosition::BottomRight,
            PadPosition::Random,
        ])
    }

    fn method(&mut self) -> Result<BboxMethod> {
        self.pick(&[BboxMethod::LargestBox, BboxMethod::Ellipse])
    }

    fn children(&mut self, depth: usize) -> Result<Vec<Transform>> {
        let n = self.u.int_in_range(0usize..=4)?;
        (0..n).map(|_| self.transform(depth + 1)).collect()
    }

    /// A random transform; compositions recurse up to `cfg.max_depth`.
    pub fn transform(&mut self, depth: usize) -> Result<Transform> {
        let leaf_only = depth >= self.cfg.max_depth;
        let k = if leaf_only {
            self.u.int_in_range(4u8..=29)?
        } else {
            self.u.int_in_range(0u8..=30)?
        };
        Ok(match k {
            0 => Transform::Compose {
                transforms: self.children(depth)?,
                p: self.p()?,
            },
            1 => Transform::OneOf {
                transforms: self.children(depth)?,
                p: self.p()?,
            },
            2 => {
                let transforms = self.children(depth)?;
                let n = if self.wild()? {
                    self.pick(&WILD_USIZE)?
                } else {
                    self.u.int_in_range(0..=transforms.len() + 1)?
                };
                Transform::SomeOf {
                    transforms,
                    n,
                    replace: self.bool()?,
                    p: self.p()?,
                }
            }
            3 => Transform::Sequential {
                transforms: self.children(depth)?,
                p: self.p()?,
            },
            4 => Transform::HorizontalFlip { p: self.p()? },
            5 => Transform::VerticalFlip { p: self.p()? },
            6 => Transform::Transpose { p: self.p()? },
            7 => Transform::RandomRotate90 { p: self.p()? },
            8 | 9 => {
                let (height, width) = (self.size(1, 40)?, self.size(1, 40)?);
                let (pad_if_needed, pad_position, border_mode) = (self.bool()?, self.pad_pos()?, self.border()?);
                let (fill, fill_mask, p) = (self.fill()?, self.fill()?, self.p()?);
                if k == 8 {
                    Transform::RandomCrop {
                        height,
                        width,
                        pad_if_needed,
                        pad_position,
                        border_mode,
                        fill,
                        fill_mask,
                        p,
                    }
                } else {
                    Transform::CenterCrop {
                        height,
                        width,
                        pad_if_needed,
                        pad_position,
                        border_mode,
                        fill,
                        fill_mask,
                        p,
                    }
                }
            }
            10 => Transform::RandomResizedCrop {
                height: self.size(1, 48)?,
                width: self.size(1, 48)?,
                scale: self.range(0.0, 1.5)?,
                ratio: self.range(0.0, 5.0)?,
                interpolation: self.interp()?,
                mask_interpolation: self.interp()?,
                p: self.p()?,
            },
            11 => Transform::Resize {
                height: self.size(1, 64)?,
                width: self.size(1, 64)?,
                interpolation: self.interp()?,
                mask_interpolation: self.interp()?,
                p: self.p()?,
            },
            12 | 13 => {
                let n = self.u.int_in_range(0usize..=3)?;
                let max_size = (0..n).map(|_| self.size(1, 64)).collect::<Result<Vec<_>>>()?;
                let (interpolation, mask_interpolation, p) = (self.interp()?, self.interp()?, self.p()?);
                if k == 12 {
                    Transform::LongestMaxSize {
                        max_size,
                        interpolation,
                        mask_interpolation,
                        p,
                    }
                } else {
                    Transform::SmallestMaxSize {
                        max_size,
                        interpolation,
                        mask_interpolation,
                        p,
                    }
                }
            }
            14 => {
                let div = |g: &mut Self| -> Result<Option<usize>> {
                    Ok(if g.bool()? { Some(g.size(1, 16)?) } else { None })
                };
                Transform::PadIfNeeded {
                    min_height: self.size(0, 64)?,
                    min_width: self.size(0, 64)?,
                    pad_height_divisor: div(self)?,
                    pad_width_divisor: div(self)?,
                    position: self.pad_pos()?,
                    border_mode: self.border()?,
                    fill: self.fill()?,
                    fill_mask: self.fill()?,
                    p: self.p()?,
                }
            }
            15 => Transform::Rotate {
                limit: self.sym(0.0, 180.0)?,
                interpolation: self.interp()?,
                mask_interpolation: self.interp()?,
                border_mode: self.border()?,
                fill: self.fill()?,
                fill_mask: self.fill()?,
                rotate_method: self.method()?,
                fit_output: self.bool()?,
                crop_border: self.bool()?,
                p: self.p()?,
            },
            16 => Transform::Affine {
                scale_x: self.range(0.2, 3.0)?,
                scale_y: self.range(0.2, 3.0)?,
                keep_ratio: self.bool()?,
                balanced_scale: self.bool()?,
                translate_x: self.sym(0.0, 0.5)?,
                translate_y: self.sym(0.0, 0.5)?,
                translate_px: self.bool()?,
                rotate: self.sym(0.0, 180.0)?,
                shear_x: self.sym(0.0, 60.0)?,
                shear_y: self.sym(0.0, 60.0)?,
                interpolation: self.interp()?,
                mask_interpolation: self.interp()?,
                border_mode: self.border()?,
                fill: self.fill()?,
                fill_mask: self.fill()?,
                rotate_method: self.method()?,
                fit_output: self.bool()?,
                p: self.p()?,
            },
            17 => Transform::ShiftScaleRotate {
                shift_limit_x: self.sym(0.0, 0.5)?,
                shift_limit_y: self.sym(0.0, 0.5)?,
                scale_limit: self.range(0.1, 2.0)?,
                rotate_limit: self.sym(0.0, 180.0)?,
                interpolation: self.interp()?,
                mask_interpolation: self.interp()?,
                border_mode: self.border()?,
                fill: self.fill()?,
                fill_mask: self.fill()?,
                rotate_method: self.method()?,
                p: self.p()?,
            },
            18 => Transform::Perspective {
                scale: self.range(0.0, 1.0)?,
                keep_size: self.bool()?,
                fit_output: self.bool()?,
                interpolation: self.interp()?,
                mask_interpolation: self.interp()?,
                border_mode: self.border()?,
                fill: self.fill()?,
                fill_mask: self.fill()?,
                p: self.p()?,
            },
            19 => Transform::ElasticTransform {
                alpha: self.f(0.0, 100.0)?,
                sigma: self.f(0.0, 60.0)?,
                interpolation: self.interp()?,
                mask_interpolation: self.interp()?,
                approximate: self.bool()?,
                same_dxdy: self.bool()?,
                noise_distribution: self.pick(&[NoiseDistribution::Gaussian, NoiseDistribution::Uniform])?,
                border_mode: self.border()?,
                fill: self.fill()?,
                fill_mask: self.fill()?,
                p: self.p()?,
            },
            20 => Transform::ColorJitter {
                brightness: self.range(0.0, 2.0)?,
                contrast: self.range(0.0, 2.0)?,
                saturation: self.range(0.0, 2.0)?,
                hue: self.sym(0.0, 0.5)?,
                p: self.p()?,
            },
            21 => Transform::RandomBrightnessContrast {
                brightness_limit: self.sym(0.0, 1.0)?,
                contrast_limit: self.sym(0.0, 1.0)?,
                brightness_by_max: self.bool()?,
                ensure_safe_range: self.bool()?,
                p: self.p()?,
            },
            22 => Transform::HueSaturationValue {
                hue_shift_limit: self.sym(0.0, 180.0)?,
                sat_shift_limit: self.sym(0.0, 255.0)?,
                val_shift_limit: self.sym(0.0, 255.0)?,
                p: self.p()?,
            },
            23 => Transform::RandomGamma {
                gamma_limit: self.range(1.0, 300.0)?,
                p: self.p()?,
            },
            24 => Transform::CLAHE {
                clip_limit: self.range(0.0, 10.0)?,
                tile_grid_size: (self.size(1, 16)?, self.size(1, 16)?),
                p: self.p()?,
            },
            25 => Transform::GaussNoise {
                std_range: self.range(0.0, 1.0)?,
                mean_range: self.sym(0.0, 0.5)?,
                per_channel: self.bool()?,
                noise_scale_factor: self.f(0.01, 1.0)?,
                p: self.p()?,
            },
            26 => Transform::ToGray {
                num_output_channels: self.size(1, 4)?,
                method: self.pick(&[
                    ToGrayMethod::WeightedAverage,
                    ToGrayMethod::Desaturation,
                    ToGrayMethod::Average,
                    ToGrayMethod::Max,
                ])?,
                p: self.p()?,
            },
            27 => {
                let fill = match self.u.int_in_range(0u8..=3)? {
                    0 => DropoutFill::Mode("random".into()),
                    1 => DropoutFill::Mode("random_uniform".into()),
                    2 if self.wild()? => DropoutFill::Mode("inpaint_telea".into()),
                    _ => DropoutFill::Value(self.fill()?),
                };
                let pixels = self.bool()?;
                let (hr, wr) = if pixels {
                    (self.range(0.0, 64.0)?, self.range(0.0, 64.0)?)
                } else {
                    (self.range(0.0, 1.0)?, self.range(0.0, 1.0)?)
                };
                let lo = self.size(0, 4)?;
                let hi = self.size(0, 8)?;
                Transform::CoarseDropout {
                    num_holes_range: if self.wild()? { (lo, hi) } else { (lo.min(hi), lo.max(hi)) },
                    hole_height_range: hr,
                    hole_width_range: wr,
                    fill,
                    fill_mask: if self.bool()? { Some(self.fill()?) } else { None },
                    bbox_handling: self.pick(&[DropoutBboxes::Shrink, DropoutBboxes::Visibility, DropoutBboxes::Keep])?,
                    keypoint_handling: self.pick(&[
                        DropoutKeypoints::Auto,
                        DropoutKeypoints::Remove,
                        DropoutKeypoints::Keep,
                    ])?,
                    p: self.p()?,
                }
            }
            28 => {
                let nm = self.u.int_in_range(0usize..=4)?;
                let mean = (0..nm).map(|_| self.f(0.0, 1.0)).collect::<Result<Vec<_>>>()?;
                let ns = self.u.int_in_range(0usize..=4)?;
                let std = (0..ns).map(|_| self.f(0.01, 1.0)).collect::<Result<Vec<_>>>()?;
                Transform::Normalize {
                    mean,
                    std,
                    max_pixel_value: if self.wild()? { self.f(-1.0, 1.0)? } else { 255.0 },
                    normalization: self.pick(&[
                        NormMode::Standard,
                        NormMode::Image,
                        NormMode::ImagePerChannel,
                        NormMode::MinMax,
                        NormMode::MinMaxPerChannel,
                    ])?,
                    p: self.p()?,
                }
            }
            29 => {
                let a = self.size(0, 15)?;
                let b = self.size(0, 15)?;
                Transform::GaussianBlur {
                    blur_limit: if self.wild()? { (a, b) } else { (a.min(b), a.max(b)) },
                    sigma_limit: self.range(0.0, 8.0)?,
                    p: self.p()?,
                }
            }
            // a deep chain of single-child compositions
            _ => {
                let levels = self.u.int_in_range(1usize..=80)?;
                let mut t = self.transform(self.cfg.max_depth)?;
                for i in 0..levels {
                    t = match i % 3 {
                        0 => Transform::OneOf {
                            transforms: vec![t],
                            p: 1.0,
                        },
                        1 => Transform::SomeOf {
                            transforms: vec![t],
                            n: 1,
                            replace: false,
                            p: 1.0,
                        },
                        _ => Transform::Compose {
                            transforms: vec![t],
                            p: 1.0,
                        },
                    };
                }
                t
            }
        })
    }

    pub fn bbox_params(&mut self) -> Result<BboxParams> {
        let format = self.pick(&[BboxFormat::PascalVoc, BboxFormat::Coco, BboxFormat::Yolo, BboxFormat::Albumentations])?;
        let mut p = BboxParams::new(format);
        p.clip = self.u.ratio(3u8, 4)?;
        if self.bool()? {
            p.min_area = self.f(0.0, 50.0)?;
            p.min_visibility = self.f(0.0, 1.0)?;
            p.min_width = self.f(0.0, 5.0)?;
            p.min_height = self.f(0.0, 5.0)?;
            p.max_accept_ratio = if self.bool()? { Some(self.f(1.0, 10.0)?) } else { None };
        }
        Ok(p)
    }

    pub fn keypoint_params(&mut self) -> Result<KeypointParams> {
        let format = self.pick(&[
            KeypointFormat::Xy,
            KeypointFormat::Yx,
            KeypointFormat::Xya,
            KeypointFormat::Xys,
            KeypointFormat::Xyas,
            KeypointFormat::Xysa,
        ])?;
        let mut p = KeypointParams::new(format);
        p.remove_invisible = self.bool()?;
        p.angle_in_degrees = self.bool()?;
        p.pixel_index_coords = self.bool()?;
        Ok(p)
    }

    pub fn spec(&mut self, n_max: usize) -> Result<PipelineSpec> {
        let n = self.u.int_in_range(0..=n_max)?;
        let transforms = (0..n).map(|_| self.transform(0)).collect::<Result<Vec<_>>>()?;
        let mut spec = PipelineSpec::new(transforms);
        spec.p = self.p()?;
        if self.u.ratio(3u8, 4)? {
            spec.bbox_params = Some(self.bbox_params()?);
        }
        if self.bool()? {
            spec.keypoint_params = Some(self.keypoint_params()?);
        }
        Ok(spec)
    }

    /// Image height and width: degenerate, small, medium, or long and thin.
    pub fn dims(&mut self) -> Result<(usize, usize)> {
        Ok(match self.u.int_in_range(0u8..=9)? {
            0 => (self.u.int_in_range(0..=2)?, self.u.int_in_range(0..=2)?),
            1 => (1, 1),
            2 => (1, self.u.int_in_range(1..=200_000)?),
            3 => (self.u.int_in_range(1..=200_000)?, self.u.int_in_range(1..=2)?),
            4 => (self.u.int_in_range(300..=1024)?, self.u.int_in_range(300..=1024)?),
            5 | 6 => (self.u.int_in_range(17..=128)?, self.u.int_in_range(17..=128)?),
            _ => (self.u.int_in_range(1..=16)?, self.u.int_in_range(1..=16)?),
        })
    }

    fn fill_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let pat_len = self.u.int_in_range(1usize..=64)?;
        let pat: Vec<u8> = (0..pat_len).map(|_| self.u.arbitrary()).collect::<Result<_>>()?;
        Ok((0..n).map(|i| pat[(i * 7 + i / pat_len) % pat_len]).collect())
    }

    /// A buffer of the given kind: 0 = u8, 1 = f32, 2 = u16, 3 = i32.
    pub fn buf(&mut self, h: usize, w: usize, c: usize, kind: u8) -> Result<Buf> {
        let n = h * w * c;
        let bytes = self.fill_bytes(n)?;
        Ok(match kind {
            0 => Buf::U8(Array3::from_shape_vec((h, w, c), bytes).unwrap()),
            1 => {
                let wild = self.wild()?;
                let mut v: Vec<f32> = bytes.iter().map(|&b| b as f32 / 255.0).collect();
                if wild && n > 0 {
                    for (i, s) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -5.0, 1e30].iter().enumerate() {
                        v[(i * 7919) % n] = *s;
                    }
                }
                Buf::F32(Array3::from_shape_vec((h, w, c), v).unwrap())
            }
            2 => Buf::U16(Array3::from_shape_vec((h, w, c), bytes.iter().map(|&b| b as u16 * 257).collect()).unwrap()),
            _ => Buf::I32(Array3::from_shape_vec((h, w, c), bytes.iter().map(|&b| b as i32 - 128).collect()).unwrap()),
        })
    }

    fn coord(&mut self, extent: f64) -> Result<f64> {
        if self.wild()? {
            return self.f(-extent, 2.0 * extent);
        }
        Ok(self.u.int_in_range(0u32..=1000)? as f64 / 1000.0 * extent)
    }

    /// Boxes in `format` for an `h x w` image (sane: inside the image; wild: anything).
    pub fn boxes(&mut self, format: BboxFormat, h: usize, w: usize) -> Result<Vec<[f64; 4]>> {
        let n = self.u.int_in_range(0usize..=8)?;
        let norm = matches!(format, BboxFormat::Yolo | BboxFormat::Albumentations);
        let (ex, ey) = if norm { (1.0, 1.0) } else { (w as f64, h as f64) };
        (0..n)
            .map(|_| {
                let (a, b) = (self.coord(ex)?, self.coord(ex)?);
                let (c, d) = (self.coord(ey)?, self.coord(ey)?);
                let (x0, x1) = (a.min(b), a.max(b));
                let (y0, y1) = (c.min(d), c.max(d));
                let r = match format {
                    BboxFormat::PascalVoc | BboxFormat::Albumentations => [x0, y0, x1, y1],
                    BboxFormat::Coco => [x0, y0, x1 - x0, y1 - y0],
                    BboxFormat::Yolo => {
                        let (bw, bh) = ((x1 - x0).min(2.0 * x0.min(1.0 - x0)), (y1 - y0).min(2.0 * y0.min(1.0 - y0)));
                        [x0, y0, bw.max(0.0), bh.max(0.0)]
                    }
                };
                if self.wild()? { Ok([r[2], r[3], r[0], r[1]]) } else { Ok(r) }
            })
            .collect()
    }

    pub fn keypoints(&mut self, h: usize, w: usize) -> Result<Vec<[f64; 4]>> {
        let n = self.u.int_in_range(0usize..=8)?;
        (0..n)
            .map(|_| {
                Ok([
                    self.coord(w as f64)?,
                    self.coord(h as f64)?,
                    self.f(-360.0, 360.0)?,
                    self.f(0.0, 4.0)?,
                ])
            })
            .collect()
    }

    /// A random sample for `spec`.
    pub fn input(&mut self, spec: &PipelineSpec) -> Result<Input> {
        let (h, w) = self.dims()?;
        let c = self.pick(&[3usize, 3, 3, 1, 1, 2, 4])?;
        let (h, w) = if h * w * c > self.cfg.max_elems {
            (h.min(self.cfg.max_elems / (w.max(1) * c)).max(1), w)
        } else {
            (h, w)
        };
        let (h, w) = if h * w * c > self.cfg.max_elems { (1, 1) } else { (h, w) };
        let kind = if self.u.ratio(1u8, 16)? { 2 } else { self.u.int_in_range(0u8..=1)? };
        let mut inp = Input::image(self.buf(h, w, c, kind)?);
        let nm = self.u.int_in_range(0usize..=2)?;
        for _ in 0..nm {
            let mc = self.pick(&[1usize, 1, 2, 3])?;
            let kind = self.u.int_in_range(0u8..=3)?;
            // occasionally a mask of the wrong size
            let (mh, mw) = if self.u.ratio(1u8, 32)? { (h + 1, w) } else { (h, w) };
            inp.masks.push(self.buf(mh, mw, mc, kind)?);
        }
        if self.u.ratio(1u8, 8)? {
            let kind = self.u.int_in_range(0u8..=1)?;
            inp.extra_images.push(self.buf(h, w, c, kind)?);
        }
        let format = spec.bbox_params.as_ref().map(|b| b.format).unwrap_or(BboxFormat::PascalVoc);
        if spec.bbox_params.is_some() || self.u.ratio(1u8, 16)? {
            inp.bboxes = self.boxes(format, h, w)?;
        }
        if spec.keypoint_params.is_some() || self.u.ratio(1u8, 16)? {
            inp.keypoints = self.keypoints(h, w)?;
        }
        Ok(inp)
    }
}

fn buf_eq(a: &Buf, b: &Buf) -> bool {
    match (a, b) {
        (Buf::U8(x), Buf::U8(y)) => x == y,
        (Buf::U16(x), Buf::U16(y)) => x == y,
        (Buf::I32(x), Buf::I32(y)) => x == y,
        (Buf::F32(x), Buf::F32(y)) => {
            x.shape() == y.shape() && x.iter().zip(y.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        _ => false,
    }
}

/// Two outputs of the same seed must be identical (bit for bit).
pub fn check_same(a: &Output, b: &Output) {
    assert!(buf_eq(&a.image, &b.image), "image differs between identical runs");
    assert_eq!(a.masks.len(), b.masks.len());
    for (x, y) in a.masks.iter().zip(&b.masks) {
        assert!(buf_eq(x, y), "mask differs between identical runs");
    }
    assert_eq!(a.bbox_ids, b.bbox_ids);
    assert_eq!(a.keypoint_ids, b.keypoint_ids);
    let bits = |v: &[[f64; 4]]| v.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.bboxes), bits(&b.bboxes));
    assert_eq!(bits(&a.keypoints), bits(&b.keypoints));
}

/// Invariants of a successful output.
pub fn check_output(spec: &PipelineSpec, inp_boxes: usize, inp_kps: usize, out: &Output) {
    let (h, w, c) = out.image.dims();
    assert!(h > 0 && w > 0 && c > 0, "empty output image {h}x{w}x{c}");
    for m in out.masks.iter().chain(out.extra_images.iter()) {
        let (mh, mw, _) = m.dims();
        assert_eq!((mh, mw), (h, w), "mask/extra image size differs from the image");
    }
    assert_eq!(out.bboxes.len(), out.bbox_ids.len());
    assert!(out.bbox_ids.windows(2).all(|p| p[0] < p[1]), "bbox ids not increasing: {:?}", out.bbox_ids);
    assert!(out.bbox_ids.iter().all(|&i| i < inp_boxes));
    if let Some(bp) = &spec.bbox_params {
        for b in &out.bboxes {
            let [x0, y0, x1, y1] = bp.format.to_abs(*b, h, w);
            assert!(b.iter().all(|v| v.is_finite()), "non-finite output box {b:?}");
            let e = 1e-6 * (w.max(h) as f64);
            assert!(
                x0 >= -e && y0 >= -e && x1 <= w as f64 + e && y1 <= h as f64 + e && x0 < x1 && y0 < y1,
                "output box {b:?} ({:?}) not a non-empty box inside the {w}x{h} image",
                bp.format
            );
            if bp.min_area.is_finite() {
                assert!((x1 - x0) * (y1 - y0) >= bp.min_area * (1.0 - 1e-9) - 1e-9, "box below min_area");
            }
        }
    }
    assert_eq!(out.keypoints.len(), out.keypoint_ids.len());
    assert!(out.keypoint_ids.windows(2).all(|p| p[0] < p[1]));
    assert!(out.keypoint_ids.iter().all(|&i| i < inp_kps));
    if let Some(kp) = &spec.keypoint_params {
        if kp.remove_invisible {
            let off = if kp.pixel_index_coords { 0.5 } else { 0.0 };
            for k in &out.keypoints {
                let (x, y) = match kp.format {
                    KeypointFormat::Yx => (k[1], k[0]),
                    _ => (k[0], k[1]),
                };
                let (x, y) = (x + off, y + off);
                assert!(
                    x >= 0.0 && y >= 0.0 && x < w as f64 + 1e-9 && y < h as f64 + 1e-9,
                    "visible keypoint ({x}, {y}) outside the {w}x{h} image"
                );
            }
        }
    }
}

fn show() -> bool {
    static SHOW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SHOW.get_or_init(|| std::env::var_os("AUGRS_FUZZ_SHOW").is_some())
}

/// Run one generated case: build, apply, check, and (sometimes) check determinism.
pub fn run_case(g: &mut G<'_, '_>, n_max: usize) -> Result<()> {
    let spec = g.spec(n_max)?;
    let inp = g.input(&spec)?;
    let seed: u64 = g.u.arbitrary()?;
    let twice = g.u.ratio(1u8, 8)?;
    if show() {
        // AUGRS_FUZZ_SHOW=1 <target> <artifact>: print the decoded case (for triage)
        eprintln!("spec = {}", serde_json::to_string(&spec).unwrap_or_default());
        eprintln!(
            "image {:?} {}, masks {:?}, extra {:?}, seed {seed}\nbboxes {:?}\nkeypoints {:?}",
            inp.image.dims(),
            inp.image.dtype_name(),
            inp.masks.iter().map(|m| (m.dims(), m.dtype_name())).collect::<Vec<_>>(),
            inp.extra_images.iter().map(|m| m.dims()).collect::<Vec<_>>(),
            inp.bboxes,
            inp.keypoints
        );
    }
    let Ok(pipe) = augrs_core::Pipeline::new(spec.clone(), Some(seed)) else {
        return Ok(());
    };
    // a valid spec must survive a JSON round trip
    let json = pipe.to_json();
    let back = augrs_core::Pipeline::from_json(&json, Some(seed)).expect("valid spec fails to reload from its JSON");
    assert_eq!(back.spec(), pipe.spec(), "JSON round trip changed the spec");
    let (nb, nk) = (inp.bboxes.len(), inp.keypoints.len());
    let second = if twice { Some(inp.clone()) } else { None };
    if let Ok(out) = pipe.apply_with_seed(inp, seed, false) {
        check_output(&spec, nb, nk, &out);
        if let Some(inp2) = second {
            let out2 = pipe.apply_with_seed(inp2, seed, true).expect("second identical run failed");
            check_same(&out, &out2);
        }
    }
    Ok(())
}
