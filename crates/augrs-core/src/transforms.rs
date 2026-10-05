//! The transform catalogue. A [`Transform`] is plain data (serde-serialisable,
//! tagged by `"type"`); randomness is sampled at apply time from the RNG.

use crate::border::BorderMode;
use crate::buffer::Buf;
use crate::error::{Result, input, param};
use crate::geo::{DispField, Geo, apply_geo};
use crate::geometry::{Affine2, Homography};
use crate::ops::clahe::clahe_u8;
use crate::ops::color::{self, GrayMethod};
use crate::ops::hsv::{HsvEdit, hsv_edit_u8, hue_lut};
use crate::ops::noise::{fill_normal, fill_uniform};
use crate::ops::resize::Rect;
use crate::ops::{gaussian_blur, gaussian_kernel_1d, normalize};
use crate::pipeline::Sample;
use crate::rng::Rng;
use crate::targets::BboxMethod;
use ndarray::Array3;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::json;

pub use crate::ops::color::GrayMethod as ToGrayMethod;
pub use crate::ops::resize::Interp;

fn half() -> f64 {
    0.5
}
fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}
fn d_rrc_scale() -> (f64, f64) {
    (0.08, 1.0)
}
fn d_rrc_ratio() -> (f64, f64) {
    (0.75, 4.0 / 3.0)
}
fn d_fill() -> Vec<f64> {
    vec![0.0]
}
fn d_rotate_limit() -> (f64, f64) {
    (-90.0, 90.0)
}
fn d_unit() -> (f64, f64) {
    (1.0, 1.0)
}
fn d_shift() -> (f64, f64) {
    (-0.0625, 0.0625)
}
fn d_ssr_scale() -> (f64, f64) {
    (0.9, 1.1)
}
fn d_ssr_rotate() -> (f64, f64) {
    (-45.0, 45.0)
}
fn d_jitter() -> (f64, f64) {
    (0.8, 1.2)
}
fn d_hue() -> (f64, f64) {
    (-0.5, 0.5)
}
fn d_mean() -> Vec<f64> {
    vec![0.485, 0.456, 0.406]
}
fn d_std() -> Vec<f64> {
    vec![0.229, 0.224, 0.225]
}
fn d_maxpix() -> f64 {
    255.0
}
fn d_sigma() -> (f64, f64) {
    (0.5, 3.0)
}
fn d_pm02() -> (f64, f64) {
    (-0.2, 0.2)
}
fn d_hue_shift() -> (f64, f64) {
    (-20.0, 20.0)
}
fn d_sat_shift() -> (f64, f64) {
    (-30.0, 30.0)
}
fn d_gamma() -> (f64, f64) {
    (80.0, 120.0)
}
fn d_clahe_clip() -> (f64, f64) {
    (1.0, 4.0)
}
fn d_tiles() -> (usize, usize) {
    (8, 8)
}
fn d_noise_std() -> (f64, f64) {
    (0.2, 0.44)
}
fn d_three() -> usize {
    3
}
fn d_holes() -> (usize, usize) {
    (1, 2)
}
fn d_hole_size() -> (f64, f64) {
    (0.1, 0.2)
}
fn d_persp_scale() -> (f64, f64) {
    (0.05, 0.1)
}
fn d_sigma50() -> f64 {
    50.0
}
fn d_one_usize() -> usize {
    1
}

/// Accept either a number or a list of numbers.
fn num_or_vec<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<f64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NV {
        N(f64),
        V(Vec<f64>),
    }
    Ok(match NV::deserialize(d)? {
        NV::N(x) => vec![x],
        NV::V(v) => v,
    })
}

fn usize_or_vec<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<usize>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NV {
        N(usize),
        V(Vec<usize>),
    }
    Ok(match NV::deserialize(d)? {
        NV::N(x) => vec![x],
        NV::V(v) => v,
    })
}

fn opt_num_or_vec<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<Vec<f64>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NV {
        N(f64),
        V(Vec<f64>),
    }
    Ok(Option::<NV>::deserialize(d)?.map(|v| match v {
        NV::N(x) => vec![x],
        NV::V(v) => v,
    }))
}

/// Where the image ends up inside the padded canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PadPosition {
    #[default]
    Center,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Random,
}

/// How `CoarseDropout` fills the holes of images.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DropoutFill {
    /// A constant (one value per channel, or one value for all).
    Value(#[serde(deserialize_with = "num_or_vec")] Vec<f64>),
    /// `"random"` (independent random value per pixel) or `"random_uniform"` (one random colour per hole).
    Mode(String),
}

impl Default for DropoutFill {
    fn default() -> Self {
        DropoutFill::Value(vec![0.0])
    }
}

/// How `CoarseDropout` treats bounding boxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DropoutBboxes {
    /// Lower the box visibility by the covered fraction and shrink the box to
    /// its visible extent (Albumentations behaviour). Fully covered boxes are dropped.
    #[default]
    Shrink,
    /// Only lower the visibility (the box keeps its coordinates).
    Visibility,
    /// Ignore holes.
    Keep,
}

/// How `CoarseDropout` treats keypoints that fall inside a hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DropoutKeypoints {
    /// Remove them when `KeypointParams.remove_invisible` is set (Albumentations behaviour).
    #[default]
    Auto,
    Remove,
    Keep,
}

/// Noise used to build elastic displacement fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NoiseDistribution {
    #[default]
    Gaussian,
    Uniform,
}

/// `Normalize` modes (Albumentations `normalization`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormMode {
    /// `(x - mean * max_pixel_value) / (std * max_pixel_value)`.
    #[default]
    Standard,
    /// `(x - mean(x)) / (std(x) + 1e-4)` over all values, clipped to `[-20, 20]`.
    Image,
    /// Same per channel.
    ImagePerChannel,
    /// `(x - min) / (max - min)` over all values.
    MinMax,
    /// Same per channel.
    MinMaxPerChannel,
}

/// All transforms. Defaults follow Albumentations 2.0 (the last MIT release).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Transform {
    /// Apply each child in order (each with its own probability).
    Compose {
        transforms: Vec<Transform>,
        #[serde(default = "one")]
        p: f64,
    },
    /// Apply exactly one child, chosen with probability proportional to its `p`.
    OneOf {
        transforms: Vec<Transform>,
        #[serde(default = "half")]
        p: f64,
    },
    /// With probability `p`, pick `n` children uniformly (with or without
    /// replacement), keep their original order, and run each with its own `p`.
    SomeOf {
        transforms: Vec<Transform>,
        #[serde(default = "d_one_usize")]
        n: usize,
        #[serde(default)]
        replace: bool,
        #[serde(default = "one")]
        p: f64,
    },
    /// With probability `p`, run every child in order (each with its own `p`).
    Sequential {
        transforms: Vec<Transform>,
        #[serde(default = "half")]
        p: f64,
    },
    HorizontalFlip {
        #[serde(default = "half")]
        p: f64,
    },
    VerticalFlip {
        #[serde(default = "half")]
        p: f64,
    },
    /// Swap rows and columns.
    Transpose {
        #[serde(default = "half")]
        p: f64,
    },
    /// Rotate by a random multiple of 90 degrees (`k` uniform in 0..=3).
    RandomRotate90 {
        #[serde(default = "one")]
        p: f64,
    },
    RandomCrop {
        height: usize,
        width: usize,
        /// Pad first when the image is smaller than the crop (otherwise it is an error).
        #[serde(default)]
        pad_if_needed: bool,
        #[serde(default)]
        pad_position: PadPosition,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default = "one")]
        p: f64,
    },
    CenterCrop {
        height: usize,
        width: usize,
        #[serde(default)]
        pad_if_needed: bool,
        #[serde(default)]
        pad_position: PadPosition,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default = "one")]
        p: f64,
    },
    RandomResizedCrop {
        height: usize,
        width: usize,
        #[serde(default = "d_rrc_scale")]
        scale: (f64, f64),
        #[serde(default = "d_rrc_ratio")]
        ratio: (f64, f64),
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default = "one")]
        p: f64,
    },
    Resize {
        height: usize,
        width: usize,
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default = "one")]
        p: f64,
    },
    /// Rescale so the longest side equals `max_size` (one of them, at random, if several).
    LongestMaxSize {
        #[serde(deserialize_with = "usize_or_vec")]
        max_size: Vec<usize>,
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default = "one")]
        p: f64,
    },
    SmallestMaxSize {
        #[serde(deserialize_with = "usize_or_vec")]
        max_size: Vec<usize>,
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default = "one")]
        p: f64,
    },
    /// Pad to at least `min_height x min_width` and/or to a multiple of the divisors.
    PadIfNeeded {
        #[serde(default)]
        min_height: usize,
        #[serde(default)]
        min_width: usize,
        #[serde(default)]
        pad_height_divisor: Option<usize>,
        #[serde(default)]
        pad_width_divisor: Option<usize>,
        #[serde(default)]
        position: PadPosition,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default = "one")]
        p: f64,
    },
    Rotate {
        #[serde(default = "d_rotate_limit")]
        limit: (f64, f64),
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default)]
        rotate_method: BboxMethod,
        /// Enlarge the output so the whole rotated image is visible.
        #[serde(default)]
        fit_output: bool,
        /// Crop the largest axis-aligned rectangle with no border pixels (output size changes).
        #[serde(default)]
        crop_border: bool,
        #[serde(default = "half")]
        p: f64,
    },
    /// Scale, translate, rotate and shear about the image centre.
    Affine {
        #[serde(default = "d_unit")]
        scale_x: (f64, f64),
        #[serde(default = "d_unit")]
        scale_y: (f64, f64),
        /// Use the x scale for y too.
        #[serde(default)]
        keep_ratio: bool,
        /// Sample zoom-in and zoom-out equally often (when the range straddles 1).
        #[serde(default)]
        balanced_scale: bool,
        /// Translation range as a fraction of width (or pixels if `translate_px`).
        #[serde(default)]
        translate_x: (f64, f64),
        #[serde(default)]
        translate_y: (f64, f64),
        #[serde(default)]
        translate_px: bool,
        #[serde(default)]
        rotate: (f64, f64),
        #[serde(default)]
        shear_x: (f64, f64),
        #[serde(default)]
        shear_y: (f64, f64),
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default)]
        rotate_method: BboxMethod,
        #[serde(default)]
        fit_output: bool,
        #[serde(default = "half")]
        p: f64,
    },
    /// `scale_limit` is the multiplicative range (Albumentations' `1 + scale_limit`).
    ShiftScaleRotate {
        #[serde(default = "d_shift")]
        shift_limit_x: (f64, f64),
        #[serde(default = "d_shift")]
        shift_limit_y: (f64, f64),
        #[serde(default = "d_ssr_scale")]
        scale_limit: (f64, f64),
        #[serde(default = "d_ssr_rotate")]
        rotate_limit: (f64, f64),
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default)]
        rotate_method: BboxMethod,
        #[serde(default = "half")]
        p: f64,
    },
    /// Random four-point perspective warp. Each corner moves inwards by
    /// `|N(0, scale)| mod 0.32` of the image size; the jittered quadrilateral
    /// is mapped onto the output rectangle. Boxes are the bounding boxes of
    /// their 4 projected corners.
    Perspective {
        #[serde(default = "d_persp_scale")]
        scale: (f64, f64),
        /// Resize the result back to the input size (otherwise the output size follows the quadrilateral).
        #[serde(default = "yes")]
        keep_size: bool,
        /// Keep the whole warped input image in view.
        #[serde(default)]
        fit_output: bool,
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default = "half")]
        p: f64,
    },
    /// Elastic deformation: smooth random displacement fields (noise normalised to
    /// `[-1, 1]`, Gaussian-blurred with `sigma`, scaled by `alpha`).
    ElasticTransform {
        #[serde(default = "one")]
        alpha: f64,
        #[serde(default = "d_sigma50")]
        sigma: f64,
        #[serde(default)]
        interpolation: Interp,
        #[serde(default = "Interp::nearest")]
        mask_interpolation: Interp,
        /// Use a fixed 17x17 smoothing kernel (faster, less smooth).
        #[serde(default)]
        approximate: bool,
        /// Use the same field for x and y.
        #[serde(default)]
        same_dxdy: bool,
        #[serde(default)]
        noise_distribution: NoiseDistribution,
        #[serde(default)]
        border_mode: BorderMode,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill: Vec<f64>,
        #[serde(default = "d_fill", deserialize_with = "num_or_vec")]
        fill_mask: Vec<f64>,
        #[serde(default = "half")]
        p: f64,
    },
    /// torchvision-style jitter applied in random order. Factor ranges for
    /// brightness/contrast/saturation; hue shift range in turns (`[-0.5, 0.5]`).
    ColorJitter {
        #[serde(default = "d_jitter")]
        brightness: (f64, f64),
        #[serde(default = "d_jitter")]
        contrast: (f64, f64),
        #[serde(default = "d_jitter")]
        saturation: (f64, f64),
        #[serde(default = "d_hue")]
        hue: (f64, f64),
        #[serde(default = "half")]
        p: f64,
    },
    /// `img * (1 + c) + b * max` (or `b * mean(img)` if `brightness_by_max` is false).
    RandomBrightnessContrast {
        #[serde(default = "d_pm02")]
        brightness_limit: (f64, f64),
        #[serde(default = "d_pm02")]
        contrast_limit: (f64, f64),
        #[serde(default = "yes")]
        brightness_by_max: bool,
        /// Limit alpha/beta so no value can saturate.
        #[serde(default)]
        ensure_safe_range: bool,
        #[serde(default = "half")]
        p: f64,
    },
    /// Shift hue (OpenCV units, 180 = full turn), saturation and value (0..255) in 8-bit HSV.
    HueSaturationValue {
        #[serde(default = "d_hue_shift")]
        hue_shift_limit: (f64, f64),
        #[serde(default = "d_sat_shift")]
        sat_shift_limit: (f64, f64),
        #[serde(default = "d_hue_shift")]
        val_shift_limit: (f64, f64),
        #[serde(default = "half")]
        p: f64,
    },
    /// `img ** (gamma / 100)` with gamma drawn from `gamma_limit`.
    RandomGamma {
        #[serde(default = "d_gamma")]
        gamma_limit: (f64, f64),
        #[serde(default = "half")]
        p: f64,
    },
    /// Contrast-limited adaptive histogram equalisation (on L of Lab for RGB).
    #[allow(clippy::upper_case_acronyms)]
    CLAHE {
        #[serde(default = "d_clahe_clip")]
        clip_limit: (f64, f64),
        #[serde(default = "d_tiles")]
        tile_grid_size: (usize, usize),
        #[serde(default = "half")]
        p: f64,
    },
    /// Additive Gaussian noise; std and mean are fractions of the max value (255 or 1).
    GaussNoise {
        #[serde(default = "d_noise_std")]
        std_range: (f64, f64),
        #[serde(default)]
        mean_range: (f64, f64),
        #[serde(default = "yes")]
        per_channel: bool,
        /// Generate noise at this fraction of the resolution and upsample (bilinear).
        #[serde(default = "one")]
        noise_scale_factor: f64,
        #[serde(default = "half")]
        p: f64,
    },
    ToGray {
        #[serde(default = "d_three")]
        num_output_channels: usize,
        #[serde(default)]
        method: GrayMethod,
        #[serde(default = "half")]
        p: f64,
    },
    /// Drop rectangular holes. Hole sizes are pixels if `hole_height_range.1 >= 1`,
    /// otherwise fractions of the image size (Albumentations rule).
    CoarseDropout {
        #[serde(default = "d_holes")]
        num_holes_range: (usize, usize),
        #[serde(default = "d_hole_size")]
        hole_height_range: (f64, f64),
        #[serde(default = "d_hole_size")]
        hole_width_range: (f64, f64),
        #[serde(default)]
        fill: DropoutFill,
        /// Fill value for masks; `None` leaves masks untouched.
        #[serde(default, deserialize_with = "opt_num_or_vec")]
        fill_mask: Option<Vec<f64>>,
        #[serde(default)]
        bbox_handling: DropoutBboxes,
        #[serde(default)]
        keypoint_handling: DropoutKeypoints,
        #[serde(default = "half")]
        p: f64,
    },
    /// Converts the image to float32.
    Normalize {
        #[serde(default = "d_mean")]
        mean: Vec<f64>,
        #[serde(default = "d_std")]
        std: Vec<f64>,
        #[serde(default = "d_maxpix")]
        max_pixel_value: f64,
        #[serde(default)]
        normalization: NormMode,
        #[serde(default = "one")]
        p: f64,
    },
    /// Kernel size range (odd, `(0, 0)` = derive from sigma) and sigma range.
    GaussianBlur {
        #[serde(default)]
        blur_limit: (usize, usize),
        #[serde(default = "d_sigma")]
        sigma_limit: (f64, f64),
        #[serde(default = "half")]
        p: f64,
    },
}

impl Interp {
    pub fn nearest() -> Interp {
        Interp::Nearest
    }
}

/// Per-call context that transforms need besides the RNG.
pub(crate) struct Ctx {
    pub remove_invisible_kps: bool,
}

fn check_range(name: &str, field: &str, r: (f64, f64)) -> Result<()> {
    if !(r.0.is_finite() && r.1.is_finite()) || r.0 > r.1 {
        return param(format!("{name}.{field}: invalid range {r:?} (need finite lo <= hi)"));
    }
    Ok(())
}

fn check_fill(name: &str, fill: &[f64]) -> Result<()> {
    if fill.is_empty() {
        return param(format!("{name}.fill must not be empty"));
    }
    Ok(())
}

fn check_children(ts: &[Transform]) -> Result<()> {
    ts.iter().try_for_each(|t| t.validate())
}

impl Transform {
    pub fn p(&self) -> f64 {
        use Transform::*;
        match self {
            Compose { p, .. }
            | OneOf { p, .. }
            | SomeOf { p, .. }
            | Sequential { p, .. }
            | HorizontalFlip { p }
            | VerticalFlip { p }
            | Transpose { p }
            | RandomRotate90 { p }
            | RandomCrop { p, .. }
            | CenterCrop { p, .. }
            | RandomResizedCrop { p, .. }
            | Resize { p, .. }
            | LongestMaxSize { p, .. }
            | SmallestMaxSize { p, .. }
            | PadIfNeeded { p, .. }
            | Rotate { p, .. }
            | Affine { p, .. }
            | ShiftScaleRotate { p, .. }
            | Perspective { p, .. }
            | ElasticTransform { p, .. }
            | ColorJitter { p, .. }
            | RandomBrightnessContrast { p, .. }
            | HueSaturationValue { p, .. }
            | RandomGamma { p, .. }
            | CLAHE { p, .. }
            | GaussNoise { p, .. }
            | ToGray { p, .. }
            | CoarseDropout { p, .. }
            | Normalize { p, .. }
            | GaussianBlur { p, .. } => *p,
        }
    }

    pub fn name(&self) -> &'static str {
        use Transform::*;
        match self {
            Compose { .. } => "Compose",
            OneOf { .. } => "OneOf",
            SomeOf { .. } => "SomeOf",
            Sequential { .. } => "Sequential",
            HorizontalFlip { .. } => "HorizontalFlip",
            VerticalFlip { .. } => "VerticalFlip",
            Transpose { .. } => "Transpose",
            RandomRotate90 { .. } => "RandomRotate90",
            RandomCrop { .. } => "RandomCrop",
            CenterCrop { .. } => "CenterCrop",
            RandomResizedCrop { .. } => "RandomResizedCrop",
            Resize { .. } => "Resize",
            LongestMaxSize { .. } => "LongestMaxSize",
            SmallestMaxSize { .. } => "SmallestMaxSize",
            PadIfNeeded { .. } => "PadIfNeeded",
            Rotate { .. } => "Rotate",
            Affine { .. } => "Affine",
            ShiftScaleRotate { .. } => "ShiftScaleRotate",
            Perspective { .. } => "Perspective",
            ElasticTransform { .. } => "ElasticTransform",
            ColorJitter { .. } => "ColorJitter",
            RandomBrightnessContrast { .. } => "RandomBrightnessContrast",
            HueSaturationValue { .. } => "HueSaturationValue",
            RandomGamma { .. } => "RandomGamma",
            CLAHE { .. } => "CLAHE",
            GaussNoise { .. } => "GaussNoise",
            ToGray { .. } => "ToGray",
            CoarseDropout { .. } => "CoarseDropout",
            Normalize { .. } => "Normalize",
            GaussianBlur { .. } => "GaussianBlur",
        }
    }

    /// Validate parameters (recursively).
    pub fn validate(&self) -> Result<()> {
        use Transform::*;
        let n = self.name();
        let p = self.p();
        if !(0.0..=1.0).contains(&p) {
            return param(format!("{n}.p must be in [0, 1], got {p}"));
        }
        match self {
            Compose { transforms, .. } | OneOf { transforms, .. } | Sequential { transforms, .. } => {
                check_children(transforms)?
            }
            SomeOf {
                transforms,
                n: k,
                replace,
                ..
            } => {
                check_children(transforms)?;
                if !*replace && *k > transforms.len() {
                    return param(format!(
                        "{n}: n = {k} > {} transforms (use replace=True)",
                        transforms.len()
                    ));
                }
            }
            HorizontalFlip { .. } | VerticalFlip { .. } | Transpose { .. } | RandomRotate90 { .. } => {}
            RandomCrop {
                height, width, fill, ..
            }
            | CenterCrop {
                height, width, fill, ..
            } => {
                if *height == 0 || *width == 0 {
                    return param(format!("{n}: height and width must be > 0"));
                }
                check_fill(n, fill)?;
            }
            Resize { height, width, .. } => {
                if *height == 0 || *width == 0 {
                    return param(format!("{n}: height and width must be > 0"));
                }
            }
            RandomResizedCrop {
                height,
                width,
                scale,
                ratio,
                ..
            } => {
                if *height == 0 || *width == 0 {
                    return param(format!("{n}: height and width must be > 0"));
                }
                check_range(n, "scale", *scale)?;
                check_range(n, "ratio", *ratio)?;
                if scale.0 <= 0.0 || ratio.0 <= 0.0 {
                    return param(format!("{n}: scale and ratio must be positive"));
                }
            }
            LongestMaxSize { max_size, .. } | SmallestMaxSize { max_size, .. } => {
                if max_size.is_empty() || max_size.contains(&0) {
                    return param(format!("{n}.max_size must be non-empty and > 0"));
                }
            }
            PadIfNeeded {
                fill,
                pad_height_divisor,
                pad_width_divisor,
                ..
            } => {
                check_fill(n, fill)?;
                if *pad_height_divisor == Some(0) || *pad_width_divisor == Some(0) {
                    return param(format!("{n}: divisors must be > 0"));
                }
            }
            Rotate {
                limit,
                fill,
                fit_output,
                crop_border,
                ..
            } => {
                check_range(n, "limit", *limit)?;
                check_fill(n, fill)?;
                if *fit_output && *crop_border {
                    return param(format!("{n}: fit_output and crop_border are mutually exclusive"));
                }
            }
            Affine {
                scale_x,
                scale_y,
                translate_x,
                translate_y,
                rotate,
                shear_x,
                shear_y,
                fill,
                ..
            } => {
                for (f, r) in [
                    ("scale_x", scale_x),
                    ("scale_y", scale_y),
                    ("translate_x", translate_x),
                    ("translate_y", translate_y),
                    ("rotate", rotate),
                    ("shear_x", shear_x),
                    ("shear_y", shear_y),
                ] {
                    check_range(n, f, *r)?;
                }
                if scale_x.0 <= 0.0 || scale_y.0 <= 0.0 {
                    return param(format!("{n}: scale must be > 0"));
                }
                if shear_x.0.abs().max(shear_x.1.abs()) >= 90.0 || shear_y.0.abs().max(shear_y.1.abs()) >= 90.0 {
                    return param(format!("{n}: |shear| must be < 90 degrees"));
                }
                check_fill(n, fill)?;
            }
            ShiftScaleRotate {
                shift_limit_x,
                shift_limit_y,
                scale_limit,
                rotate_limit,
                fill,
                ..
            } => {
                check_range(n, "shift_limit_x", *shift_limit_x)?;
                check_range(n, "shift_limit_y", *shift_limit_y)?;
                check_range(n, "scale_limit", *scale_limit)?;
                check_range(n, "rotate_limit", *rotate_limit)?;
                if scale_limit.0 <= 0.0 {
                    return param(format!("{n}: scale must stay > 0 (scale_limit lower bound > -1)"));
                }
                check_fill(n, fill)?;
            }
            Perspective { scale, fill, .. } => {
                check_range(n, "scale", *scale)?;
                if scale.0 < 0.0 {
                    return param(format!("{n}.scale must be >= 0"));
                }
                check_fill(n, fill)?;
            }
            ElasticTransform { alpha, sigma, fill, .. } => {
                if !alpha.is_finite() || !(sigma.is_finite() && *sigma > 0.0) {
                    return param(format!("{n}: alpha must be finite and sigma > 0"));
                }
                check_fill(n, fill)?;
            }
            ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
                ..
            } => {
                for (f, r) in [
                    ("brightness", brightness),
                    ("contrast", contrast),
                    ("saturation", saturation),
                ] {
                    check_range(n, f, *r)?;
                    if r.0 < 0.0 {
                        return param(format!("{n}.{f}: factors must be >= 0"));
                    }
                }
                check_range(n, "hue", *hue)?;
                if hue.0 < -0.5 || hue.1 > 0.5 {
                    return param(format!("{n}.hue must lie in [-0.5, 0.5]"));
                }
            }
            RandomBrightnessContrast {
                brightness_limit,
                contrast_limit,
                ..
            } => {
                check_range(n, "brightness_limit", *brightness_limit)?;
                check_range(n, "contrast_limit", *contrast_limit)?;
            }
            HueSaturationValue {
                hue_shift_limit,
                sat_shift_limit,
                val_shift_limit,
                ..
            } => {
                check_range(n, "hue_shift_limit", *hue_shift_limit)?;
                check_range(n, "sat_shift_limit", *sat_shift_limit)?;
                check_range(n, "val_shift_limit", *val_shift_limit)?;
            }
            RandomGamma { gamma_limit, .. } => {
                check_range(n, "gamma_limit", *gamma_limit)?;
                if gamma_limit.0 <= 0.0 {
                    return param(format!("{n}.gamma_limit must be > 0"));
                }
            }
            CLAHE {
                clip_limit,
                tile_grid_size,
                ..
            } => {
                check_range(n, "clip_limit", *clip_limit)?;
                if clip_limit.0 < 0.0 || tile_grid_size.0 == 0 || tile_grid_size.1 == 0 {
                    return param(format!("{n}: clip_limit must be >= 0 and the tile grid non-empty"));
                }
            }
            GaussNoise {
                std_range,
                mean_range,
                noise_scale_factor,
                ..
            } => {
                check_range(n, "std_range", *std_range)?;
                check_range(n, "mean_range", *mean_range)?;
                if std_range.0 < 0.0 || !(*noise_scale_factor > 0.0 && *noise_scale_factor <= 1.0) {
                    return param(format!("{n}: std must be >= 0 and noise_scale_factor in (0, 1]"));
                }
            }
            ToGray {
                num_output_channels, ..
            } => {
                if *num_output_channels == 0 {
                    return param(format!("{n}.num_output_channels must be > 0"));
                }
            }
            CoarseDropout {
                num_holes_range,
                hole_height_range,
                hole_width_range,
                fill,
                ..
            } => {
                if num_holes_range.0 > num_holes_range.1 {
                    return param(format!("{n}.num_holes_range: lo > hi"));
                }
                check_range(n, "hole_height_range", *hole_height_range)?;
                check_range(n, "hole_width_range", *hole_width_range)?;
                if hole_height_range.0 < 0.0 || hole_width_range.0 < 0.0 {
                    return param(format!("{n}: hole sizes must be >= 0"));
                }
                if let DropoutFill::Mode(m) = fill {
                    if m != "random" && m != "random_uniform" {
                        return param(format!(
                            "{n}.fill: unsupported mode {m:?} (use a value, \"random\" or \"random_uniform\")"
                        ));
                    }
                }
            }
            Normalize {
                mean,
                std,
                max_pixel_value,
                ..
            } => {
                if mean.is_empty() || std.is_empty() {
                    return param(format!("{n}: mean/std must not be empty"));
                }
                if std.contains(&0.0) || *max_pixel_value == 0.0 {
                    return param(format!("{n}: std and max_pixel_value must be non-zero"));
                }
            }
            GaussianBlur {
                blur_limit,
                sigma_limit,
                ..
            } => {
                if blur_limit.0 > blur_limit.1 {
                    return param(format!("{n}.blur_limit: lo > hi"));
                }
                check_range(n, "sigma_limit", *sigma_limit)?;
                if sigma_limit.0 < 0.0 || (blur_limit.1 == 0 && sigma_limit.1 <= 0.0) {
                    return param(format!("{n}: need sigma > 0 when blur_limit is (0, 0)"));
                }
            }
        }
        Ok(())
    }

    /// Apply with probability `p`.
    pub(crate) fn run(&self, s: &mut Sample, rng: &mut Rng, ctx: &Ctx) -> Result<()> {
        if rng.chance(self.p()) {
            self.apply_forced(s, rng, ctx)?;
        }
        Ok(())
    }

    /// Apply unconditionally (sampling this transform's random parameters).
    pub(crate) fn apply_forced(&self, s: &mut Sample, rng: &mut Rng, ctx: &Ctx) -> Result<()> {
        use Transform::*;
        let n = self.name();
        let (h, w, _) = s.image.dims();
        let (hf, wf) = (h as f64, w as f64);
        let rk = ctx.remove_invisible_kps;
        match self {
            Compose { transforms, .. } | Sequential { transforms, .. } => {
                for t in transforms {
                    t.run(s, rng, ctx)?;
                }
            }
            OneOf { transforms, .. } => {
                let total: f64 = transforms.iter().map(|t| t.p()).sum();
                if total > 0.0 {
                    let mut r = rng.uniform(0.0, total);
                    let mut chosen = transforms.len() - 1;
                    for (i, t) in transforms.iter().enumerate() {
                        if r < t.p() {
                            chosen = i;
                            break;
                        }
                        r -= t.p();
                    }
                    transforms[chosen].apply_forced(s, rng, ctx)?;
                }
            }
            SomeOf {
                transforms,
                n: k,
                replace,
                ..
            } => {
                let len = transforms.len();
                if len == 0 {
                    return Ok(());
                }
                let mut idx: Vec<usize> = if *replace {
                    (0..*k).map(|_| rng.int_inclusive(0, len as i64 - 1) as usize).collect()
                } else {
                    let mut all: Vec<usize> = (0..len).collect();
                    rng.shuffle(&mut all);
                    all.truncate((*k).min(len));
                    all
                };
                idx.sort_unstable();
                s.record(n, json!({"indices": idx}));
                for i in idx {
                    transforms[i].run(s, rng, ctx)?;
                }
            }
            HorizontalFlip { .. } => {
                s.record(n, json!({}));
                apply_geo(s, &Geo::HFlip, rk)?;
            }
            VerticalFlip { .. } => {
                s.record(n, json!({}));
                apply_geo(s, &Geo::VFlip, rk)?;
            }
            Transpose { .. } => {
                s.record(n, json!({}));
                apply_geo(s, &Geo::Transpose, rk)?;
            }
            RandomRotate90 { .. } => {
                let k = rng.int_inclusive(0, 3) as u8;
                s.record(n, json!({"factor": k}));
                if k != 0 {
                    apply_geo(s, &Geo::Rot90(k), rk)?;
                }
            }
            RandomCrop {
                height,
                width,
                pad_if_needed,
                pad_position,
                border_mode,
                fill,
                fill_mask,
                ..
            }
            | CenterCrop {
                height,
                width,
                pad_if_needed,
                pad_position,
                border_mode,
                fill,
                fill_mask,
                ..
            } => {
                let (mut h, mut w) = (h, w);
                if *pad_if_needed && (*height > h || *width > w) {
                    let g = pad_geo(rng, h, w, *height, *width, *pad_position, *border_mode, fill, fill_mask);
                    apply_geo(s, &g, rk)?;
                    (h, w, _) = s.image.dims();
                }
                if *height > h || *width > w {
                    return input(format!(
                        "{n}: crop size {height}x{width} is larger than the {h}x{w} image"
                    ));
                }
                let (y0, x0) = if matches!(self, RandomCrop { .. }) {
                    (
                        rng.int_inclusive(0, (h - height) as i64) as usize,
                        rng.int_inclusive(0, (w - width) as i64) as usize,
                    )
                } else {
                    ((h - height) / 2, (w - width) / 2)
                };
                let rect = Rect {
                    x0,
                    y0,
                    x1: x0 + width,
                    y1: y0 + height,
                };
                s.record(n, json!({"x_min": x0, "y_min": y0, "x_max": rect.x1, "y_max": rect.y1}));
                let g = Geo::CropResize {
                    rect,
                    oh: *height,
                    ow: *width,
                    interp: Interp::Linear,
                    mask_interp: Interp::Nearest,
                };
                apply_geo(s, &g, rk)?;
            }
            RandomResizedCrop {
                height,
                width,
                scale,
                ratio,
                interpolation,
                mask_interpolation,
                ..
            } => {
                let rect = rrc_rect(rng, h, w, *scale, *ratio);
                s.record(
                    n,
                    json!({"x_min": rect.x0, "y_min": rect.y0, "x_max": rect.x1, "y_max": rect.y1}),
                );
                let g = Geo::CropResize {
                    rect,
                    oh: *height,
                    ow: *width,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                };
                apply_geo(s, &g, rk)?;
            }
            Resize {
                height,
                width,
                interpolation,
                mask_interpolation,
                ..
            } => {
                s.record(n, json!({"height": height, "width": width}));
                let g = Geo::CropResize {
                    rect: Rect::full(h, w),
                    oh: *height,
                    ow: *width,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                };
                apply_geo(s, &g, rk)?;
            }
            LongestMaxSize {
                max_size,
                interpolation,
                mask_interpolation,
                ..
            }
            | SmallestMaxSize {
                max_size,
                interpolation,
                mask_interpolation,
                ..
            } => {
                let side = if matches!(self, LongestMaxSize { .. }) {
                    h.max(w)
                } else {
                    h.min(w)
                };
                let target = if max_size.len() == 1 {
                    max_size[0]
                } else {
                    max_size[rng.int_inclusive(0, max_size.len() as i64 - 1) as usize]
                };
                let sc = target as f64 / side as f64;
                let oh = ((hf * sc).round() as usize).max(1);
                let ow = ((wf * sc).round() as usize).max(1);
                s.record(n, json!({"height": oh, "width": ow}));
                let g = Geo::CropResize {
                    rect: Rect::full(h, w),
                    oh,
                    ow,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                };
                apply_geo(s, &g, rk)?;
            }
            PadIfNeeded {
                min_height,
                min_width,
                pad_height_divisor,
                pad_width_divisor,
                position,
                border_mode,
                fill,
                fill_mask,
                ..
            } => {
                let mut th = h.max(*min_height);
                let mut tw = w.max(*min_width);
                if let Some(d) = pad_height_divisor {
                    th = th.div_ceil(*d) * d;
                }
                if let Some(d) = pad_width_divisor {
                    tw = tw.div_ceil(*d) * d;
                }
                let g = pad_geo(rng, h, w, th, tw, *position, *border_mode, fill, fill_mask);
                if let Geo::Pad {
                    top,
                    bottom,
                    left,
                    right,
                    ..
                } = &g
                {
                    s.record(n, json!({"top": top, "bottom": bottom, "left": left, "right": right}));
                    if top + bottom + left + right > 0 {
                        apply_geo(s, &g, rk)?;
                    }
                }
            }
            Rotate {
                limit,
                interpolation,
                mask_interpolation,
                border_mode,
                fill,
                fill_mask,
                rotate_method,
                fit_output,
                crop_border,
                ..
            } => {
                let angle = rng.uniform(limit.0, limit.1);
                let (cx, cy) = (wf / 2.0, hf / 2.0);
                let mut m = Affine2::translate(-cx, -cy)
                    .then(&Affine2::rotate_deg(angle))
                    .then(&Affine2::translate(cx, cy));
                let (mut oh, mut ow) = (h, w);
                if *fit_output {
                    (m, oh, ow) = fit_affine(&m, h, w);
                } else if *crop_border {
                    let (cw, ch) = rotated_rect_with_max_area(wf, hf, angle.to_radians());
                    ow = (cw.floor() as usize).clamp(1, w);
                    oh = (ch.floor() as usize).clamp(1, h);
                    m = m.then(&Affine2::translate(-(wf - ow as f64) / 2.0, -(hf - oh as f64) / 2.0));
                }
                s.record(n, json!({"angle": angle, "height": oh, "width": ow}));
                let g = Geo::Affine {
                    m,
                    oh,
                    ow,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                    mode: *border_mode,
                    fill: fill.clone(),
                    fill_mask: fill_mask.clone(),
                    method: *rotate_method,
                };
                apply_geo(s, &g, rk)?;
            }
            Affine {
                scale_x,
                scale_y,
                keep_ratio,
                balanced_scale,
                translate_x,
                translate_y,
                translate_px,
                rotate,
                shear_x,
                shear_y,
                interpolation,
                mask_interpolation,
                border_mode,
                fill,
                fill_mask,
                rotate_method,
                fit_output,
                ..
            } => {
                let sample_scale = |rng: &mut Rng, r: (f64, f64)| {
                    if *balanced_scale && r.0 < 1.0 && r.1 > 1.0 {
                        if rng.chance(0.5) {
                            rng.uniform(r.0, 1.0)
                        } else {
                            rng.uniform(1.0, r.1)
                        }
                    } else {
                        rng.uniform(r.0, r.1)
                    }
                };
                let sx = sample_scale(rng, *scale_x);
                let sy = if *keep_ratio { sx } else { sample_scale(rng, *scale_y) };
                let mut tx = rng.uniform(translate_x.0, translate_x.1);
                let mut ty = rng.uniform(translate_y.0, translate_y.1);
                if !*translate_px {
                    tx *= wf;
                    ty *= hf;
                }
                let rot = rng.uniform(rotate.0, rotate.1);
                let shx = rng.uniform(shear_x.0, shear_x.1);
                let shy = rng.uniform(shear_y.0, shear_y.1);
                let (cx, cy) = (wf / 2.0, hf / 2.0);
                let mut m = Affine2::translate(-cx, -cy)
                    .then(&Affine2::scale(sx, sy))
                    .then(&Affine2::shear_deg(shx, shy))
                    .then(&Affine2::rotate_deg(rot))
                    .then(&Affine2::translate(cx + tx, cy + ty));
                let (mut oh, mut ow) = (h, w);
                if *fit_output {
                    (m, oh, ow) = fit_affine(&m, h, w);
                }
                s.record(
                    n,
                    json!({"scale": [sx, sy], "translate_px": [tx, ty], "rotate": rot, "shear": [shx, shy], "height": oh, "width": ow}),
                );
                let g = Geo::Affine {
                    m,
                    oh,
                    ow,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                    mode: *border_mode,
                    fill: fill.clone(),
                    fill_mask: fill_mask.clone(),
                    method: *rotate_method,
                };
                apply_geo(s, &g, rk)?;
            }
            ShiftScaleRotate {
                shift_limit_x,
                shift_limit_y,
                scale_limit,
                rotate_limit,
                interpolation,
                mask_interpolation,
                border_mode,
                fill,
                fill_mask,
                rotate_method,
                ..
            } => {
                let angle = rng.uniform(rotate_limit.0, rotate_limit.1);
                let sc = rng.uniform(scale_limit.0, scale_limit.1);
                let dx = rng.uniform(shift_limit_x.0, shift_limit_x.1) * wf;
                let dy = rng.uniform(shift_limit_y.0, shift_limit_y.1) * hf;
                let (cx, cy) = (wf / 2.0, hf / 2.0);
                let m = Affine2::translate(-cx, -cy)
                    .then(&Affine2::scale(sc, sc))
                    .then(&Affine2::rotate_deg(angle))
                    .then(&Affine2::translate(cx + dx, cy + dy));
                s.record(n, json!({"angle": angle, "scale": sc, "dx_px": dx, "dy_px": dy}));
                let g = Geo::Affine {
                    m,
                    oh: h,
                    ow: w,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                    mode: *border_mode,
                    fill: fill.clone(),
                    fill_mask: fill_mask.clone(),
                    method: *rotate_method,
                };
                apply_geo(s, &g, rk)?;
            }
            Perspective {
                scale,
                keep_size,
                fit_output,
                interpolation,
                mask_interpolation,
                border_mode,
                fill,
                fill_mask,
                ..
            } => {
                let (hm, oh, ow, quad) = perspective_params(rng, h, w, *scale, *keep_size, *fit_output)?;
                s.record(n, json!({"matrix": hm.m, "points": quad, "height": oh, "width": ow}));
                let g = Geo::Perspective {
                    hm,
                    oh,
                    ow,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                    mode: *border_mode,
                    fill: fill.clone(),
                    fill_mask: fill_mask.clone(),
                };
                apply_geo(s, &g, rk)?;
            }
            ElasticTransform {
                alpha,
                sigma,
                interpolation,
                mask_interpolation,
                approximate,
                same_dxdy,
                noise_distribution,
                border_mode,
                fill,
                fill_mask,
                ..
            } => {
                let field = elastic_field(rng, h, w, *alpha, *sigma, *approximate, *same_dxdy, *noise_distribution);
                s.record(n, json!({"alpha": alpha, "sigma": sigma}));
                let g = Geo::Field {
                    field,
                    interp: *interpolation,
                    mask_interp: *mask_interpolation,
                    mode: *border_mode,
                    fill: fill.clone(),
                    fill_mask: fill_mask.clone(),
                };
                apply_geo(s, &g, rk)?;
            }
            ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
                ..
            } => {
                let fb = rng.uniform(brightness.0, brightness.1);
                let fc = rng.uniform(contrast.0, contrast.1);
                let fs = rng.uniform(saturation.0, saturation.1);
                let fh = rng.uniform(hue.0, hue.1);
                let mut order = [0usize, 1, 2, 3];
                rng.shuffle(&mut order);
                s.record(
                    n,
                    json!({"brightness": fb, "contrast": fc, "saturation": fs, "hue": fh, "order": order}),
                );
                s.for_each_image(|img| {
                    match img {
                        Buf::U8(a) => {
                            for o in order {
                                match o {
                                    0 => color::brightness_u8(a, fb),
                                    1 => color::contrast_u8(a, fc),
                                    2 => color::saturation_u8(a, fs),
                                    _ => color::hue_u8(a, fh),
                                }
                            }
                        }
                        Buf::F32(a) => {
                            for o in order {
                                match o {
                                    0 => color::brightness_f32(a, fb),
                                    1 => color::contrast_f32(a, fc),
                                    2 => color::saturation_f32(a, fs),
                                    _ => color::hue_f32(a, fh),
                                }
                            }
                        }
                        other => return input(format!("{n}: unsupported image dtype {}", other.dtype_name())),
                    }
                    Ok(())
                })?;
            }
            RandomBrightnessContrast {
                brightness_limit,
                contrast_limit,
                brightness_by_max,
                ensure_safe_range,
                ..
            } => {
                let c = rng.uniform(contrast_limit.0, contrast_limit.1);
                let b = rng.uniform(brightness_limit.0, brightness_limit.1);
                s.record(n, json!({"alpha": 1.0 + c, "beta": b}));
                s.for_each_image(|img| {
                    let max = if matches!(img, Buf::U8(_)) { 255.0 } else { 1.0 };
                    let mut alpha = 1.0 + c;
                    let mut beta = if *brightness_by_max {
                        b * max
                    } else {
                        b * color::mean_all(img)
                    };
                    if *ensure_safe_range {
                        (alpha, beta) = safe_brightness_contrast(alpha, beta, max);
                    }
                    match img {
                        Buf::U8(a) => color::lut_apply(a, &color::multiply_add_lut(alpha, beta)),
                        Buf::F32(a) => color::multiply_add_f32(a, alpha, beta),
                        other => return input(format!("{n}: unsupported image dtype {}", other.dtype_name())),
                    }
                    Ok(())
                })?;
            }
            HueSaturationValue {
                hue_shift_limit,
                sat_shift_limit,
                val_shift_limit,
                ..
            } => {
                let dh = rng.uniform(hue_shift_limit.0, hue_shift_limit.1);
                let ds = rng.uniform(sat_shift_limit.0, sat_shift_limit.1);
                let dv = rng.uniform(val_shift_limit.0, val_shift_limit.1);
                s.record(n, json!({"hue_shift": dh, "sat_shift": ds, "val_shift": dv}));
                s.for_each_image(|img| {
                    let c = img.dims().2;
                    if c != 1 && c != 3 {
                        return input(format!("{n} expects 1- or 3-channel images, got {c} channels"));
                    }
                    let gray = c == 1;
                    // Albumentations: hue/sat shifts are ignored on gray images; offsets
                    // are truncated to integers (OpenCV `add` with an int value)
                    let edit = HsvEdit {
                        hue_lut: if dh != 0.0 && !gray { Some(hue_lut(dh)) } else { None },
                        sat_add: if gray { 0 } else { ds as i32 },
                        val_add: dv as i32,
                        keep_gray_sat: true,
                    };
                    if dh == 0.0 && ds == 0.0 && dv == 0.0 {
                        return Ok(());
                    }
                    with_u8_io(img, |a| hsv_edit_u8(a, &edit))
                })?;
            }
            RandomGamma { gamma_limit, .. } => {
                let gamma = rng.uniform(gamma_limit.0, gamma_limit.1) / 100.0;
                s.record(n, json!({"gamma": gamma}));
                s.for_each_image(|img| {
                    match img {
                        Buf::U8(a) => color::lut_apply(a, &color::gamma_lut(gamma)),
                        Buf::F32(a) => color::gamma_f32(a, gamma),
                        other => return input(format!("{n}: unsupported image dtype {}", other.dtype_name())),
                    }
                    Ok(())
                })?;
            }
            CLAHE {
                clip_limit,
                tile_grid_size,
                ..
            } => {
                let clip = rng.uniform(clip_limit.0, clip_limit.1);
                s.record(n, json!({"clip_limit": clip}));
                let (tx, ty) = *tile_grid_size;
                s.for_each_image(|img| {
                    let c = img.dims().2;
                    if c != 1 && c != 3 {
                        return input(format!("{n} expects 1- or 3-channel images, got {c} channels"));
                    }
                    with_u8_io(img, |a| *a = clahe_u8(a, clip, tx, ty))
                })?;
            }
            GaussNoise {
                std_range,
                mean_range,
                per_channel,
                noise_scale_factor,
                ..
            } => {
                let sigma = rng.uniform(std_range.0, std_range.1);
                let mean = rng.uniform(mean_range.0, mean_range.1);
                let noise_seed = rng.next_u64();
                s.record(n, json!({"sigma": sigma, "mean": mean}));
                s.for_each_image(|img| {
                    let (h, w, c) = img.dims();
                    let max = if matches!(img, Buf::U8(_)) { 255.0 } else { 1.0 };
                    let nc = if *per_channel { c } else { 1 };
                    let mut nrng = Rng::seed_from_u64(noise_seed);
                    let noise = noise_map(
                        &mut nrng,
                        h,
                        w,
                        nc,
                        (mean * max) as f32,
                        (sigma * max) as f32,
                        *noise_scale_factor,
                    );
                    match img {
                        Buf::U8(a) => {
                            let d = a.as_slice_mut().unwrap();
                            if nc == c {
                                for (v, z) in d.iter_mut().zip(noise.iter()) {
                                    *v = (*v as f32 + z + 0.5).clamp(0.0, 255.0) as u8;
                                }
                            } else {
                                for (p, z) in d.chunks_exact_mut(c).zip(noise.iter()) {
                                    for v in p {
                                        *v = (*v as f32 + z + 0.5).clamp(0.0, 255.0) as u8;
                                    }
                                }
                            }
                        }
                        Buf::F32(a) => {
                            let d = a.as_slice_mut().unwrap();
                            for (i, v) in d.iter_mut().enumerate() {
                                let z = if nc == c { noise[i] } else { noise[i / c] };
                                *v = (*v + z).clamp(0.0, 1.0);
                            }
                        }
                        other => return input(format!("{n}: unsupported image dtype {}", other.dtype_name())),
                    }
                    Ok(())
                })?;
            }
            ToGray {
                num_output_channels,
                method,
                ..
            } => {
                s.record(n, json!({}));
                s.for_each_image(|img| {
                    let c = img.dims().2;
                    if c == 1 && *num_output_channels == 1 {
                        return Ok(()); // already gray
                    }
                    if c < 3 && *method == GrayMethod::WeightedAverage && c != 1 {
                        return input(format!("{n}: weighted_average needs 3 channels, got {c}"));
                    }
                    *img = match &*img {
                        Buf::U8(a) => Buf::U8(color::to_gray_u8(a, *method, *num_output_channels)),
                        Buf::F32(a) => Buf::F32(color::to_gray_f32(a, *method, *num_output_channels)),
                        other => return input(format!("{n}: unsupported image dtype {}", other.dtype_name())),
                    };
                    Ok(())
                })?;
            }
            CoarseDropout {
                num_holes_range,
                hole_height_range,
                hole_width_range,
                fill,
                fill_mask,
                bbox_handling,
                keypoint_handling,
                ..
            } => {
                let holes = sample_holes(rng, h, w, *num_holes_range, *hole_height_range, *hole_width_range);
                let fill_seed = rng.next_u64();
                s.record(n, json!({"holes": holes}));
                if holes.is_empty() {
                    return Ok(());
                }
                s.for_each_image(|img| {
                    let mut frng = Rng::seed_from_u64(fill_seed);
                    fill_holes(img, &holes, fill, &mut frng);
                    Ok(())
                })?;
                if let Some(fm) = fill_mask {
                    let mf = DropoutFill::Value(fm.clone());
                    let mut frng = Rng::seed_from_u64(fill_seed);
                    for m in s.masks.iter_mut() {
                        fill_holes(m, &holes, &mf, &mut frng);
                    }
                }
                if !s.bboxes.is_empty() && *bbox_handling != DropoutBboxes::Keep {
                    dropout_boxes(&mut s.bboxes, &holes, h, w, *bbox_handling == DropoutBboxes::Shrink);
                }
                let remove = match keypoint_handling {
                    DropoutKeypoints::Auto => rk,
                    DropoutKeypoints::Remove => true,
                    DropoutKeypoints::Keep => false,
                };
                if remove {
                    s.keypoints.retain(|k| {
                        !holes.iter().any(|&[x0, y0, x1, y1]| {
                            k.x >= x0 as f64 && k.x < x1 as f64 && k.y >= y0 as f64 && k.y < y1 as f64
                        })
                    });
                }
            }
            Normalize {
                mean,
                std,
                max_pixel_value,
                normalization,
                ..
            } => {
                s.record(n, json!({}));
                s.for_each_image(|img| {
                    *img = Buf::F32(match normalization {
                        NormMode::Standard => normalize(img, mean, std, *max_pixel_value)?,
                        mode => normalize_per_image(img, *mode),
                    });
                    Ok(())
                })?;
            }
            GaussianBlur {
                blur_limit,
                sigma_limit,
                ..
            } => {
                let ksize = if blur_limit.1 == 0 {
                    0
                } else {
                    let lo = (blur_limit.0.max(3)) | 1;
                    let hi = (blur_limit.1.max(lo)) | 1;
                    lo + 2 * rng.int_inclusive(0, ((hi - lo) / 2) as i64) as usize
                };
                let sigma = rng.uniform(sigma_limit.0, sigma_limit.1);
                let kernel = gaussian_kernel_1d(sigma, ksize);
                s.record(n, json!({"ksize": kernel.len(), "sigma": sigma}));
                s.for_each_image(|img| {
                    *img = match &*img {
                        Buf::U8(a) => Buf::U8(crate::ops::blur::gaussian_blur_u8(a, &kernel)),
                        other => crate::map_buf!(other, a => gaussian_blur(a, &kernel)),
                    };
                    Ok(())
                })?;
            }
        }
        Ok(())
    }
}

// ---- helpers -------------------------------------------------------------------

/// Run a u8-only op on a u8 image, or on a float image via `round(x * 255)`
/// and back (Albumentations' `uint8_io`).
fn with_u8_io(img: &mut Buf, f: impl FnOnce(&mut Array3<u8>)) -> Result<()> {
    match img {
        Buf::U8(a) => {
            f(a);
            Ok(())
        }
        Buf::F32(a) => {
            let mut u = a.mapv(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8);
            f(&mut u);
            *a = u.mapv(|v| v as f32 / 255.0);
            Ok(())
        }
        other => input(format!("unsupported image dtype {}", other.dtype_name())),
    }
}

#[allow(clippy::too_many_arguments)]
fn pad_geo(
    rng: &mut Rng,
    h: usize,
    w: usize,
    th: usize,
    tw: usize,
    position: PadPosition,
    mode: BorderMode,
    fill: &[f64],
    fill_mask: &[f64],
) -> Geo {
    let ph = th.saturating_sub(h);
    let pw = tw.saturating_sub(w);
    let (top, left) = match position {
        PadPosition::Center => (ph / 2, pw / 2),
        PadPosition::TopLeft => (0, 0),
        PadPosition::TopRight => (0, pw),
        PadPosition::BottomLeft => (ph, 0),
        PadPosition::BottomRight => (ph, pw),
        PadPosition::Random => (
            rng.int_inclusive(0, ph as i64) as usize,
            rng.int_inclusive(0, pw as i64) as usize,
        ),
    };
    Geo::Pad {
        top,
        bottom: ph - top,
        left,
        right: pw - left,
        mode,
        fill: fill.to_vec(),
        fill_mask: fill_mask.to_vec(),
    }
}

/// Translate `m` so the whole transformed `h x w` image is in view; returns the new map and size.
fn fit_affine(m: &Affine2, h: usize, w: usize) -> (Affine2, usize, usize) {
    let (wf, hf) = (w as f64, h as f64);
    let mut r = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in [(0.0, 0.0), (wf, 0.0), (0.0, hf), (wf, hf)] {
        let (px, py) = m.apply(x, y);
        r = (r.0.min(px), r.1.min(py), r.2.max(px), r.3.max(py));
    }
    // whole-pixel bounds keep the pixel grid aligned (like Albumentations' floor/ceil)
    let (x0, y0) = ((r.0 + 1e-6).floor(), (r.1 + 1e-6).floor());
    let ow = (((r.2 - 1e-6).ceil() - x0) as usize).max(1);
    let oh = (((r.3 - 1e-6).ceil() - y0) as usize).max(1);
    (m.then(&Affine2::translate(-x0, -y0)), oh, ow)
}

/// Largest axis-aligned rectangle inside a `w x h` rectangle rotated by `angle` (radians).
fn rotated_rect_with_max_area(w: f64, h: f64, angle: f64) -> (f64, f64) {
    if w <= 0.0 || h <= 0.0 {
        return (0.0, 0.0);
    }
    let wide = w >= h;
    let (long, short) = if wide { (w, h) } else { (h, w) };
    let (sa, ca) = (angle.sin().abs(), angle.cos().abs());
    let (wr, hr) = if short <= 2.0 * sa * ca * long || (sa - ca).abs() < 1e-10 {
        let x = 0.5 * short;
        if wide {
            (x / sa.max(1e-12), x / ca.max(1e-12))
        } else {
            (x / ca.max(1e-12), x / sa.max(1e-12))
        }
    } else {
        let cos2a = ca * ca - sa * sa;
        ((w * ca - h * sa) / cos2a, (h * ca - w * sa) / cos2a)
    };
    (wr.min(w), hr.min(h))
}

fn safe_brightness_contrast(alpha: f64, beta: f64, max: f64) -> (f64, f64) {
    if alpha > 0.0 {
        let b = beta.clamp(0.0, max);
        (alpha.min((max - b) / max), b)
    } else {
        let b = beta.min(max);
        (alpha.max(-b / max), b)
    }
}

/// Homography, output height, output width, source quadrilateral.
type PerspectiveParams = (Homography, usize, usize, [(f64, f64); 4]);

/// Random perspective: returns the homography (continuous coords), the output
/// size and the jittered source quadrilateral (tl, tr, br, bl).
fn perspective_params(
    rng: &mut Rng,
    h: usize,
    w: usize,
    scale: (f64, f64),
    keep_size: bool,
    fit_output: bool,
) -> Result<PerspectiveParams> {
    let (wf, hf) = (w as f64, h as f64);
    let sc = rng.uniform(scale.0, scale.1);
    let mut j = [0.0f64; 8];
    for v in j.iter_mut() {
        *v = (rng.normal() * sc).abs() % 0.32;
    }
    let quad = [
        (j[0] * wf, j[1] * hf),
        ((1.0 - j[2]) * wf, j[3] * hf),
        ((1.0 - j[4]) * wf, (1.0 - j[5]) * hf),
        (j[6] * wf, (1.0 - j[7]) * hf),
    ];
    let dist = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1).max(2.0);
    let mw = dist(quad[1], quad[0]).max(dist(quad[2], quad[3]));
    let mh = dist(quad[2], quad[1]).max(dist(quad[3], quad[0]));
    let (mut ow, mut oh) = ((mw as usize).max(1), (mh as usize).max(1));
    let (owf, ohf) = (ow as f64, oh as f64);
    let Some(mut hm) = Homography::from_quad(quad, [(0.0, 0.0), (owf, 0.0), (owf, ohf), (0.0, ohf)]) else {
        return input("Perspective: degenerate quadrilateral");
    };
    if fit_output {
        let mut r = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (x, y) in [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)] {
            let Some((px, py)) = hm.apply(x, y) else {
                return input("Perspective: image corner maps to infinity");
            };
            r = (r.0.min(px), r.1.min(py), r.2.max(px), r.3.max(py));
        }
        hm = hm.then(&Homography::from_affine(&Affine2::translate(-r.0, -r.1)));
        ow = ((r.2 - r.0).round() as usize).max(1);
        oh = ((r.3 - r.1).round() as usize).max(1);
    }
    if keep_size {
        hm = hm.then(&Homography::from_affine(&Affine2::scale(
            wf / ow as f64,
            hf / oh as f64,
        )));
        (ow, oh) = (w, h);
    }
    Ok((hm, oh, ow, quad))
}

/// Separable blur of an f32 plane with replicate borders.
fn blur_plane(src: &[f32], h: usize, w: usize, kernel: &[f32]) -> Vec<f32> {
    use crate::border::{BorderMode, border_index};
    let r = kernel.len() / 2;
    let mut tmp = vec![0f32; h * w];
    let mut padded = vec![0f32; w + 2 * r];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for (x, p) in padded.iter_mut().enumerate() {
            *p = row[border_index(x as isize - r as isize, w, BorderMode::Replicate).unwrap()];
        }
        let o = &mut tmp[y * w..(y + 1) * w];
        for (j, &k) in kernel.iter().enumerate() {
            for (ov, &pv) in o.iter_mut().zip(&padded[j..j + w]) {
                *ov += k * pv;
            }
        }
    }
    let mut out = vec![0f32; h * w];
    for y in 0..h {
        let o = &mut out[y * w..(y + 1) * w];
        for (j, &k) in kernel.iter().enumerate() {
            let sy = border_index(y as isize + j as isize - r as isize, h, BorderMode::Replicate).unwrap();
            for (ov, &tv) in o.iter_mut().zip(&tmp[sy * w..(sy + 1) * w]) {
                *ov += k * tv;
            }
        }
    }
    out
}

/// Bilinear upsampling of an f32 plane (pixel-centre aligned).
fn upsample_plane(src: &[f32], sh: usize, sw: usize, h: usize, w: usize) -> Vec<f32> {
    let a = Array3::from_shape_vec((sh, sw, 1), src.to_vec()).unwrap();
    crate::ops::resize(&a, Rect::full(sh, sw), h, w, Interp::Linear)
        .into_raw_vec_and_offset()
        .0
}

#[allow(clippy::too_many_arguments)]
fn elastic_field(
    rng: &mut Rng,
    h: usize,
    w: usize,
    alpha: f64,
    sigma: f64,
    approximate: bool,
    same: bool,
    dist: NoiseDistribution,
) -> DispField {
    // Large kernels are expensive: for sigma > 8 the noise is generated on a grid
    // `f = sigma / 4` times coarser, blurred with sigma 4 and upsampled; dividing
    // by `f` keeps the field's standard deviation (blurred white noise scales as 1/sigma).
    let f = if approximate || sigma <= 8.0 { 1.0 } else { sigma / 4.0 };
    let (gh, gw) = (
        ((h as f64 / f).ceil() as usize).max(1),
        ((w as f64 / f).ceil() as usize).max(1),
    );
    let gsigma = sigma / f;
    let ksize = if approximate {
        17
    } else {
        ((gsigma * 8.0 + 1.0).round() as usize) | 1
    };
    let kernel = gaussian_kernel_1d(gsigma, ksize.min(2 * gh.max(gw) + 1));
    let nf = if same { 1 } else { 2 };
    let mut fields: Vec<Vec<f32>> = (0..nf)
        .map(|_| {
            let mut v = vec![0f32; gh * gw];
            match dist {
                NoiseDistribution::Gaussian => fill_normal(rng, &mut v, 0.0, 1.0),
                NoiseDistribution::Uniform => fill_uniform(rng, &mut v, -1.0, 1.0),
            }
            v
        })
        .collect();
    if dist == NoiseDistribution::Gaussian {
        let m = fields.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
        if m > 1e-6 {
            fields.iter_mut().flatten().for_each(|v| *v /= m);
        }
    }
    let scale = (alpha / f) as f32;
    let fields: Vec<Vec<f32>> = fields
        .into_iter()
        .map(|v| {
            let mut b = blur_plane(&v, gh, gw, &kernel);
            b.iter_mut().for_each(|x| *x *= scale);
            if (gh, gw) != (h, w) {
                upsample_plane(&b, gh, gw, h, w)
            } else {
                b
            }
        })
        .collect();
    let dx = fields[0].clone();
    let dy = if same { fields[0].clone() } else { fields[1].clone() };
    DispField { h, w, dx, dy }
}

/// `n` holes `[x0, y0, x1, y1]` (Albumentations' CoarseDropout sampling rules).
fn sample_holes(
    rng: &mut Rng,
    h: usize,
    w: usize,
    num: (usize, usize),
    hr: (f64, f64),
    wr: (f64, f64),
) -> Vec<[usize; 4]> {
    let k = rng.int_inclusive(num.0 as i64, num.1 as i64) as usize;
    let pixels = hr.1 >= 1.0;
    (0..k)
        .map(|_| {
            let (hh, hw) = if pixels {
                let hh = rng.int_inclusive(hr.0 as i64, (hr.1.min(h as f64)) as i64) as usize;
                let hw = rng.int_inclusive(wr.0 as i64, (wr.1.min(w as f64)) as i64) as usize;
                (hh, hw)
            } else {
                (
                    (h as f64 * rng.uniform(hr.0, hr.1)) as usize,
                    (w as f64 * rng.uniform(wr.0, wr.1)) as usize,
                )
            };
            let (hh, hw) = (hh.min(h), hw.min(w));
            let y0 = rng.int_inclusive(0, (h - hh) as i64) as usize;
            let x0 = rng.int_inclusive(0, (w - hw) as i64) as usize;
            [x0, y0, x0 + hw, y0 + hh]
        })
        .collect()
}

fn fill_holes(img: &mut Buf, holes: &[[usize; 4]], fill: &DropoutFill, rng: &mut Rng) {
    fn go<T: crate::buffer::Element>(
        a: &mut Array3<T>,
        holes: &[[usize; 4]],
        fill: &DropoutFill,
        rng: &mut Rng,
        max: f64,
        int: bool,
    ) {
        let (_, w, c) = a.dim();
        let d = a.as_slice_mut().unwrap();
        let draw = |rng: &mut Rng| -> T {
            if int {
                T::from_f64(rng.int_inclusive(0, max as i64) as f64)
            } else {
                T::from_f64(rng.uniform(0.0, max))
            }
        };
        for &[x0, y0, x1, y1] in holes {
            let color: Vec<T> = match fill {
                DropoutFill::Value(v) => (0..c)
                    .map(|k| T::from_f64(*v.get(k).or(v.last()).unwrap_or(&0.0)))
                    .collect(),
                DropoutFill::Mode(m) if m == "random_uniform" => (0..c).map(|_| draw(rng)).collect(),
                DropoutFill::Mode(_) => vec![],
            };
            for y in y0..y1 {
                let row = &mut d[(y * w + x0) * c..(y * w + x1) * c];
                if color.is_empty() {
                    row.iter_mut().for_each(|v| *v = draw(rng));
                } else {
                    for p in row.chunks_exact_mut(c) {
                        p.copy_from_slice(&color);
                    }
                }
            }
        }
    }
    match img {
        Buf::U8(a) => go(a, holes, fill, rng, 255.0, true),
        Buf::U16(a) => go(a, holes, fill, rng, 65535.0, true),
        Buf::I32(a) => go(a, holes, fill, rng, 255.0, true),
        Buf::F32(a) => go(a, holes, fill, rng, 1.0, false),
    }
}

/// Update boxes for dropout holes: visibility *= uncovered fraction (pixel grid);
/// optionally shrink to the uncovered extent; drop fully covered boxes.
fn dropout_boxes(boxes: &mut Vec<crate::targets::BBox>, holes: &[[usize; 4]], h: usize, w: usize, shrink: bool) {
    let mut covered = vec![false; h * w];
    for &[x0, y0, x1, y1] in holes {
        for y in y0..y1 {
            covered[y * w + x0..y * w + x1].iter_mut().for_each(|c| *c = true);
        }
    }
    boxes.retain_mut(|b| {
        let px0 = (b.x0.floor().max(0.0) as usize).min(w);
        let py0 = (b.y0.floor().max(0.0) as usize).min(h);
        let px1 = (b.x1.ceil() as usize).min(w);
        let py1 = (b.y1.ceil() as usize).min(h);
        let total = (px1.saturating_sub(px0)) * (py1.saturating_sub(py0));
        if total == 0 {
            return true;
        }
        let (mut vis, mut ext) = (0usize, (usize::MAX, usize::MAX, 0usize, 0usize));
        for y in py0..py1 {
            for x in px0..px1 {
                if !covered[y * w + x] {
                    vis += 1;
                    ext = (ext.0.min(x), ext.1.min(y), ext.2.max(x + 1), ext.3.max(y + 1));
                }
            }
        }
        if vis == 0 {
            return false;
        }
        b.visibility *= vis as f64 / total as f64;
        if shrink {
            b.x0 = b.x0.max(ext.0 as f64);
            b.y0 = b.y0.max(ext.1 as f64);
            b.x1 = b.x1.min(ext.2 as f64);
            b.y1 = b.y1.min(ext.3 as f64);
        }
        true
    });
}

fn noise_map(rng: &mut Rng, h: usize, w: usize, c: usize, mean: f32, std: f32, scale_factor: f64) -> Vec<f32> {
    if scale_factor >= 1.0 {
        let mut v = vec![0f32; h * w * c];
        fill_normal(rng, &mut v, mean, std);
        return v;
    }
    let sh = ((h as f64 * scale_factor) as usize).max(1);
    let sw = ((w as f64 * scale_factor) as usize).max(1);
    let mut v = vec![0f32; sh * sw * c];
    fill_normal(rng, &mut v, mean, std);
    let a = Array3::from_shape_vec((sh, sw, c), v).unwrap();
    crate::ops::resize(&a, Rect::full(sh, sw), h, w, Interp::Linear)
        .into_raw_vec_and_offset()
        .0
}

fn normalize_per_image(img: &Buf, mode: NormMode) -> Array3<f32> {
    let (h, w, c) = img.dims();
    let f: Vec<f32> = crate::with_buf!(img, a => a.iter().map(|v| crate::buffer::Element::to_f32(*v)).collect());
    // Albumentations: no epsilon for float images in `image_per_channel` (cv2.meanStdDev path)
    let eps = if mode == NormMode::ImagePerChannel && !matches!(img, Buf::U8(_)) {
        0.0
    } else {
        1e-4f64
    };
    let per_channel = matches!(mode, NormMode::ImagePerChannel | NormMode::MinMaxPerChannel) && c > 1;
    let groups: Vec<Vec<usize>> = if per_channel {
        (0..c).map(|k| vec![k]).collect()
    } else {
        vec![(0..c).collect()]
    };
    let mut out = f.clone();
    for g in groups {
        let vals = || f.chunks_exact(c).flat_map(|p| g.iter().map(move |&k| p[k] as f64));
        let n = (h * w * g.len()).max(1) as f64;
        let (sub, div, clip) = match mode {
            NormMode::Image | NormMode::ImagePerChannel => {
                let mean = vals().sum::<f64>() / n;
                let var = vals().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                (mean, var.sqrt() + eps, true)
            }
            _ => {
                let (mn, mx) = vals().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
                let range = if per_channel {
                    mx - mn + eps
                } else {
                    (mx - mn).max(f64::MIN_POSITIVE)
                };
                (mn, range, per_channel)
            }
        };
        for p in out.chunks_exact_mut(c) {
            for &k in &g {
                let v = (p[k] as f64 - sub) / div;
                p[k] = if clip { v.clamp(-20.0, 20.0) } else { v } as f32;
            }
        }
    }
    Array3::from_shape_vec((h, w, c), out).unwrap()
}

/// torchvision / Albumentations RandomResizedCrop window sampling.
fn rrc_rect(rng: &mut Rng, h: usize, w: usize, scale: (f64, f64), ratio: (f64, f64)) -> Rect {
    let area = (h * w) as f64;
    let (lr0, lr1) = (ratio.0.ln(), ratio.1.ln());
    for _ in 0..10 {
        let target = area * rng.uniform(scale.0, scale.1);
        let aspect = rng.uniform(lr0, lr1).exp();
        let cw = (target * aspect).sqrt().round() as usize;
        let ch = (target / aspect).sqrt().round() as usize;
        if cw > 0 && ch > 0 && cw <= w && ch <= h {
            let y0 = rng.int_inclusive(0, (h - ch) as i64) as usize;
            let x0 = rng.int_inclusive(0, (w - cw) as i64) as usize;
            return Rect {
                x0,
                y0,
                x1: x0 + cw,
                y1: y0 + ch,
            };
        }
    }
    // fallback: central crop with the closest allowed aspect ratio
    let in_ratio = w as f64 / h as f64;
    let (cw, ch) = if in_ratio < ratio.0 {
        (w, ((w as f64 / ratio.0).round() as usize).clamp(1, h))
    } else if in_ratio > ratio.1 {
        (((h as f64 * ratio.1).round() as usize).clamp(1, w), h)
    } else {
        (w, h)
    };
    let y0 = (h - ch) / 2;
    let x0 = (w - cw) / 2;
    Rect {
        x0,
        y0,
        x1: x0 + cw,
        y1: y0 + ch,
    }
}

// ---- ergonomic constructors for Rust users ---------------------------------

impl Transform {
    pub fn compose(transforms: Vec<Transform>) -> Self {
        Transform::Compose { transforms, p: 1.0 }
    }
    pub fn one_of(transforms: Vec<Transform>, p: f64) -> Self {
        Transform::OneOf { transforms, p }
    }
    pub fn some_of(transforms: Vec<Transform>, n: usize, p: f64) -> Self {
        Transform::SomeOf {
            transforms,
            n,
            replace: false,
            p,
        }
    }
    pub fn sequential(transforms: Vec<Transform>, p: f64) -> Self {
        Transform::Sequential { transforms, p }
    }
    pub fn hflip(p: f64) -> Self {
        Transform::HorizontalFlip { p }
    }
    pub fn vflip(p: f64) -> Self {
        Transform::VerticalFlip { p }
    }
    pub fn transpose(p: f64) -> Self {
        Transform::Transpose { p }
    }
    pub fn random_rotate90(p: f64) -> Self {
        Transform::RandomRotate90 { p }
    }
    pub fn random_crop(height: usize, width: usize) -> Self {
        Transform::RandomCrop {
            height,
            width,
            pad_if_needed: false,
            pad_position: PadPosition::Center,
            border_mode: BorderMode::Constant,
            fill: d_fill(),
            fill_mask: d_fill(),
            p: 1.0,
        }
    }
    pub fn center_crop(height: usize, width: usize) -> Self {
        Transform::CenterCrop {
            height,
            width,
            pad_if_needed: false,
            pad_position: PadPosition::Center,
            border_mode: BorderMode::Constant,
            fill: d_fill(),
            fill_mask: d_fill(),
            p: 1.0,
        }
    }
    pub fn resize(height: usize, width: usize) -> Self {
        Transform::Resize {
            height,
            width,
            interpolation: Interp::Linear,
            mask_interpolation: Interp::Nearest,
            p: 1.0,
        }
    }
    pub fn random_resized_crop(height: usize, width: usize, scale: (f64, f64)) -> Self {
        Transform::RandomResizedCrop {
            height,
            width,
            scale,
            ratio: d_rrc_ratio(),
            interpolation: Interp::Linear,
            mask_interpolation: Interp::Nearest,
            p: 1.0,
        }
    }
    pub fn pad_if_needed(min_height: usize, min_width: usize) -> Self {
        Transform::PadIfNeeded {
            min_height,
            min_width,
            pad_height_divisor: None,
            pad_width_divisor: None,
            position: PadPosition::Center,
            border_mode: BorderMode::Constant,
            fill: d_fill(),
            fill_mask: d_fill(),
            p: 1.0,
        }
    }
    pub fn rotate(limit: (f64, f64), p: f64) -> Self {
        Transform::Rotate {
            limit,
            interpolation: Interp::Linear,
            mask_interpolation: Interp::Nearest,
            border_mode: BorderMode::Constant,
            fill: d_fill(),
            fill_mask: d_fill(),
            rotate_method: BboxMethod::LargestBox,
            fit_output: false,
            crop_border: false,
            p,
        }
    }
    /// Affine with the given ranges (scale is isotropic: `keep_ratio`).
    pub fn affine(scale: (f64, f64), translate: (f64, f64), rotate: (f64, f64), shear: (f64, f64), p: f64) -> Self {
        Transform::Affine {
            scale_x: scale,
            scale_y: scale,
            keep_ratio: true,
            balanced_scale: false,
            translate_x: translate,
            translate_y: translate,
            translate_px: false,
            rotate,
            shear_x: shear,
            shear_y: shear,
            interpolation: Interp::Linear,
            mask_interpolation: Interp::Nearest,
            border_mode: BorderMode::Constant,
            fill: d_fill(),
            fill_mask: d_fill(),
            rotate_method: BboxMethod::LargestBox,
            fit_output: false,
            p,
        }
    }
    pub fn perspective(scale: (f64, f64), p: f64) -> Self {
        Transform::Perspective {
            scale,
            keep_size: true,
            fit_output: false,
            interpolation: Interp::Linear,
            mask_interpolation: Interp::Nearest,
            border_mode: BorderMode::Constant,
            fill: d_fill(),
            fill_mask: d_fill(),
            p,
        }
    }
    pub fn elastic(alpha: f64, sigma: f64, p: f64) -> Self {
        Transform::ElasticTransform {
            alpha,
            sigma,
            interpolation: Interp::Linear,
            mask_interpolation: Interp::Nearest,
            approximate: false,
            same_dxdy: false,
            noise_distribution: NoiseDistribution::Gaussian,
            border_mode: BorderMode::Constant,
            fill: d_fill(),
            fill_mask: d_fill(),
            p,
        }
    }
    pub fn color_jitter(b: f64, c: f64, s: f64, hue: f64, p: f64) -> Self {
        Transform::ColorJitter {
            brightness: ((1.0 - b).max(0.0), 1.0 + b),
            contrast: ((1.0 - c).max(0.0), 1.0 + c),
            saturation: ((1.0 - s).max(0.0), 1.0 + s),
            hue: (-hue, hue),
            p,
        }
    }
    pub fn brightness_contrast(brightness: f64, contrast: f64, p: f64) -> Self {
        Transform::RandomBrightnessContrast {
            brightness_limit: (-brightness, brightness),
            contrast_limit: (-contrast, contrast),
            brightness_by_max: true,
            ensure_safe_range: false,
            p,
        }
    }
    pub fn hue_saturation_value(hue: f64, sat: f64, val: f64, p: f64) -> Self {
        Transform::HueSaturationValue {
            hue_shift_limit: (-hue, hue),
            sat_shift_limit: (-sat, sat),
            val_shift_limit: (-val, val),
            p,
        }
    }
    pub fn coarse_dropout(num_holes: (usize, usize), hole_size: (f64, f64), p: f64) -> Self {
        Transform::CoarseDropout {
            num_holes_range: num_holes,
            hole_height_range: hole_size,
            hole_width_range: hole_size,
            fill: DropoutFill::default(),
            fill_mask: None,
            bbox_handling: DropoutBboxes::Shrink,
            keypoint_handling: DropoutKeypoints::Auto,
            p,
        }
    }
    pub fn normalize_imagenet() -> Self {
        Transform::Normalize {
            mean: d_mean(),
            std: d_std(),
            max_pixel_value: 255.0,
            normalization: NormMode::Standard,
            p: 1.0,
        }
    }
    pub fn gaussian_blur(blur_limit: (usize, usize), sigma_limit: (f64, f64), p: f64) -> Self {
        Transform::GaussianBlur {
            blur_limit,
            sigma_limit,
            p,
        }
    }
}
