//! `augrs-core`: a fast, deterministic image-augmentation engine operating on
//! HWC arrays, with bounding boxes, keypoints and masks transformed jointly with
//! the image by every geometric transform.
//!
//! # Coordinate convention
//! All geometry is done in *continuous* pixel coordinates: the image covers
//! `[0, W] x [0, H]` and the centre of pixel `(col i, row j)` is at
//! `(i + 0.5, j + 0.5)`. Every geometric transform is an exact affine map in
//! that space, and the same map is applied to the image (by inverse warping),
//! to masks (nearest neighbour), to boxes (corners or inscribed ellipse) and to
//! keypoints. This is what makes the targets stay consistent.
//!
//! # Determinism
//! Randomness comes from [`rng::Rng`] (xoshiro256++ seeded through splitmix64).
//! A [`Pipeline`] created with a seed produces the same sequence of outputs on
//! every platform; batch sample `i` uses a seed derived from `(batch_seed, i)`
//! so results do not depend on the number of threads.

pub mod border;
pub mod buffer;
pub mod error;
mod geo;
pub mod geometry;
pub mod ops;
pub mod pipeline;
pub mod rng;
pub mod targets;
pub mod transforms;

pub use border::BorderMode;
pub use buffer::{Buf, Element};
pub use error::{AugError, Result};
pub use geometry::{Affine2, Homography};
pub use pipeline::{Applied, Input, Output, Pipeline, PipelineSpec, Sample};
pub use rng::Rng;
pub use targets::{BBox, BboxFormat, BboxMethod, BboxParams, Keypoint, KeypointFormat, KeypointParams};
/// Largest image side (in pixels) that a size parameter may request or a transform may produce.
pub const MAX_SIDE: usize = 1 << 20;

/// Largest number of elements (`height * width * channels`) of an image that augrs accepts or
/// produces (this also keeps all index arithmetic inside `i32`). Fuzzing builds use a smaller
/// limit so that legitimately large outputs do not hit the fuzzer's memory limit.
pub const MAX_ELEMENTS: usize = if cfg!(fuzzing) { 1 << 24 } else { i32::MAX as usize };

/// Deepest nesting of compositions (`Compose`, `OneOf`, `SomeOf`, `Sequential`) accepted.
pub const MAX_NESTING: usize = 32;

/// Check that an `h x w x c` image is within [`MAX_SIDE`] and [`MAX_ELEMENTS`].
pub(crate) fn check_size(h: usize, w: usize, c: usize, what: &str) -> Result<()> {
    let n = h.checked_mul(w).and_then(|n| n.checked_mul(c.max(1)));
    match n {
        Some(n) if h <= MAX_SIDE && w <= MAX_SIDE && n <= MAX_ELEMENTS => Ok(()),
        _ => error::input(format!(
            "{what} would be {h}x{w}x{c}, over augrs' size limit ({MAX_SIDE} px per side, {MAX_ELEMENTS} elements)"
        )),
    }
}

pub use transforms::{
    DropoutBboxes, DropoutFill, DropoutKeypoints, Interp, NoiseDistribution, NormMode, PadPosition, ToGrayMethod,
    Transform,
};
