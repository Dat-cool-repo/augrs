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
pub use transforms::{
    DropoutBboxes, DropoutFill, DropoutKeypoints, Interp, NoiseDistribution, NormMode, PadPosition, ToGrayMethod,
    Transform,
};
