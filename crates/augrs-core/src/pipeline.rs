//! `Pipeline` = a root `Compose` + target configuration + a seeded RNG.

use crate::buffer::Buf;
use crate::error::{AugError, Result, input};
use crate::rng::{Rng, derive_seed};
use crate::targets::{
    BBox, BboxParams, Keypoint, KeypointParams, boxes_in, boxes_out, keypoints_in, keypoints_out, validate_bbox_params,
};
use crate::transforms::{Ctx, Transform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

fn one() -> f64 {
    1.0
}

/// Serialisable pipeline description.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineSpec {
    pub transforms: Vec<Transform>,
    #[serde(default = "one")]
    pub p: f64,
    #[serde(default)]
    pub bbox_params: Option<BboxParams>,
    #[serde(default)]
    pub keypoint_params: Option<KeypointParams>,
}

impl PipelineSpec {
    pub fn new(transforms: Vec<Transform>) -> Self {
        PipelineSpec {
            transforms,
            p: 1.0,
            bbox_params: None,
            keypoint_params: None,
        }
    }
    pub fn with_bboxes(mut self, p: BboxParams) -> Self {
        self.bbox_params = Some(p);
        self
    }
    pub fn with_keypoints(mut self, p: KeypointParams) -> Self {
        self.keypoint_params = Some(p);
        self
    }
}

/// A transform that was applied, with its sampled parameters (for replay/debugging).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Applied {
    pub name: String,
    pub params: serde_json::Value,
}

/// Working state that flows through the transforms.
#[derive(Clone, Debug)]
pub struct Sample {
    pub image: Buf,
    /// Additional images (`additional_targets` of type image): same geometry and
    /// same sampled photometric parameters as `image`.
    pub extra_images: Vec<Buf>,
    pub masks: Vec<Buf>,
    pub bboxes: Vec<BBox>,
    pub keypoints: Vec<Keypoint>,
    pub(crate) applied: Option<Vec<Applied>>,
}

impl Sample {
    /// Run `f` on the primary image and every extra image.
    pub(crate) fn for_each_image(
        &mut self,
        mut f: impl FnMut(&mut Buf) -> crate::error::Result<()>,
    ) -> crate::error::Result<()> {
        f(&mut self.image)?;
        for img in self.extra_images.iter_mut() {
            f(img)?;
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn record(&mut self, name: &str, params: serde_json::Value) {
        if let Some(a) = self.applied.as_mut() {
            a.push(Applied {
                name: name.to_string(),
                params,
            });
        }
    }
}

/// Data to augment. Boxes are in the pipeline's `BboxParams.format`; keypoint
/// rows are in `KeypointParams.format` (unused trailing columns ignored).
#[derive(Clone, Debug)]
pub struct Input {
    pub image: Buf,
    /// Additional images transformed exactly like `image` (must have the same height and width).
    pub extra_images: Vec<Buf>,
    pub masks: Vec<Buf>,
    pub bboxes: Vec<[f64; 4]>,
    pub keypoints: Vec<[f64; 4]>,
}

impl Input {
    pub fn image(image: Buf) -> Self {
        Input {
            image,
            extra_images: vec![],
            masks: vec![],
            bboxes: vec![],
            keypoints: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Output {
    pub image: Buf,
    pub extra_images: Vec<Buf>,
    pub masks: Vec<Buf>,
    pub bboxes: Vec<[f64; 4]>,
    /// For each output box, the index of the input box it came from.
    pub bbox_ids: Vec<usize>,
    pub keypoints: Vec<[f64; 4]>,
    pub keypoint_ids: Vec<usize>,
    pub applied: Option<Vec<Applied>>,
    /// The per-sample seed that produced this output (replay with `apply_with_seed`).
    pub seed: u64,
}

pub struct Pipeline {
    spec: PipelineSpec,
    root: Transform,
    rng: Mutex<Rng>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline").field("spec", &self.spec).finish()
    }
}

impl Pipeline {
    pub fn new(spec: PipelineSpec, seed: Option<u64>) -> Result<Self> {
        let root = Transform::Compose {
            transforms: spec.transforms.clone(),
            p: spec.p,
        };
        root.validate()?;
        if let Some(bp) = &spec.bbox_params {
            validate_bbox_params(bp)?;
        }
        let rng = match seed {
            Some(s) => Rng::seed_from_u64(s),
            None => Rng::from_entropy(),
        };
        Ok(Pipeline {
            spec,
            root,
            rng: Mutex::new(rng),
        })
    }

    pub fn from_json(json: &str, seed: Option<u64>) -> Result<Self> {
        let spec: PipelineSpec =
            serde_json::from_str(json).map_err(|e| AugError::InvalidParam(format!("bad pipeline spec: {e}")))?;
        Pipeline::new(spec, seed)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.spec).expect("spec is serialisable")
    }

    pub fn spec(&self) -> &PipelineSpec {
        &self.spec
    }

    /// Reset the RNG stream (`None` = reseed from entropy).
    pub fn set_seed(&self, seed: Option<u64>) {
        let mut g = self.rng.lock().unwrap();
        *g = match seed {
            Some(s) => Rng::seed_from_u64(s),
            None => Rng::from_entropy(),
        };
    }

    /// Draw the next per-call seed from the pipeline's stream.
    pub fn next_seed(&self) -> u64 {
        self.rng.lock().unwrap().next_u64()
    }

    /// Augment one sample using the next seed of the pipeline's stream.
    pub fn apply(&self, inp: Input) -> Result<Output> {
        let seed = self.next_seed();
        self.apply_with_seed(inp, seed, false)
    }

    /// Augment one sample with an explicit seed (fully reproducible).
    pub fn apply_with_seed(&self, inp: Input, seed: u64, record: bool) -> Result<Output> {
        let (h, w, c) = inp.image.dims();
        if h == 0 || w == 0 || c == 0 {
            return input(format!("image must be non-empty, got shape ({h}, {w}, {c})"));
        }
        if !matches!(inp.image, Buf::U8(_) | Buf::F32(_)) {
            return input(format!(
                "image dtype must be uint8 or float32, got {}",
                inp.image.dtype_name()
            ));
        }
        for (i, m) in inp.extra_images.iter().enumerate() {
            let (mh, mw, _) = m.dims();
            if (mh, mw) != (h, w) {
                return input(format!(
                    "additional image {i} has shape ({mh}, {mw}) but the image is ({h}, {w})"
                ));
            }
            if !matches!(m, Buf::U8(_) | Buf::F32(_)) {
                return input(format!(
                    "additional image {i} dtype must be uint8 or float32, got {}",
                    m.dtype_name()
                ));
            }
        }
        for (i, m) in inp.masks.iter().enumerate() {
            let (mh, mw, _) = m.dims();
            if (mh, mw) != (h, w) {
                return input(format!("mask {i} has shape ({mh}, {mw}) but the image is ({h}, {w})"));
            }
        }
        let bboxes = match (&self.spec.bbox_params, inp.bboxes.is_empty()) {
            (_, true) => vec![],
            (Some(bp), false) => boxes_in(&inp.bboxes, bp, h, w)?,
            (None, false) => return input("bboxes were passed but the pipeline has no bbox_params"),
        };
        let keypoints = match (&self.spec.keypoint_params, inp.keypoints.is_empty()) {
            (_, true) => vec![],
            (Some(kp), false) => keypoints_in(&inp.keypoints, kp)?,
            (None, false) => return input("keypoints were passed but the pipeline has no keypoint_params"),
        };
        let mut s = Sample {
            image: inp.image.into_standard(),
            extra_images: inp.extra_images.into_iter().map(Buf::into_standard).collect(),
            masks: inp.masks.into_iter().map(Buf::into_standard).collect(),
            bboxes,
            keypoints,
            applied: if record { Some(vec![]) } else { None },
        };
        let ctx = Ctx {
            remove_invisible_kps: self
                .spec
                .keypoint_params
                .as_ref()
                .map(|k| k.remove_invisible)
                .unwrap_or(true),
        };
        let mut rng = Rng::seed_from_u64(seed);
        self.root.run(&mut s, &mut rng, &ctx)?;
        let (oh, ow, _) = s.image.dims();
        let (bboxes, bbox_ids) = match &self.spec.bbox_params {
            Some(bp) => boxes_out(&s.bboxes, bp, oh, ow),
            None => (vec![], vec![]),
        };
        let (keypoints, keypoint_ids) = match &self.spec.keypoint_params {
            Some(kp) => keypoints_out(&s.keypoints, kp),
            None => (vec![], vec![]),
        };
        Ok(Output {
            image: s.image,
            extra_images: s.extra_images,
            masks: s.masks,
            bboxes,
            bbox_ids,
            keypoints,
            keypoint_ids,
            applied: s.applied,
            seed,
        })
    }

    /// Seed used for sample `i` of a batch with base seed `base`.
    pub fn batch_sample_seed(base: u64, i: usize) -> u64 {
        derive_seed(base, i as u64)
    }

    /// Augment a batch in parallel on the current rayon pool. Sample `i` uses
    /// `derive_seed(seed, i)`, so results do not depend on the thread count.
    /// When `seed` is `None` the base seed is drawn from the pipeline's stream.
    pub fn apply_batch(&self, inputs: Vec<Input>, seed: Option<u64>, record: bool) -> Vec<Result<Output>> {
        let base = seed.unwrap_or_else(|| self.next_seed());
        inputs
            .into_par_iter()
            .enumerate()
            .map(|(i, inp)| self.apply_with_seed(inp, derive_seed(base, i as u64), record))
            .collect()
    }

    /// Like [`apply_batch`](Self::apply_batch) but builds each input lazily
    /// inside the worker (lets callers copy input buffers in parallel).
    pub fn apply_batch_with<F>(&self, n: usize, seed: Option<u64>, record: bool, make: F) -> Vec<Result<Output>>
    where
        F: Fn(usize) -> Result<Input> + Sync,
    {
        let base = seed.unwrap_or_else(|| self.next_seed());
        (0..n)
            .into_par_iter()
            .map(|i| {
                let inp = make(i)?;
                self.apply_with_seed(inp, derive_seed(base, i as u64), record)
            })
            .collect()
    }
}
