//! The JSON pipeline-spec parser (`Pipeline::from_json`, which the Python package uses for
//! every pipeline), followed by applying the pipeline to a small fixed sample.
#![no_main]

use augrs_core::{Buf, Input, Pipeline};
use libfuzzer_sys::fuzz_target;
use ndarray::Array3;

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else { return };
    let Ok(pipe) = Pipeline::from_json(s, Some(7)) else { return };
    // a spec that loads must round-trip through to_json
    let back = Pipeline::from_json(&pipe.to_json(), Some(7)).expect("to_json output does not reload");
    assert_eq!(back.spec(), pipe.spec());
    let img = Array3::from_shape_fn((13, 17, 3), |(y, x, c)| ((y * 31 + x * 7 + c * 50) % 256) as u8);
    let mut inp = Input::image(Buf::U8(img));
    inp.masks.push(Buf::U8(Array3::from_shape_fn((13, 17, 1), |(y, x, _)| ((x + y) % 3) as u8)));
    if let Some(bp) = &pipe.spec().bbox_params {
        inp.bboxes = match bp.format {
            augrs_core::BboxFormat::PascalVoc => vec![[1.0, 2.0, 9.0, 11.0], [0.0, 0.0, 17.0, 13.0]],
            augrs_core::BboxFormat::Coco => vec![[1.0, 2.0, 8.0, 9.0]],
            _ => vec![[0.5, 0.5, 0.2, 0.3]],
        };
    }
    if pipe.spec().keypoint_params.is_some() {
        inp.keypoints = vec![[3.0, 4.0, 10.0, 1.0], [16.5, 12.5, 0.0, 2.0]];
    }
    let nb = inp.bboxes.len();
    let nk = inp.keypoints.len();
    if let Ok(out) = pipe.apply_with_seed(inp, 1, true) {
        augrs_fuzz::check_output(pipe.spec(), nb, nk, &out);
    }
});
