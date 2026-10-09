//! Regression tests for the robustness fixes found by fuzzing (`fuzz/`): invalid parameters
//! are rejected by `Pipeline::new`, invalid targets by `apply`, degenerate inputs never panic,
//! and nothing allocates without bound.

use augrs_core::{
    BboxFormat, BboxParams, Buf, Input, Interp, KeypointFormat, KeypointParams, MAX_NESTING, Pipeline, PipelineSpec,
    Rng, Transform,
};
use ndarray::Array3;
use std::time::{Duration, Instant};

fn img(h: usize, w: usize, c: usize) -> Buf {
    Buf::U8(Array3::from_shape_fn((h, w, c), |(y, x, k)| {
        ((y * 31 + x * 7 + k * 50) % 256) as u8
    }))
}

fn spec(ts: Vec<Transform>) -> PipelineSpec {
    PipelineSpec::new(ts)
}

fn param_err(ts: Vec<Transform>) -> String {
    match Pipeline::new(spec(ts), Some(0)) {
        Err(augrs_core::AugError::InvalidParam(m)) => m,
        other => panic!("expected InvalidParam, got {other:?}"),
    }
}

fn json_err(json: &str) -> String {
    match Pipeline::from_json(json, Some(0)) {
        Err(augrs_core::AugError::InvalidParam(m)) => m,
        other => panic!("expected InvalidParam for {json}, got {other:?}"),
    }
}

#[test]
fn deep_nesting_is_rejected_not_a_stack_or_json_failure() {
    let mut t = Transform::hflip(0.5);
    for i in 0..(MAX_NESTING + 5) {
        t = if i % 2 == 0 {
            Transform::one_of(vec![t], 1.0)
        } else {
            Transform::some_of(vec![t], 1, 1.0)
        };
    }
    assert!(param_err(vec![t]).contains("nested"));
    // the deepest accepted pipeline must survive its own JSON round trip
    let mut t = Transform::hflip(0.5);
    for _ in 0..MAX_NESTING {
        t = Transform::one_of(vec![t], 1.0);
    }
    let p = Pipeline::new(spec(vec![t]), Some(0)).unwrap();
    Pipeline::from_json(&p.to_json(), Some(0)).unwrap();
}

#[test]
fn floats_round_trip_exactly_through_json() {
    // serde_json without `float_roundtrip` parsed 22.65 as 22.650000000000002
    let mut bp = BboxParams::new(BboxFormat::Coco);
    bp.min_area = 22.65;
    bp.min_visibility = 0.199;
    let s = spec(vec![Transform::rotate((-7.35, 13.3), 0.735)]).with_bboxes(bp);
    let p = Pipeline::new(s, Some(0)).unwrap();
    let back = Pipeline::from_json(&p.to_json(), Some(0)).unwrap();
    assert_eq!(back.spec(), p.spec());
}

#[test]
fn invalid_parameters_are_rejected() {
    use Transform as T;
    let cases: Vec<(Transform, &str)> = vec![
        (T::hflip(f64::NAN), "p must be in [0, 1]"),
        (T::hflip(1.5), "p must be in [0, 1]"),
        (T::hflip(-0.1), "p must be in [0, 1]"),
        (T::resize(0, 10), "height and width must be > 0"),
        (T::resize(1 << 21, 10), "limit"),
        (T::random_crop(usize::MAX, 4), "limit"),
        (T::rotate((f64::NAN, 1.0), 1.0), "invalid range"),
        (T::rotate((10.0, -10.0), 1.0), "invalid range"),
        (T::elastic(1.0, 0.0, 1.0), "sigma"),
        (T::elastic(f64::INFINITY, 50.0, 1.0), "alpha"),
        (T::perspective((-0.1, 0.1), 1.0), "scale must be >= 0"),
        (T::gaussian_blur((3, 1 << 31), (0.5, 1.0), 1.0), "kernel sizes up to"),
        (T::gaussian_blur((0, 0), (1.0, 1e6), 1.0), "derived from sigma"),
        // sigma so large that `(sigma * 3.5) as usize` saturates; used to overflow in the check (fuzz crash)
        (T::gaussian_blur((0, 0), (1.0, 1e300), 1.0), "derived from sigma"),
        (T::gaussian_blur((0, 0), (1.0, f64::MAX), 1.0), "derived from sigma"),
        (T::coarse_dropout((1, 1 << 40), (0.1, 0.2), 1.0), "holes"),
        (
            T::SomeOf {
                transforms: vec![T::hflip(1.0)],
                n: usize::MAX,
                replace: true,
                p: 1.0,
            },
            "exceeds the limit",
        ),
    ];
    for (t, want) in cases {
        let name = t.name();
        let m = param_err(vec![t]);
        assert!(m.contains(want), "{name}: {m:?} does not mention {want:?}");
    }
    // non-finite fills and normalisation constants
    let mut t = T::rotate((-10.0, 10.0), 1.0);
    if let T::Rotate { fill, .. } = &mut t {
        *fill = vec![f64::NAN];
    }
    assert!(param_err(vec![t]).contains("finite"));
    let t = T::Normalize {
        mean: vec![0.5],
        std: vec![f64::INFINITY],
        max_pixel_value: 255.0,
        normalization: augrs_core::NormMode::Standard,
        p: 1.0,
    };
    assert!(param_err(vec![t]).contains("finite"));
    // bbox params
    type Case = (fn(&mut BboxParams), &'static str);
    let cases: [Case; 3] = [
        (|b| b.min_area = f64::NAN, "finite"),
        (|b| b.max_accept_ratio = Some(f64::INFINITY), "max_accept_ratio"),
        (|b| b.min_visibility = 1.5, "min_visibility"),
    ];
    for (f, want) in cases {
        let mut bp = BboxParams::new(BboxFormat::PascalVoc);
        f(&mut bp);
        match Pipeline::new(spec(vec![]).with_bboxes(bp), None) {
            Err(e) => assert!(e.to_string().contains(want), "{e}"),
            Ok(_) => panic!("accepted invalid bbox params"),
        }
    }
    // JSON: negative / huge sizes and non-numbers are parse errors, not panics
    json_err(r#"{"transforms":[{"type":"Resize","height":-5,"width":3}]}"#);
    json_err(r#"{"transforms":[{"type":"Resize","height":1e400,"width":3}]}"#);
    json_err(r#"{"transforms":[{"type":"Resize","height":18446744073709551616,"width":3}]}"#);
    json_err(r#"{"transforms":[{"type":"Nope"}]}"#);
}

#[test]
fn oversized_outputs_are_errors_not_allocations() {
    // fit_output with a huge (valid) scale would need a gigantic canvas
    let mut t = Transform::affine((1e6, 1e6), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0);
    if let Transform::Affine { fit_output, .. } = &mut t {
        *fit_output = true;
    }
    let p = Pipeline::new(spec(vec![t]), Some(0)).unwrap();
    let e = p.apply(Input::image(img(64, 64, 3))).unwrap_err();
    assert!(e.to_string().contains("size limit"), "{e}");
    // ToGray with an absurd channel count
    let t = Transform::ToGray {
        num_output_channels: 1 << 40,
        method: augrs_core::ToGrayMethod::Average,
        p: 1.0,
    };
    let p = Pipeline::new(spec(vec![t]), Some(0)).unwrap();
    assert!(p.apply(Input::image(img(8, 8, 3))).is_err());
    // SmallestMaxSize on a long thin image: the long side would exceed the limit
    let t = Transform::SmallestMaxSize {
        max_size: vec![512],
        interpolation: Interp::Linear,
        mask_interpolation: Interp::Nearest,
        p: 1.0,
    };
    let p = Pipeline::new(spec(vec![t]), Some(0)).unwrap();
    assert!(p.apply(Input::image(img(1, 100_000, 1))).is_err());
}

#[test]
fn degenerate_images() {
    let p = Pipeline::new(spec(vec![Transform::hflip(1.0)]), Some(0)).unwrap();
    for (h, w, c) in [(0, 0, 3), (0, 5, 3), (5, 0, 3), (5, 5, 0)] {
        assert!(p.apply(Input::image(img(h, w, c))).is_err(), "{h}x{w}x{c}");
    }
    // every transform on a 1x1 image: Ok, or a clean error
    let all = vec![
        Transform::vflip(1.0),
        Transform::transpose(1.0),
        Transform::random_rotate90(1.0),
        Transform::random_resized_crop(4, 4, (0.08, 1.0)),
        Transform::rotate((-30.0, 30.0), 1.0),
        Transform::affine((0.5, 2.0), (-0.2, 0.2), (-30.0, 30.0), (-20.0, 20.0), 1.0),
        Transform::perspective((1.0, 1.0), 1.0),
        Transform::elastic(50.0, 1e-9, 1.0),
        Transform::color_jitter(0.5, 0.5, 0.5, 0.5, 1.0),
        Transform::coarse_dropout((3, 3), (50.0, 80.0), 1.0),
        Transform::gaussian_blur((3, 7), (0.1, 5.0), 1.0),
        Transform::normalize_imagenet(),
    ];
    for t in all {
        let name = t.name();
        let p = Pipeline::new(spec(vec![t]), Some(1)).unwrap();
        for (h, w) in [(1, 1), (1, 7), (7, 1)] {
            if let Ok(o) = p.apply(Input::image(img(h, w, 3))) {
                let (oh, ow, _) = o.image.dims();
                assert!(oh > 0 && ow > 0, "{name}");
            }
        }
    }
}

#[test]
fn wide_images_warp_exactly_beyond_60000_px() {
    // the warp/remap kernels clamped source coordinates to +-60000 px
    let w = 70_000;
    let a = Array3::from_shape_fn((2, w, 1), |(_, x, _)| (x % 251) as u8);
    let mut t = Transform::affine((1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0);
    if let Transform::Affine {
        translate_x,
        translate_px,
        interpolation,
        ..
    } = &mut t
    {
        *translate_x = (5.0, 5.0);
        *translate_px = true;
        *interpolation = Interp::Nearest;
    }
    let p = Pipeline::new(spec(vec![t]), Some(0)).unwrap();
    let o = p.apply(Input::image(Buf::U8(a.clone()))).unwrap();
    let Buf::U8(b) = &o.image else { panic!() };
    for x in [100, 60_500, 65_000, 69_990] {
        assert_eq!(b[[0, x, 0]], a[[0, x - 5, 0]], "x = {x}");
    }
    // displacement-field remap (elastic with alpha 0 is the identity)
    let p = Pipeline::new(spec(vec![Transform::elastic(0.0, 10.0, 1.0)]), Some(0)).unwrap();
    let o = p.apply(Input::image(Buf::U8(a.clone()))).unwrap();
    let Buf::U8(b) = &o.image else { panic!() };
    assert_eq!(b, &a);
}

#[test]
fn huge_elastic_displacements_finish_quickly() {
    // a huge displacement made the box/keypoint inverse search scan (and overflow) a window
    // of the displacement's size for every boundary point
    let s = spec(vec![Transform::elastic(1e7, 2.0, 1.0)])
        .with_bboxes(BboxParams::new(BboxFormat::PascalVoc))
        .with_keypoints(KeypointParams::new(KeypointFormat::Xy));
    let p = Pipeline::new(s, Some(3)).unwrap();
    let mut inp = Input::image(img(200, 200, 3));
    inp.bboxes = vec![[10.0, 10.0, 190.0, 190.0]; 8];
    inp.keypoints = vec![[50.0, 60.0, 0.0, 0.0]; 8];
    let t0 = Instant::now();
    p.apply(inp).unwrap();
    assert!(t0.elapsed() < Duration::from_secs(20), "{:?}", t0.elapsed());
}

#[test]
fn invalid_targets_raise_and_degenerate_ones_are_filtered() {
    let s = spec(vec![Transform::hflip(1.0)]).with_bboxes(BboxParams::new(BboxFormat::PascalVoc));
    let p = Pipeline::new(s, Some(0)).unwrap();
    let run = |boxes: Vec<[f64; 4]>| {
        let mut inp = Input::image(img(10, 20, 3));
        inp.bboxes = boxes;
        p.apply(inp)
    };
    assert!(run(vec![[f64::NAN, 0.0, 1.0, 1.0]]).is_err());
    assert!(run(vec![[0.0, 0.0, f64::INFINITY, 1.0]]).is_err());
    assert!(run(vec![[5.0, 0.0, 1.0, 1.0]]).is_err()); // inverted
    assert!(run(vec![]).unwrap().bboxes.is_empty());
    // zero-area boxes and boxes entirely outside the image are dropped; partly outside: clipped
    let o = run(vec![
        [3.0, 3.0, 3.0, 8.0],
        [30.0, 0.0, 40.0, 5.0],
        [-5.0, 2.0, 4.0, 6.0],
    ])
    .unwrap();
    assert_eq!(o.bbox_ids, vec![2]);
    assert_eq!(o.bboxes, vec![[16.0, 2.0, 20.0, 6.0]]);
    // keypoints: non-finite values raise; with remove_invisible, outside points are dropped
    // even when no geometric transform moves them
    let s = spec(vec![Transform::color_jitter(0.1, 0.1, 0.1, 0.0, 1.0)])
        .with_keypoints(KeypointParams::new(KeypointFormat::Xya));
    let p = Pipeline::new(s, Some(0)).unwrap();
    let mut inp = Input::image(img(10, 20, 3));
    inp.keypoints = vec![[1.0, 1.0, f64::NAN, 0.0]];
    assert!(p.apply(inp).is_err());
    let mut inp = Input::image(img(10, 20, 3));
    inp.keypoints = vec![[1.0, 1.0, 0.0, 0.0], [0.75, -0.4, 0.0, 0.0], [25.0, 3.0, 0.0, 0.0]];
    assert_eq!(p.apply(inp).unwrap().keypoint_ids, vec![0]);
}

#[test]
fn rng_handles_extreme_ranges() {
    let mut r = Rng::seed_from_u64(1);
    for _ in 0..1000 {
        let x = r.uniform(-f64::MAX, f64::MAX);
        assert!(x.is_finite());
        let i = r.int_inclusive(i64::MIN, i64::MAX);
        let _ = i;
        let j = r.int_inclusive(-5, 5);
        assert!((-5..=5).contains(&j));
    }
    // extreme but valid ranges in a transform do not produce NaN geometry
    let p = Pipeline::new(spec(vec![Transform::rotate((-f64::MAX, f64::MAX), 1.0)]), Some(0)).unwrap();
    p.apply(Input::image(img(9, 9, 3))).unwrap();
}

#[test]
fn hsv_shifts_beyond_the_pixel_range_saturate() {
    // fuzz crash: a value shift near f64::MAX became i32::MAX and `v + val_add` overflowed
    for (sat, val) in [
        (0.0, f64::MAX),
        (0.0, -f64::MAX),
        (f64::MAX, 0.0),
        (-1e300, 1e300),
        (1e10, -1e10),
    ] {
        let t = Transform::HueSaturationValue {
            hue_shift_limit: (0.0, 0.0),
            sat_shift_limit: (sat, sat),
            val_shift_limit: (val, val),
            p: 1.0,
        };
        let p = Pipeline::new(spec(vec![t]), Some(0)).unwrap();
        for c in [1, 3] {
            p.apply(Input::image(img(5, 7, c)))
                .unwrap_or_else(|e| panic!("sat {sat} val {val} c {c}: {e}"));
        }
    }
}

#[test]
fn color_jitter_huge_factors_saturate() {
    // fuzz crash: a saturation factor of 1e9 quantised to i32::MAX and `pixel * fq` overflowed
    for (b, c, s) in [
        (1.0, 1.0, 1e9),
        (1.0, 1.0, f64::MAX),
        (1e9, 1.0, 1.0),
        (1.0, f64::MAX, 1.0),
        (300.0, 300.0, 300.0),
    ] {
        let t = Transform::ColorJitter {
            brightness: (b, b),
            contrast: (c, c),
            saturation: (s, s),
            hue: (0.0, 0.0),
            p: 1.0,
        };
        let p = Pipeline::new(spec(vec![t]), Some(0)).unwrap();
        for ch in [1, 3, 4] {
            p.apply(Input::image(img(9, 7, ch)))
                .unwrap_or_else(|e| panic!("b {b} c {c} s {s} ch {ch}: {e}"));
        }
    }
}

#[test]
fn coarse_dropout_holes_larger_than_the_image() {
    let s = spec(vec![Transform::coarse_dropout((4, 4), (500.0, 900.0), 1.0)])
        .with_bboxes(BboxParams::new(BboxFormat::PascalVoc));
    let p = Pipeline::new(s, Some(0)).unwrap();
    let mut inp = Input::image(img(16, 16, 3));
    inp.bboxes = vec![[1.0, 1.0, 8.0, 8.0]];
    let o = p.apply(inp).unwrap();
    // the whole image is a hole: the box is fully covered and dropped
    assert!(o.bboxes.is_empty());
    let Buf::U8(a) = &o.image else { panic!() };
    assert!(a.iter().all(|&v| v == 0));
}

#[test]
fn perspective_fit_output_never_fails_for_valid_parameters() {
    // strong jitter can put the homography's horizon inside the image: an image corner then
    // maps to infinity, which used to be an error ("image corner maps to infinity")
    let mut t = Transform::perspective((0.05, 0.32), 1.0);
    if let Transform::Perspective {
        fit_output, keep_size, ..
    } = &mut t
    {
        *fit_output = true;
        *keep_size = false;
    }
    let s = spec(vec![t]).with_bboxes(BboxParams::new(BboxFormat::PascalVoc));
    let p = Pipeline::new(s, None).unwrap();
    for seed in 0..3000u64 {
        let mut inp = Input::image(img(12 + (seed % 9) as usize, 10 + (seed % 7) as usize, 3));
        inp.bboxes = vec![[1.0, 1.0, 6.0, 7.0]];
        let o = p.apply_with_seed(inp, seed, false).unwrap();
        let (h, w, _) = o.image.dims();
        assert!(h <= 200 && w <= 200, "{h}x{w}");
    }
}
