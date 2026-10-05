//! Property tests: joint correctness of image, mask, boxes and keypoints.
#![allow(clippy::needless_range_loop, clippy::manual_clamp)]

use augrs_core::{
    BboxFormat, BboxMethod, BboxParams, BorderMode, Buf, DropoutBboxes, DropoutFill, DropoutKeypoints, Input, Interp,
    KeypointFormat, KeypointParams, PadPosition, Pipeline, PipelineSpec, Transform,
};
use ndarray::Array3;
use proptest::prelude::*;

fn pipe(ts: Vec<Transform>, fmt: BboxFormat, seed: u64) -> Pipeline {
    let mut kp = KeypointParams::new(KeypointFormat::Xya);
    kp.remove_invisible = false;
    let spec = PipelineSpec::new(ts)
        .with_bboxes(BboxParams::new(fmt))
        .with_keypoints(kp);
    Pipeline::new(spec, Some(seed)).unwrap()
}

fn test_image(h: usize, w: usize) -> Array3<u8> {
    Array3::from_shape_fn((h, w, 3), |(y, x, c)| ((y * 7 + x * 13 + c * 61) % 251) as u8)
}

fn u8_of(b: &Buf) -> &Array3<u8> {
    match b {
        Buf::U8(a) => a,
        _ => panic!("expected u8"),
    }
}

/// Bounding box `(x0, y0, x1, y1)` of the non-zero mask pixels (pixel extents).
fn mask_bbox(m: &Array3<u8>) -> Option<(f64, f64, f64, f64)> {
    let (h, w, _) = m.dim();
    let mut r: Option<(usize, usize, usize, usize)> = None;
    for y in 0..h {
        for x in 0..w {
            if m[[y, x, 0]] != 0 {
                r = Some(match r {
                    None => (x, y, x, y),
                    Some((a, b, c, d)) => (a.min(x), b.min(y), c.max(x), d.max(y)),
                });
            }
        }
    }
    r.map(|(a, b, c, d)| (a as f64, b as f64, c as f64 + 1.0, d as f64 + 1.0))
}

fn rect_mask(h: usize, w: usize, b: (usize, usize, usize, usize)) -> Array3<u8> {
    Array3::from_shape_fn((h, w, 1), |(y, x, _)| {
        if x >= b.0 && x < b.2 && y >= b.1 && y < b.3 {
            255
        } else {
            0
        }
    })
}

prop_compose! {
    fn img_and_box()(h in 8usize..64, w in 8usize..64)
        (h in Just(h), w in Just(w),
         x0 in 0..w - 2, y0 in 0..h - 2, bw in 1usize..64, bh in 1usize..64)
        -> (usize, usize, (usize, usize, usize, usize)) {
        let x1 = (x0 + bw).min(w);
        let y1 = (y0 + bh).min(h);
        (h, w, (x0, y0, x1.max(x0 + 1), y1.max(y0 + 1)))
    }
}

fn fmt_strategy() -> impl Strategy<Value = BboxFormat> {
    prop_oneof![
        Just(BboxFormat::PascalVoc),
        Just(BboxFormat::Coco),
        Just(BboxFormat::Yolo),
        Just(BboxFormat::Albumentations)
    ]
}

fn run(
    p: &Pipeline,
    img: Array3<u8>,
    mask: Option<Array3<u8>>,
    bbox: [f64; 4],
    kps: Vec<[f64; 4]>,
    seed: u64,
) -> augrs_core::Output {
    let inp = Input {
        image: Buf::U8(img),
        extra_images: vec![],
        masks: mask.into_iter().map(Buf::U8).collect(),
        bboxes: vec![bbox],
        keypoints: kps,
    };
    p.apply_with_seed(inp, seed, false).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    /// Flipping twice (H, V, or both) is the identity on every target, in every bbox format.
    #[test]
    fn double_flip_is_identity((h, w, b) in img_and_box(), fmt in fmt_strategy(),
                               kx in 0.0f64..1.0, ky in 0.0f64..1.0, ang in 0.0f64..360.0,
                               which in 0usize..3) {
        let flip = |p| match which { 0 => Transform::hflip(p), 1 => Transform::vflip(p), _ => Transform::compose(vec![Transform::hflip(p), Transform::vflip(p)]) };
        let p = pipe(vec![flip(1.0), flip(1.0)], fmt, 0);
        let img = test_image(h, w);
        let mask = rect_mask(h, w, b);
        let abs = [b.0 as f64, b.1 as f64, b.2 as f64, b.3 as f64];
        let bbox = fmt.from_abs(abs, h, w);
        let kp = [kx * w as f64, ky * h as f64, ang, 0.0];
        let out = run(&p, img.clone(), Some(mask.clone()), bbox, vec![kp], 1);
        prop_assert_eq!(u8_of(&out.image), &img);
        prop_assert_eq!(u8_of(&out.masks[0]), &mask);
        for i in 0..4 { prop_assert!((out.bboxes[0][i] - bbox[i]).abs() < 1e-9); }
        prop_assert!((out.keypoints[0][0] - kp[0]).abs() < 1e-9);
        prop_assert!((out.keypoints[0][1] - kp[1]).abs() < 1e-9);
        let da = (out.keypoints[0][2] - ang).rem_euclid(360.0);
        prop_assert!(da < 1e-6 || (360.0 - da) < 1e-6, "angle {} -> {}", ang, out.keypoints[0][2]);
    }

    /// A single flip moves a marked pixel exactly where the keypoint at its centre goes,
    /// and the mask's box equals the transformed box.
    #[test]
    fn flip_crop_pad_keep_pixels_and_points_together((h, w, b) in img_and_box(), px in 0.0f64..1.0, py in 0.0f64..1.0,
                                                     seed in any::<u64>()) {
        let ts = vec![
            Transform::hflip(0.5),
            Transform::vflip(0.5),
            Transform::PadIfNeeded { min_height: h + 7, min_width: w + 3, pad_height_divisor: None, pad_width_divisor: None,
                position: PadPosition::Random, border_mode: BorderMode::Constant, fill: vec![0.0], fill_mask: vec![0.0], p: 0.7 },
            Transform::random_rotate90(0.7),
            Transform::transpose(0.5),
            Transform::random_crop(h.min(6).max(4), w.min(6).max(4)),
        ];
        let p = pipe(ts, BboxFormat::PascalVoc, 0);
        // mark one pixel in a second mask, put a keypoint at its centre
        let (mx, my) = (((px * w as f64) as usize).min(w - 1), ((py * h as f64) as usize).min(h - 1));
        let mut dot = Array3::<u8>::zeros((h, w, 1));
        dot[[my, mx, 0]] = 1;
        let inp = Input {
            image: Buf::U8(test_image(h, w)),
            extra_images: vec![],
            masks: vec![Buf::U8(rect_mask(h, w, b)), Buf::U8(dot)],
            bboxes: vec![[b.0 as f64, b.1 as f64, b.2 as f64, b.3 as f64]],
            keypoints: vec![[mx as f64 + 0.5, my as f64 + 0.5, 0.0, 0.0]],
        };
        let out = p.apply_with_seed(inp, seed, false).unwrap();
        let (oh, ow, _) = out.image.dims();
        let kp = out.keypoints[0];
        let dotm = u8_of(&out.masks[1]);
        let inside = kp[0] >= 0.0 && kp[1] >= 0.0 && kp[0] < ow as f64 && kp[1] < oh as f64;
        if inside {
            prop_assert_eq!(dotm[[kp[1].floor() as usize, kp[0].floor() as usize, 0]], 1);
            prop_assert_eq!(dotm.iter().filter(|&&v| v == 1).count(), 1);
        } else {
            prop_assert_eq!(dotm.iter().filter(|&&v| v == 1).count(), 0);
        }
        // mask box == transformed bbox (exact for pixel permutations)
        match (mask_bbox(u8_of(&out.masks[0])), out.bboxes.first()) {
            (Some(mb), Some(bb)) => {
                prop_assert!((mb.0 - bb[0]).abs() < 1e-9 && (mb.1 - bb[1]).abs() < 1e-9
                    && (mb.2 - bb[2]).abs() < 1e-9 && (mb.3 - bb[3]).abs() < 1e-9, "mask {:?} box {:?}", mb, bb);
            }
            (None, None) => {}
            (m, bb) => prop_assert!(false, "mask {:?} vs box {:?}", m, bb),
        }
    }

    /// Under resizes / random-resized-crops, the mask box and the bbox agree within one output pixel.
    #[test]
    fn resize_crop_mask_box_consistent((h, w, b) in img_and_box(), oh in 5usize..80, ow in 5usize..80,
                                        seed in any::<u64>()) {
        let ts = vec![
            Transform::random_resized_crop(oh, ow, (0.3, 1.0)),
            Transform::hflip(0.5),
        ];
        let p = pipe(ts, BboxFormat::Coco, 0);
        let mask = rect_mask(h, w, b);
        let bbox = BboxFormat::Coco.from_abs([b.0 as f64, b.1 as f64, b.2 as f64, b.3 as f64], h, w);
        let out = run(&p, test_image(h, w), Some(mask), bbox, vec![], seed);
        if let (Some(mb), Some(bb)) = (mask_bbox(u8_of(&out.masks[0])), out.bboxes.first()) {
            let bb = BboxFormat::Coco.to_abs(*bb, out.image.dims().0, out.image.dims().1);
            let tol = 1.0 + 1e-9;
            prop_assert!((mb.0 - bb[0]).abs() <= tol && (mb.1 - bb[1]).abs() <= tol
                && (mb.2 - bb[2]).abs() <= tol && (mb.3 - bb[3]).abs() <= tol, "mask {:?} box {:?}", mb, bb);
        }
    }

    /// Under rotation/affine, every box corner (tracked as a keypoint) that is still inside
    /// the image lies inside the transformed (clipped) box, and every mask pixel of the
    /// object lies inside the box (largest_box method).
    #[test]
    fn rotation_box_contains_corners_and_mask((h, w, b) in img_and_box(), seed in any::<u64>(), kind in 0usize..6) {
        let t = match kind {
            0 => Transform::rotate((-180.0, 180.0), 1.0),
            1 => Transform::Affine {
                scale_x: (0.7, 1.4), scale_y: (0.7, 1.4), keep_ratio: false, balanced_scale: true,
                translate_x: (-0.2, 0.2), translate_y: (-0.2, 0.2), translate_px: false,
                rotate: (-60.0, 60.0), shear_x: (-20.0, 20.0), shear_y: (-20.0, 20.0),
                interpolation: Interp::Linear, mask_interpolation: Interp::Nearest, border_mode: BorderMode::Constant,
                fill: vec![0.0], fill_mask: vec![0.0], rotate_method: BboxMethod::LargestBox, fit_output: seed % 2 == 0, p: 1.0 },
            2 => Transform::ShiftScaleRotate {
                shift_limit_x: (-0.1, 0.1), shift_limit_y: (-0.1, 0.1), scale_limit: (0.8, 1.2),
                rotate_limit: (-90.0, 90.0), interpolation: Interp::Linear, mask_interpolation: Interp::Nearest,
                border_mode: BorderMode::Reflect101, fill: vec![0.0], fill_mask: vec![0.0],
                rotate_method: BboxMethod::LargestBox, p: 1.0 },
            3 => Transform::Perspective {
                scale: (0.05, 0.3), keep_size: seed % 2 == 0, fit_output: seed % 3 == 0, interpolation: Interp::Linear,
                mask_interpolation: Interp::Nearest, border_mode: BorderMode::Constant, fill: vec![0.0], fill_mask: vec![0.0], p: 1.0 },
            4 => Transform::elastic(40.0, 4.0, 1.0),
            _ => {
                let mut r = Transform::rotate((-180.0, 180.0), 1.0);
                if let Transform::Rotate { fit_output, .. } = &mut r { *fit_output = true; }
                r
            }
        };
        let p = pipe(vec![t], BboxFormat::PascalVoc, 0);
        let abs = [b.0 as f64, b.1 as f64, b.2 as f64, b.3 as f64];
        let corners = vec![
            [abs[0], abs[1], 0.0, 0.0], [abs[2], abs[1], 0.0, 0.0],
            [abs[0], abs[3], 0.0, 0.0], [abs[2], abs[3], 0.0, 0.0],
        ];
        let out = run(&p, test_image(h, w), Some(rect_mask(h, w, b)), abs, corners, seed);
        let (oh, ow, _) = out.image.dims();
        let eps = 1e-6;
        match out.bboxes.first() {
            Some(bb) => {
                for k in &out.keypoints {
                    let inside = k[0] >= 0.0 && k[1] >= 0.0 && k[0] <= ow as f64 && k[1] <= oh as f64;
                    if inside {
                        prop_assert!(k[0] >= bb[0] - eps && k[0] <= bb[2] + eps && k[1] >= bb[1] - eps && k[1] <= bb[3] + eps,
                            "corner {:?} outside box {:?}", k, bb);
                    }
                }
                // every object pixel (nearest-sampled mask) has its centre near the box
                // (nearest sampling may pick a source pixel up to half a pixel away; the
                // affine case can stretch that by its scale/shear). Reflect borders mirror
                // the object, so skip that case.
                let tol = match kind { 1 | 3 => 2.5, 4 => 1.5, _ => 1.0 };
                let m = u8_of(&out.masks[0]);
                for y in 0..oh { for x in 0..ow {
                    if m[[y, x, 0]] != 0 && kind != 2 {
                        let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
                        prop_assert!(cx >= bb[0] - tol && cx <= bb[2] + tol && cy >= bb[1] - tol && cy <= bb[3] + tol,
                            "mask pixel ({}, {}) outside box {:?}", x, y, bb);
                    }
                }}
            }
            None => {
                // box fully left the image: the object mask must be (nearly) gone too
                if kind != 2 {
                    let m = u8_of(&out.masks[0]);
                    let n = m.iter().filter(|&&v| v != 0).count();
                    prop_assert!(n <= ow.max(oh), "box dropped but {} mask pixels remain", n);
                }
            }
        }
    }

    /// Same seed => identical output; batch results do not depend on the thread count.
    #[test]
    fn deterministic_and_thread_independent(seed in any::<u64>()) {
        let ts = vec![
            Transform::random_resized_crop(24, 32, (0.2, 1.0)),
            Transform::hflip(0.5),
            Transform::rotate((-30.0, 30.0), 0.5),
            Transform::color_jitter(0.3, 0.3, 0.3, 0.1, 0.8),
            Transform::gaussian_blur((3, 5), (0.5, 1.5), 0.5),
            Transform::some_of(vec![Transform::perspective((0.05, 0.1), 1.0), Transform::elastic(20.0, 4.0, 1.0),
                Transform::coarse_dropout((1, 3), (0.1, 0.3), 1.0), Transform::hue_saturation_value(20.0, 30.0, 20.0, 1.0)], 2, 0.8),
            Transform::GaussNoise { std_range: (0.05, 0.1), mean_range: (0.0, 0.0), per_channel: true, noise_scale_factor: 0.5, p: 0.5 },
            Transform::normalize_imagenet(),
        ];
        let p1 = pipe(ts.clone(), BboxFormat::Yolo, seed);
        let p2 = pipe(ts, BboxFormat::Yolo, seed);
        let mk = |i: usize| Input {
            image: Buf::U8(test_image(40 + i, 50)),
            extra_images: vec![],
            masks: vec![],
            bboxes: vec![[0.5, 0.5, 0.4, 0.3]],
            keypoints: vec![],
        };
        for i in 0..3 {
            let a = p1.apply(mk(i)).unwrap();
            let b = p2.apply(mk(i)).unwrap();
            prop_assert_eq!(&a.image, &b.image);
            prop_assert_eq!(&a.bboxes, &b.bboxes);
        }
        let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let r1 = pool1.install(|| p1.apply_batch((0..6).map(mk).collect(), Some(seed), false));
        let r4 = pool4.install(|| p1.apply_batch((0..6).map(mk).collect(), Some(seed), false));
        for (a, b) in r1.iter().zip(r4.iter()) {
            let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
            prop_assert_eq!(&a.image, &b.image);
            prop_assert_eq!(&a.bboxes, &b.bboxes);
        }
    }
}

#[test]
fn visibility_filtering() {
    // 20x20 image, box covering x in [0, 10); crop x in [5, 15) keeps half of it.
    let spec = |min_vis: f64| {
        let mut bp = BboxParams::new(BboxFormat::PascalVoc);
        bp.min_visibility = min_vis;
        PipelineSpec::new(vec![
            Transform::Compose {
                transforms: vec![Transform::pad_if_needed(20, 20)],
                p: 1.0,
            },
            Transform::center_crop(20, 10),
        ])
        .with_bboxes(bp)
    };
    let inp = || Input {
        image: Buf::U8(test_image(20, 20)),
        extra_images: vec![],
        masks: vec![],
        bboxes: vec![[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]],
        keypoints: vec![],
    };
    let out = Pipeline::new(spec(0.0), Some(0)).unwrap().apply(inp()).unwrap();
    assert_eq!(out.bbox_ids, vec![0, 1]);
    assert_eq!(out.bboxes[0], [0.0, 0.0, 5.0, 10.0]);
    let out = Pipeline::new(spec(0.6), Some(0)).unwrap().apply(inp()).unwrap();
    assert_eq!(out.bbox_ids, vec![1]);
}

#[test]
fn keypoint_angles_follow_flips_and_rotation() {
    let mut kp = KeypointParams::new(KeypointFormat::Xyas);
    kp.remove_invisible = false;
    let p = Pipeline::new(
        PipelineSpec::new(vec![Transform::hflip(1.0)]).with_keypoints(kp.clone()),
        Some(0),
    )
    .unwrap();
    let mut inp = Input::image(Buf::U8(test_image(10, 20)));
    inp.keypoints = vec![[3.0, 4.0, 30.0, 2.0]];
    let out = p.apply(inp).unwrap();
    assert!((out.keypoints[0][0] - 17.0).abs() < 1e-9);
    assert!((out.keypoints[0][2] - 150.0).abs() < 1e-9); // pi - angle
    assert!((out.keypoints[0][3] - 2.0).abs() < 1e-9);

    // a 90-degree CCW rotation about the centre of a square image
    let p = Pipeline::new(
        PipelineSpec::new(vec![Transform::rotate((90.0, 90.0), 1.0)]).with_keypoints(kp),
        Some(0),
    )
    .unwrap();
    let mut inp = Input::image(Buf::U8(test_image(10, 10)));
    inp.keypoints = vec![[7.5, 5.0, 0.0, 1.0]];
    let out = p.apply(inp).unwrap();
    assert!((out.keypoints[0][0] - 5.0).abs() < 1e-9 && (out.keypoints[0][1] - 2.5).abs() < 1e-9);
    assert!((out.keypoints[0][2] - 270.0).abs() < 1e-9); // pointing up on screen
}

#[test]
fn rotate_90_moves_pixels_like_points() {
    // On a square image a 90 degree rotation is an exact pixel permutation.
    let (h, w) = (12, 12);
    let p = pipe(vec![Transform::rotate((90.0, 90.0), 1.0)], BboxFormat::PascalVoc, 0);
    let mut dot = Array3::<u8>::zeros((h, w, 1));
    dot[[2, 9, 0]] = 1;
    let inp = Input {
        image: Buf::U8(test_image(h, w)),
        extra_images: vec![],
        masks: vec![Buf::U8(dot)],
        bboxes: vec![[9.0, 2.0, 10.0, 3.0]],
        keypoints: vec![[9.5, 2.5, 0.0, 0.0]],
    };
    let out = p.apply(inp).unwrap();
    let k = out.keypoints[0];
    let m = u8_of(&out.masks[0]);
    assert_eq!(m[[k[1].floor() as usize, k[0].floor() as usize, 0]], 1);
    let bb = out.bboxes[0];
    assert!((bb[0] - k[0].floor()).abs() < 1e-9 && (bb[1] - k[1].floor()).abs() < 1e-9);
    assert!((bb[2] - bb[0] - 1.0).abs() < 1e-9 && (bb[3] - bb[1] - 1.0).abs() < 1e-9);
}

#[test]
fn ellipse_method_is_tighter() {
    let mk = |method| {
        let mut t = Transform::rotate((45.0, 45.0), 1.0);
        if let Transform::Rotate { rotate_method, .. } = &mut t {
            *rotate_method = method;
        }
        let p = Pipeline::new(
            PipelineSpec::new(vec![t]).with_bboxes(BboxParams::new(BboxFormat::PascalVoc)),
            Some(0),
        )
        .unwrap();
        let mut inp = Input::image(Buf::U8(test_image(100, 100)));
        inp.bboxes = vec![[40.0, 40.0, 60.0, 60.0]];
        p.apply(inp).unwrap().bboxes[0]
    };
    let lb = mk(BboxMethod::LargestBox);
    let el = mk(BboxMethod::Ellipse);
    let s2 = 2f64.sqrt();
    assert!((lb[2] - lb[0] - 20.0 * s2).abs() < 1e-6);
    assert!((el[2] - el[0] - 20.0).abs() < 1e-6);
}

#[test]
fn errors_are_reported() {
    let p = Pipeline::new(PipelineSpec::new(vec![Transform::center_crop(50, 50)]), Some(0)).unwrap();
    assert!(p.apply(Input::image(Buf::U8(test_image(10, 10)))).is_err());
    let bad = PipelineSpec::new(vec![Transform::HorizontalFlip { p: 2.0 }]);
    assert!(Pipeline::new(bad, None).is_err());
    let p = Pipeline::new(PipelineSpec::new(vec![Transform::hflip(1.0)]), Some(0)).unwrap();
    let mut inp = Input::image(Buf::U8(test_image(10, 10)));
    inp.bboxes = vec![[0.0, 0.0, 1.0, 1.0]];
    assert!(p.apply(inp).is_err(), "bboxes without bbox_params must error");
}

#[test]
fn json_spec_roundtrip() {
    let json = r#"{"transforms": [
        {"type": "RandomResizedCrop", "height": 64, "width": 64, "scale": [0.5, 1.0]},
        {"type": "OneOf", "transforms": [{"type": "HorizontalFlip"}, {"type": "VerticalFlip", "p": 0.2}], "p": 0.9},
        {"type": "Affine", "rotate": [-10, 10], "border_mode": "reflect101", "interpolation": "nearest"},
        {"type": "Normalize"}
    ], "bbox_params": {"format": "coco", "min_visibility": 0.1}}"#;
    let p = Pipeline::from_json(json, Some(3)).unwrap();
    let p2 = Pipeline::from_json(&p.to_json(), Some(3)).unwrap();
    assert_eq!(p.spec(), p2.spec());
    let out = p.apply(Input::image(Buf::U8(test_image(100, 120)))).unwrap();
    assert_eq!(out.image.dims(), (64, 64, 3));
    assert!(matches!(out.image, Buf::F32(_)));
}

// ---- new transforms ----------------------------------------------------------

fn dropout(fill_mask: Option<f64>, bbox_handling: DropoutBboxes, kps: DropoutKeypoints) -> Transform {
    Transform::CoarseDropout {
        num_holes_range: (1, 6),
        hole_height_range: (0.05, 0.4),
        hole_width_range: (0.05, 0.4),
        fill: DropoutFill::Value(vec![7.0]),
        fill_mask: fill_mask.map(|v| vec![v]),
        bbox_handling,
        keypoint_handling: kps,
        p: 1.0,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    /// CoarseDropout: masks are filled exactly where the image is, keypoints inside holes
    /// are removed, and the (shrunk) box still contains every uncovered object pixel.
    #[test]
    fn coarse_dropout_targets_consistent((h, w, b) in img_and_box(), seed in any::<u64>(),
                                         kx in 0.0f64..1.0, ky in 0.0f64..1.0) {
        let p = pipe(vec![dropout(Some(1.0), DropoutBboxes::Shrink, DropoutKeypoints::Remove)], BboxFormat::PascalVoc, 0);
        // image is never 7 except in holes
        let img = test_image(h, w).mapv(|v| if v == 7 { 8 } else { v });
        let holes_probe = Array3::<u8>::zeros((h, w, 1));
        let obj = rect_mask(h, w, b);
        let kp = [kx * w as f64, ky * h as f64, 0.0, 0.0];
        let inp = Input {
            image: Buf::U8(img),
            extra_images: vec![],
            masks: vec![Buf::U8(holes_probe), Buf::U8(obj.clone())],
            bboxes: vec![[b.0 as f64, b.1 as f64, b.2 as f64, b.3 as f64]],
            keypoints: vec![kp],
        };
        let out = p.apply_with_seed(inp, seed, false).unwrap();
        let im = u8_of(&out.image);
        let holes = u8_of(&out.masks[0]);
        let objm = u8_of(&out.masks[1]);
        for y in 0..h { for x in 0..w {
            let in_hole = holes[[y, x, 0]] == 1;
            prop_assert_eq!(in_hole, im[[y, x, 0]] == 7 && im[[y, x, 1]] == 7 && im[[y, x, 2]] == 7);
            // fill_mask 1 also overwrote the object mask inside holes
            if in_hole { prop_assert_eq!(objm[[y, x, 0]], 1); }
        }}
        let kp_hole = holes[[((ky * h as f64) as usize).min(h - 1), ((kx * w as f64) as usize).min(w - 1), 0]] == 1;
        prop_assert_eq!(out.keypoints.is_empty(), kp_hole);
        // uncovered object pixels: inside the original box and not in a hole
        let mut uncovered = vec![];
        for y in b.1..b.3 { for x in b.0..b.2 { if holes[[y, x, 0]] == 0 { uncovered.push((x, y)); } } }
        match out.bboxes.first() {
            None => prop_assert!(uncovered.is_empty()),
            Some(bb) => {
                prop_assert!(!uncovered.is_empty());
                for (x, y) in uncovered {
                    prop_assert!(x as f64 >= bb[0] && x as f64 + 1.0 <= bb[2] && y as f64 >= bb[1] && y as f64 + 1.0 <= bb[3]);
                }
            }
        }
    }

    /// Extra images go through exactly the same transforms as the image.
    #[test]
    fn extra_images_follow_the_image(seed in any::<u64>(), h in 16usize..48, w in 16usize..48) {
        let ts = vec![
            Transform::random_resized_crop(24, 28, (0.3, 1.0)),
            Transform::random_rotate90(0.5),
            Transform::rotate((-30.0, 30.0), 0.5),
            Transform::perspective((0.05, 0.1), 0.3),
            Transform::elastic(10.0, 3.0, 0.3),
            Transform::color_jitter(0.3, 0.3, 0.3, 0.1, 0.8),
            Transform::hue_saturation_value(20.0, 30.0, 20.0, 0.5),
            Transform::brightness_contrast(0.2, 0.2, 0.5),
            Transform::GaussNoise { std_range: (0.05, 0.1), mean_range: (0.0, 0.0), per_channel: true, noise_scale_factor: 1.0, p: 0.5 },
            Transform::CoarseDropout { num_holes_range: (1, 3), hole_height_range: (0.1, 0.3), hole_width_range: (0.1, 0.3),
                fill: DropoutFill::Mode("random".into()), fill_mask: None, bbox_handling: DropoutBboxes::Shrink,
                keypoint_handling: DropoutKeypoints::Auto, p: 0.5 },
            Transform::gaussian_blur((3, 5), (0.5, 1.5), 0.5),
        ];
        let p = Pipeline::new(PipelineSpec::new(ts), Some(seed)).unwrap();
        let img = test_image(h, w);
        let inp = Input { image: Buf::U8(img.clone()), extra_images: vec![Buf::U8(img)], masks: vec![], bboxes: vec![], keypoints: vec![] };
        let out = p.apply_with_seed(inp, seed, false).unwrap();
        prop_assert_eq!(&out.image, &out.extra_images[0]);
    }

    /// fit_output keeps every corner of the input in view; crop_border leaves no border pixels.
    #[test]
    fn fit_output_and_crop_border(angle in -180.0f64..180.0, h in 10usize..60, w in 10usize..60) {
        let mut t = Transform::rotate((angle, angle), 1.0);
        if let Transform::Rotate { fit_output, .. } = &mut t { *fit_output = true; }
        let p = pipe(vec![t], BboxFormat::PascalVoc, 0);
        let corners: Vec<[f64; 4]> = [(0.0, 0.0), (w as f64, 0.0), (0.0, h as f64), (w as f64, h as f64)]
            .iter().map(|&(x, y)| [x, y, 0.0, 0.0]).collect();
        let out = run(&p, test_image(h, w), None, [0.0, 0.0, w as f64, h as f64], corners, 0);
        let (oh, ow, _) = out.image.dims();
        for k in &out.keypoints {
            prop_assert!(k[0] > -1e-6 && k[1] > -1e-6 && k[0] < ow as f64 + 1e-6 && k[1] < oh as f64 + 1e-6, "{:?} {}x{}", k, oh, ow);
        }
        let bb = out.bboxes[0];
        // whole-pixel bounds: at most one pixel of slack on each side
        prop_assert!(bb[0] <= 1.0 && bb[1] <= 1.0 && (bb[2] - ow as f64).abs() <= 1.0 && (bb[3] - oh as f64).abs() <= 1.0);

        let mut t = Transform::rotate((angle, angle), 1.0);
        if let Transform::Rotate { crop_border, interpolation, .. } = &mut t { *crop_border = true; *interpolation = Interp::Nearest; }
        let p = Pipeline::new(PipelineSpec::new(vec![t]), Some(0)).unwrap();
        let white = Array3::from_elem((h, w, 1), 255u8);
        let out = p.apply(Input::image(Buf::U8(white))).unwrap();
        let o = u8_of(&out.image);
        let (oh, ow, _) = o.dim();
        prop_assert!(oh <= h && ow <= w);
        let zeros = o.iter().filter(|&&v| v == 0).count();
        prop_assert!(zeros <= 2, "{} border pixels in {}x{}", zeros, oh, ow);
    }
}

#[test]
fn some_of_picks_n_distinct_in_order() {
    let ts = vec![Transform::some_of(
        vec![Transform::hflip(1.0), Transform::vflip(1.0), Transform::transpose(1.0)],
        2,
        1.0,
    )];
    let p = Pipeline::new(PipelineSpec::new(ts), Some(1)).unwrap();
    let mut seen = std::collections::HashSet::new();
    for seed in 0..64 {
        let out = p
            .apply_with_seed(Input::image(Buf::U8(test_image(6, 9))), seed, true)
            .unwrap();
        let applied = out.applied.unwrap();
        let idx: Vec<usize> = serde_json::from_value(applied[0].params["indices"].clone()).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(idx[0] < idx[1]);
        let names: Vec<&str> = applied[1..].iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names.len(), 2);
        seen.insert(idx);
    }
    assert_eq!(seen.len(), 3, "all 3 pairs should occur");
    // Sequential with p = 1 applies everything
    let p = Pipeline::new(
        PipelineSpec::new(vec![Transform::sequential(
            vec![Transform::hflip(1.0), Transform::hflip(1.0)],
            1.0,
        )]),
        Some(0),
    )
    .unwrap();
    let img = test_image(5, 7);
    let out = p.apply(Input::image(Buf::U8(img.clone()))).unwrap();
    assert_eq!(u8_of(&out.image), &img);
}

#[test]
fn soft_masks_use_linear_interpolation() {
    let mut t = Transform::resize(20, 20);
    if let Transform::Resize { mask_interpolation, .. } = &mut t {
        *mask_interpolation = Interp::Linear;
    }
    let p = Pipeline::new(PipelineSpec::new(vec![t]), Some(0)).unwrap();
    let m = Array3::from_shape_fn((10, 10, 1), |(_, x, _)| if x < 5 { 0.0f32 } else { 1.0 });
    let mut inp = Input::image(Buf::U8(test_image(10, 10)));
    inp.masks = vec![Buf::F32(m)];
    let out = p.apply(inp).unwrap();
    let Buf::F32(o) = &out.masks[0] else { panic!() };
    assert!(
        o.iter().any(|&v| v > 0.1 && v < 0.9),
        "linear mask has intermediate values"
    );
    // default stays nearest
    let p = Pipeline::new(PipelineSpec::new(vec![Transform::resize(20, 20)]), Some(0)).unwrap();
    let mut inp = Input::image(Buf::U8(test_image(10, 10)));
    inp.masks = vec![Buf::F32(Array3::from_shape_fn((10, 10, 1), |(_, x, _)| {
        if x < 5 { 0.0f32 } else { 1.0 }
    }))];
    let out = p.apply(inp).unwrap();
    let Buf::F32(o) = &out.masks[0] else { panic!() };
    assert!(o.iter().all(|&v| v == 0.0 || v == 1.0));
}

#[test]
fn perspective_identity_like_and_keypoint_projection() {
    // scale 0 -> no jitter -> identity map (keep_size)
    let p = pipe(vec![Transform::perspective((0.0, 0.0), 1.0)], BboxFormat::PascalVoc, 0);
    let img = test_image(30, 40);
    let out = run(
        &p,
        img.clone(),
        None,
        [3.0, 4.0, 20.0, 25.0],
        vec![[10.5, 7.5, 30.0, 0.0]],
        5,
    );
    assert_eq!(u8_of(&out.image), &img);
    for (a, b) in out.bboxes[0].iter().zip([3.0, 4.0, 20.0, 25.0]) {
        assert!((a - b).abs() < 1e-6);
    }
    assert!((out.keypoints[0][0] - 10.5).abs() < 1e-6 && (out.keypoints[0][2] - 30.0).abs() < 1e-6);
}

#[test]
fn json_spec_accepts_new_transforms() {
    let json = r#"{"transforms": [
        {"type": "SomeOf", "n": 2, "transforms": [{"type": "Transpose"}, {"type": "RandomRotate90"}, {"type": "ToGray", "method": "max"}]},
        {"type": "Sequential", "p": 1.0, "transforms": [{"type": "CLAHE", "p": 1.0}, {"type": "RandomGamma"}]},
        {"type": "CoarseDropout", "fill": "random_uniform", "fill_mask": 0, "p": 1.0},
        {"type": "CoarseDropout", "fill": [1, 2, 3], "hole_height_range": [4, 8], "hole_width_range": [4, 8]},
        {"type": "Perspective", "fit_output": true, "fill": 5},
        {"type": "ElasticTransform", "alpha": 30, "sigma": 5, "approximate": true},
        {"type": "RandomBrightnessContrast", "brightness_by_max": false, "ensure_safe_range": true},
        {"type": "HueSaturationValue", "p": 1.0},
        {"type": "GaussNoise", "per_channel": false, "noise_scale_factor": 0.25},
        {"type": "Normalize", "normalization": "image_per_channel"}
    ]}"#;
    let p = Pipeline::from_json(json, Some(3)).unwrap();
    let p2 = Pipeline::from_json(&p.to_json(), Some(3)).unwrap();
    assert_eq!(p.spec(), p2.spec());
    for seed in 0..8 {
        let a = p
            .apply_with_seed(Input::image(Buf::U8(test_image(48, 64))), seed, false)
            .unwrap();
        let b = p2
            .apply_with_seed(Input::image(Buf::U8(test_image(48, 64))), seed, false)
            .unwrap();
        assert_eq!(a.image, b.image);
        assert!(matches!(a.image, Buf::F32(_)));
    }
    assert!(
        Pipeline::from_json(
            r#"{"transforms": [{"type": "CoarseDropout", "fill": "inpaint_telea"}]}"#,
            None
        )
        .is_err()
    );
}
