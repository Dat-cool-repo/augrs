//! Per-op timing on a synthetic 480x640 RGB image (single thread).
//! `cargo run --release -p augrs-core --example profile_ops`

use augrs_core::ops::{self, color, resize::Rect};
use augrs_core::{Affine2, BorderMode, Buf, Interp};
use ndarray::Array3;
use std::hint::black_box;
use std::time::Instant;

fn time<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    f();
    let mut best = f64::INFINITY;
    let mut total = 0.0;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64() * 1e6;
        best = best.min(dt);
        total += dt;
    }
    println!("{name:40} min {best:8.1} us   mean {:8.1} us", total / iters as f64);
}

fn main() {
    let img = Array3::from_shape_fn((480, 640, 3), |(y, x, c)| ((y * 7 + x * 13 + c * 61) % 251) as u8);
    let out512 = ops::resize_u8(
        &img,
        Rect {
            x0: 50,
            y0: 40,
            x1: 450,
            y1: 400,
        },
        512,
        512,
        Interp::Linear,
    );
    let n = 50;
    time("resize crop->512 linear", n, || {
        black_box(ops::resize_u8(
            &img,
            Rect {
                x0: 50,
                y0: 40,
                x1: 450,
                y1: 400,
            },
            512,
            512,
            Interp::Linear,
        ));
    });
    time("hflip 512", n, || {
        black_box(ops::hflip(&out512));
    });
    let m = Affine2::translate(-256.0, -256.0)
        .then(&Affine2::scale(1.05, 0.95))
        .then(&Affine2::rotate_deg(10.0))
        .then(&Affine2::translate(260.0, 250.0));
    let inv = m.inverse().unwrap();
    time("warp affine 512 linear", n, || {
        black_box(ops::warp_affine_u8(
            &out512,
            &inv,
            512,
            512,
            Interp::Linear,
            BorderMode::Constant,
            &[0.0],
        ));
    });
    time("warp affine 512 nearest (mask, c=1)", n, || {
        let mk = Array3::<u8>::zeros((512, 512, 1));
        black_box(ops::warp_affine(
            &mk,
            &inv,
            512,
            512,
            Interp::Nearest,
            BorderMode::Constant,
            &[0.0],
        ));
    });
    time("brightness", n, || {
        let mut a = out512.clone();
        color::brightness_u8(&mut a, 1.1);
        black_box(a);
    });
    time("contrast", n, || {
        let mut a = out512.clone();
        color::contrast_u8(&mut a, 1.1);
        black_box(a);
    });
    time("saturation", n, || {
        let mut a = out512.clone();
        color::saturation_u8(&mut a, 1.1);
        black_box(a);
    });
    time("hue", n, || {
        let mut a = out512.clone();
        color::hue_u8(&mut a, 0.03);
        black_box(a);
    });
    time("hsv round trip only (no edit)", n, || {
        let mut a = out512.clone();
        let e = augrs_core::ops::hsv::HsvEdit {
            hue_lut: None,
            sat_add: 0,
            val_add: 0,
            keep_gray_sat: true,
        };
        augrs_core::ops::hsv::hsv_edit_u8(&mut a, &e);
        black_box(a);
    });
    time("hue_saturation_value (h, s, v)", n, || {
        let mut a = out512.clone();
        let e = augrs_core::ops::hsv::HsvEdit {
            hue_lut: Some(augrs_core::ops::hsv::hue_lut(7.0)),
            sat_add: 10,
            val_add: -5,
            keep_gray_sat: true,
        };
        augrs_core::ops::hsv::hsv_edit_u8(&mut a, &e);
        black_box(a);
    });
    time("clone 512 (baseline for color ops)", n, || {
        black_box(out512.clone());
    });
    let k = ops::gaussian_kernel_1d(1.0, 5);
    time("gaussian blur k=5", n, || {
        black_box(ops::blur::gaussian_blur_u8(&out512, &k));
    });
    time("alloc + fill 3 MB (f32 512x512x3)", n, || {
        black_box(vec![1f32; 512 * 512 * 3]);
    });
    time("alloc + fill 768 KB (u8 512x512x3)", n, || {
        black_box(vec![1u8; 512 * 512 * 3]);
    });
    let b = Buf::U8(out512.clone());
    time("normalize", n, || {
        black_box(ops::normalize(&b, &[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225], 255.0).unwrap());
    });
}
