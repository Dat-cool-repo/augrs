# Why augrs exists

This document keeps the original motivation, research and scope for augrs. For usage, see the
[README](../README.md).

## The problem

[Albumentations](https://github.com/albumentations-team/albumentations) (MIT, about 15k stars)
became the standard image augmentation library for training vision models. Its GitHub repository
was **archived in June 2025**; the last MIT release on PyPI is `2.0.8` (2025-05-27). Its successor,
AlbumentationsX, is **AGPL-3.0 or commercial**, which many companies cannot use.

The alternatives leave gaps:

- **torchvision v2 transforms** have fewer operations, especially for keeping boxes, masks and
  keypoints consistent, and run in Python.
- **kornia-rs** has basic image operations but no high-level, multi-target augmentation pipeline.
- Rust ML frameworks such as [Burn](https://github.com/tracel-ai/burn) have no augmentation library
  (see [tracel-ai/burn#207](https://github.com/tracel-ai/burn/issues/207), open since 2023).

Python augmentation is also bound by the GIL: with Albumentations, the usual way to use more than
one core is a pool of worker processes (`DataLoader(num_workers=...)`), which costs memory,
start-up time and the pickling of every output array.

## Landscape (checked October 2026)

- **Albumentations** (MIT): archived (last push 2025-06-25); last PyPI release `2.0.8`.
- **AlbumentationsX** (successor): `AGPL-3.0-only` or commercial; still actively released.
- **[albu/sinter](https://github.com/albu/sinter)** (MIT, "compiler-based augmentation with
  operator fusion", by Albumentations authors): created 2026-01, described as a research project,
  very early. It claims bbox/mask/keypoint support and Python bindings, so it is the closest
  project to watch.
- **crates.io**: `albumrust` (MIT, pure Rust augmentation inspired by Albumentations, no Python
  bindings), `purpur` (stale since 2022), and Burn dataloader pipeline pieces
  (`bunsen-firehose-image` / `bimm-firehose-image`) without an Albumentations-style joint-target API.
  No Rust-backed, Albumentations-compatible package was found on PyPI.

Conclusion: a permissively licensed library with a Rust core, an Albumentations-like Python API,
joint boxes/masks/keypoints and GIL-free batches was still missing.

## Original scope

MVP:

1. Core operations in Rust on `u8`/`f32` HWC buffers: HorizontalFlip, VerticalFlip, RandomCrop,
   RandomResizedCrop, Resize, Rotate, Affine, ColorJitter, Normalize, GaussianBlur, CoarseDropout.
2. **Target sync**: one random draw applied consistently to the image, masks, bounding boxes
   (pascal_voc / coco / yolo) and keypoints.
3. `Compose` / `OneOf` / per-transform probability `p` with Albumentations semantics.
4. A Python API (`A.Compose([...], bbox_params=...)`) with numpy in and out, releasing the GIL.
5. Seeded determinism and replay (return the applied parameters).

Stretch goals:

- A batch API on a thread pool, and benchmarks against Albumentations and torchvision.
- A Burn-native crate, so Rust training does not go through Python.
- YAML/JSON serialisation compatible with Albumentations' format.
- A WASM demo page showing augmentations live.

All MVP items, the batch API, the benchmarks against Albumentations and the serialisation are
implemented; the Burn crate, torchvision benchmarks and the WASM demo are not (yet).

## Ground rules

- **Licensing hygiene.** No code from AGPL AlbumentationsX. augrs is written from the math, from
  the MIT-licensed Albumentations 2.0.8 behaviour (as observed through its public API and tests
  against it) and from OpenCV's documented semantics. See
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- **Parity with tolerances.** Pixel-for-pixel parity with OpenCV-based implementations is not
  always possible or desirable; every transform has a documented parity level (bit-exact, close
  or differs) and a test that pins it.
- **Determinism first.** The same seed gives the same output, regardless of the number of threads
  or the SIMD level of the CPU.
