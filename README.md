# augrs

[![CI](https://github.com/Dat-cool-repo/augrs/actions/workflows/ci.yml/badge.svg)](https://github.com/Dat-cool-repo/augrs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Fast, GIL-free image augmentation with a Rust core and an Albumentations-compatible Python API.**

augrs transforms images, masks, bounding boxes and keypoints together with a single random draw,
releases the GIL while it works, and can augment a whole batch on its own thread pool. Its Python
API follows Albumentations 2.0.8, so most pipelines port by changing the import, and it reads and
writes Albumentations config files. It is dual-licensed under MIT or Apache-2.0.

```python
import augrs as A   # instead of: import albumentations as A

t = A.Compose([A.RandomResizedCrop(size=(512, 512)), A.HorizontalFlip(), A.ColorJitter(0.2, 0.2, 0.2, 0.05)],
              bbox_params=A.BboxParams(format="coco", label_fields=["labels"]), seed=0)
out = t(image=img, mask=mask, bboxes=boxes, labels=labels)
```

> **Status:** early (0.1.0, alpha). `pip install augrs-py`, then `import augrs` (see [Install](#install)).

## Why

- **Albumentations is archived.** The MIT-licensed library most vision projects use was archived
  in June 2025; its last release is 2.0.8.
- **Its successor is AGPL.** AlbumentationsX is AGPL-3.0 or commercial, which many companies
  cannot ship. augrs is MIT OR Apache-2.0 and contains no AlbumentationsX code.
- **Python augmentation is GIL-bound.** Using more than one core with Albumentations means worker
  processes, which cost memory and the pickling of every output. augrs runs in Rust with the GIL
  released, so Python threads or `augment_batch(num_threads=...)` scale without extra processes.

More background, including the survey of alternatives, is in [docs/MOTIVATION.md](docs/MOTIVATION.md).

## Features

- **26 transforms plus 4 composition types** (`Compose`, `OneOf`, `SomeOf`, `Sequential`) with
  Albumentations names, arguments, defaults and per-transform probability `p`.
- **Joint targets.** Every geometric transform is a single point map (affine, homography or
  displacement field) applied consistently to images, masks, bounding boxes (`pascal_voc`, `coco`,
  `yolo`, `albumentations`) and keypoints (`xy`, `yx`, `xya`, `xys`, `xyas`, `xysa`), with
  Albumentations' filtering options (`min_area`, `min_visibility`, `min_width`, `min_height`,
  `max_accept_ratio`, `remove_invisible`) and label fields.
- **`additional_targets`:** extra images, masks, box sets and keypoint sets get the same geometry
  and the same sampled colour parameters.
- **Seeded determinism.** The same seed gives the same output, independent of the thread count
  and of the CPU's SIMD support. Single draws can be replayed, and applied parameters returned.
- **GIL-free batches.** `augment_batch(images, ..., num_threads=4)` runs on a rayon thread pool;
  single calls release the GIL too.
- **Fast kernels.** Hand-written AVX2/FMA kernels with runtime detection and a portable scalar
  fallback (bit-identical results) for warps, resizes, 8-bit HSV, saturation and normalisation;
  mimalloc in the Python extension.
- **Albumentations configs.** `augrs.load` reads JSON/YAML files written by Albumentations'
  `A.save` / `A.to_dict`; `augrs.save` writes files that Albumentations can load.
- **Tested parity.** A pytest suite compares augrs against Albumentations 2.0.8 and OpenCV
  transform by transform, and Rust property tests check that targets stay consistent.

## Supported transforms

Parity is measured against Albumentations 2.0.8 (with OpenCV) **given the same sampled
parameters**, since augrs uses its own random number generator: random transforms produce the same
*distribution* of results, not the same individual samples.

- **Bit-exact:** identical output for the same parameters.
- **Close:** small numeric differences (bounds below, all pinned by tests).
- **Differs:** a deliberate, documented difference (see
  [Differences from Albumentations](#differences-from-albumentations)).

### Geometric

| Transform | Parity | Notes |
|---|---|---|
| `HorizontalFlip`, `VerticalFlip` | Bit-exact | Image, mask, boxes in all 4 formats, labels, keypoints |
| `Transpose` | Bit-exact | Same targets as flips |
| `RandomRotate90` | Bit-exact | Every factor, vs `np.rot90` and Albumentations' box/keypoint functions |
| `CenterCrop` | Bit-exact | Including `pad_if_needed`<sup>3</sup> |
| `RandomCrop` | Bit-exact | Given the same window (image, mask, boxes in all 4 formats, keypoints), including `pad_if_needed`<sup>3</sup> |
| `PadIfNeeded` | Bit-exact | `min_height`/`min_width` and `pad_*_divisor`, every position and border mode<sup>3</sup> |
| `Resize` | Close | Linear: max diff 1 level, on < 1% of pixels<sup>2</sup> (tested sizes of 16 px and up). Nearest: `INTER_NEAREST_EXACT`, identical except where an output pixel centre lies exactly halfway between two input pixels (OpenCV breaks such ties either way). Cubic/area/lanczos: antialiased `fast_image_resize` kernels |
| `RandomResizedCrop` | Close | Crop + `Resize` given the window; window distribution (area, aspect) tested against Albumentations' sampler |
| `LongestMaxSize`, `SmallestMaxSize` | Close | As `Resize`, same output size (Python rounding); a list of sizes picks one at random |
| `Rotate` | Close | vs `cv2.warpAffine` with the same matrix: mean diff < 0.6 (OpenCV quantises to 1/32 px); `crop_border`, `fit_output` |
| `Affine` | Close | As `Rotate`, with fixed parameters vs Albumentations (scale, rotate, shear, whole-pixel translation); `fit_output` canvas within 1 px; `balanced_scale` |
| `ShiftScaleRotate` | Close | As `Affine`, with fixed parameters vs Albumentations |
| `Perspective` | Differs | Warp matches `cv2.warpPerspective`; no extra zoom with `keep_size=True` |
| `ElasticTransform` | Differs | Coarse-grid noise for `sigma > 8`; exact keypoint inverse |

### Photometric and other

| Transform | Parity | Notes |
|---|---|---|
| `HueSaturationValue` | Bit-exact | All 16.7M RGB colours through OpenCV's 8-bit HSV round trip<sup>1</sup> |
| `ColorJitter` | Close | Hue bit-exact<sup>1</sup>; brightness/contrast/saturation within 1 level |
| `RandomBrightnessContrast` | Bit-exact | uint8; float32 within 1e-6. `brightness_by_max`, `ensure_safe_range` |
| `RandomGamma` | Bit-exact | uint8 |
| `ToGray` | Bit-exact | `weighted_average`, `average`, `max`, `desaturation`; 1 or 3 output channels |
| `CLAHE` | Close | Gray images bit-exact<sup>2</sup>; RGB mean diff < 1 level (float Lab conversion) |
| `GaussianBlur` | Close | Within 1 level of OpenCV |
| `Normalize` | Close | Within 2e-4; `standard`, `image`, `image_per_channel`, `min_max`, `min_max_per_channel` |
| `GaussNoise` | Differs | Same distribution parameters; different sampler (see below) |
| `CoarseDropout` | Bit-exact | Given the same holes: image/mask fill, box (`shrink`) and keypoint handling |

<sup>1</sup> Bit-exact against OpenCV builds that fuse multiply-adds (the Linux x86-64 wheels).
OpenCV's Windows (MSVC) wheels do not, and a few values (< 0.1%, the tested bound) differ by one
level there.

<sup>2</sup> Against OpenCV's x86-64 builds. OpenCV's arm64 builds (macOS on Apple silicon, Linux
aarch64) use different kernels for some 8-bit ops, so the reference itself moves: linear resize
differs by one level on up to ~22% of pixels for some sizes (two levels on some single-channel
upscales), gray CLAHE on < 0.5% of pixels, and the HSV round trip<sup>1</sup> on < 0.3% of
values (0.02-0.07% measured on Linux aarch64). augrs produces the same output on x86 and arm; the
tests allow these differences only on arm builds of OpenCV.

<sup>3</sup> With reflecting border modes (`reflect`, `reflect101`), Albumentations 2.0.8 also adds
mirrored copies of boxes and keypoints that fall in the padded area; augrs keeps one box per
object. Images and masks are identical.

### Compositions

`Compose` (nestable), `OneOf`, `SomeOf` (Albumentations 2.x semantics: uniform choice of `n`,
original order, each child keeps its `p`) and `Sequential`, all with `p`.

Transforms not listed here, and a few options (`CoarseDropout(fill="inpaint_*")`,
`ToGray(method="from_lab" | "pca")`, `area_for_downscale`, `max_size_hw`), are not supported yet;
loading a config that uses them raises `NotImplementedError` naming the transform or argument.

## Install

```bash
pip install augrs-py      # the import name is `augrs`
```

The PyPI name is `augrs-py` because `augrs` was already taken by an unrelated package. Prebuilt
abi3 wheels (CPython 3.9+) cover Linux x86_64 / aarch64, macOS x86_64 / arm64 and Windows x86_64.

To build from source you need a Rust toolchain (1.87 or newer, via [rustup](https://rustup.rs))
and Python 3.9 or newer. pip builds the extension with [maturin](https://www.maturin.rs) in
release mode (a few minutes the first time).

```bash
git clone https://github.com/Dat-cool-repo/augrs
cd augrs
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install .            # or, without cloning: pip install git+https://github.com/Dat-cool-repo/augrs

# for development: an editable install, or an abi3 wheel (one wheel for every Python >= 3.9)
pip install maturin
maturin develop --release
maturin build --release --out dist && pip install dist/augrs_py-*.whl
```

`pyyaml` is needed only for YAML configs (`pip install pyyaml`). The Rust core is the
`augrs-core` crate in this workspace (also not on crates.io yet):

```toml
[dependencies]
augrs-core = { git = "https://github.com/Dat-cool-repo/augrs" }
ndarray = "0.17"
```

## Python quick start

### Detection and segmentation

```python
import numpy as np
import augrs as A

t = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.25, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), p=0.5),
        A.OneOf([A.HueSaturationValue(), A.RandomBrightnessContrast()], p=0.8),
        A.CoarseDropout(fill_mask=0, p=0.3),
        A.Normalize(),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.1),
    seed=0,
)

image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)   # HWC, uint8 or float32
mask = np.zeros((480, 640), dtype=np.uint8)
mask[100:200, 150:300] = 1
boxes = [(150, 100, 150, 100), (20, 30, 60, 80)]                     # coco: x, y, w, h
labels = ["car", "person"]

out = t(image=image, mask=mask, bboxes=boxes, labels=labels)
out["image"]    # (512, 512, 3) float32, normalised
out["mask"]     # (512, 512) uint8, same geometry as the image
out["bboxes"], out["labels"]   # transformed boxes; labels of dropped boxes are dropped too
```

Extra images, masks, box sets and keypoint sets go through `additional_targets`
(e.g. `additional_targets={"depth": "image", "mask2": "mask"}`), and keypoints through
`keypoint_params=A.KeypointParams(format="xy", label_fields=["kp_labels"])`.

### Batches without the GIL

```python
images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(32)]
outs = t.augment_batch(
    images,
    bboxes=[boxes] * 32,          # every other argument: one entry per image
    labels=[labels] * 32,
    seed=1,
    num_threads=4,
)
batch = np.stack([o["image"] for o in outs])   # (32, 512, 512, 3)
```

Sample `i` uses a seed derived from `(seed, i)`, so the result does not depend on `num_threads`.
Calling `t(...)` from several Python threads also scales, since each call releases the GIL.

### Albumentations configs

```python
import augrs

t = augrs.load("albumentations_pipeline.yaml")   # written by albumentations' A.save (JSON or YAML)
augrs.save(t, "pipeline.json")                    # loadable by augrs and by albumentations' A.load

# or convert an in-memory Albumentations pipeline
import albumentations
t = augrs.from_dict(albumentations.to_dict(albu_pipeline))
```

### PyTorch DataLoader

[`examples/torch_dataloader.py`](examples/torch_dataloader.py) augments whole batches in
`collate_fn`, so no worker processes are needed:

```python
class AugrsCollate:
    def __init__(self, transform, num_threads=4):
        self.t, self.num_threads = transform, num_threads

    def __call__(self, batch):
        imgs, boxes, labels = zip(*batch)
        outs = self.t.augment_batch(list(imgs), bboxes=list(boxes), labels=list(labels),
                                    num_threads=self.num_threads)
        x = torch.from_numpy(np.stack([o["image"] for o in outs])).permute(0, 3, 1, 2)
        return x, [{"boxes": torch.as_tensor(np.asarray(o["bboxes"]).reshape(-1, 4)),
                    "labels": torch.as_tensor(o["labels"])} for o in outs]

loader = DataLoader(dataset, batch_size=32, num_workers=0, collate_fn=AugrsCollate(t))
```

Run it with `python examples/torch_dataloader.py` (synthetic images) or with `--data data` after
downloading the COCO subset; it prints a throughput comparison with Albumentations.

## Rust quick start

```rust
use augrs_core::{BboxFormat, BboxParams, Buf, Input, Pipeline, PipelineSpec, Transform};
use ndarray::Array3;

fn main() -> augrs_core::Result<()> {
    let spec = PipelineSpec::new(vec![
        Transform::random_resized_crop(512, 512, (0.25, 1.0)),
        Transform::hflip(0.5),
        Transform::some_of(
            vec![
                Transform::rotate((-15.0, 15.0), 1.0),
                Transform::perspective((0.05, 0.1), 1.0),
            ],
            1,
            0.5,
        ),
        Transform::color_jitter(0.2, 0.2, 0.2, 0.05, 0.8),
        Transform::normalize_imagenet(),
    ])
    .with_bboxes(BboxParams::new(BboxFormat::Coco));
    let pipe = Pipeline::new(spec, Some(42))?;

    let img = Array3::<u8>::zeros((480, 640, 3)); // HWC
    let mut inp = Input::image(Buf::U8(img));
    inp.bboxes = vec![[10.0, 20.0, 50.0, 40.0]];
    let out = pipe.apply(inp)?; // out.image, out.masks, out.bboxes, out.bbox_ids, out.keypoints, ...
    println!("{} boxes kept", out.bboxes.len());

    // batches run on rayon; sample i uses a seed derived from (7, i)
    let batch = vec![Input::image(Buf::U8(Array3::zeros((480, 640, 3))))];
    let _outs = pipe.apply_batch(batch, Some(7), false);
    Ok(())
}
```

Transforms are a serde-serialisable `Transform` enum, so a `PipelineSpec` can also be built from
JSON (`Pipeline::from_json`).

## Benchmarks

augrs vs Albumentations 2.0.8 on 300 COCO val2017 images (mean 482x577, 2145 boxes), decoded once
up front; only augmentation is timed. Pipeline: `RandomResizedCrop(512x512, scale=(0.25, 1))`,
`HorizontalFlip(0.5)`, `Affine(rotate ±15, scale 0.9-1.1, translate ±6.25%, p=0.5)`,
`ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.8)`, `GaussianBlur((3, 5), p=0.2)`, `Normalize()`.
"Detection" adds COCO boxes and labels, "segmentation" a uint8 mask. OpenCV was limited to one
thread per worker.

Machine: Intel i9-13900H laptop (P- and E-cores) under WSL2. Each number is the median of 3 runs
(each run the best of 2), with the individual runs in brackets.

| images/s (higher is better) | detection | segmentation |
|---|---:|---:|
| Albumentations, 1 thread | 230 [259, 230, 221] | 221 [303, 221, 213] |
| Albumentations, 4 Python threads | 251 [271, 251, 227] | 669 [735, 571, 669] |
| Albumentations, 4 processes (fork pool, no result transfer) | 757 [796, 757, 685] | 794 [891, 705, 794] |
| augrs, 1 thread, per-image `t(...)` calls | **417** [417, 397, 418] | **389** [389, 374, 396] |
| augrs, `augment_batch(num_threads=1)` | **461** [474, 457, 461] | **407** [407, 396, 409] |
| augrs, 4 Python threads calling `t(...)` | **1504** [1613, 1504, 1463] | **1310** [1293, 1310, 1325] |
| augrs, `augment_batch(num_threads=2)` | 833 [871, 833, 799] | 767 [767, 758, 767] |
| augrs, `augment_batch(num_threads=4)` | **1393** [1493, 1332, 1393] | **1348** [1382, 1348, 1254] |

> **Noise caveat:** the machine was shared with other jobs, and single runs vary by up to ±25%
> (mostly P-core vs E-core scheduling). Read the ratios rather than the absolute numbers, and
> run the benchmark on your own hardware.

- **Single thread:** augrs is about 1.8-2x Albumentations.
- **Four threads:** about 1.7-2x Albumentations' 4 worker processes, without extra processes, and
  5-6x Albumentations with 4 Python threads on detection (its box bookkeeping holds the GIL). The
  process-pool row also excludes shipping the float outputs back to the parent, which a real
  `DataLoader` pays.
- **PyTorch DataLoader** (`examples/torch_dataloader.py`, 256 COCO images to 384x384 with boxes,
  batch 32): Albumentations in `__getitem__` 312 img/s (0 workers) / 635 img/s (4 workers); augrs
  in `__getitem__` 483 img/s (0 workers); augrs `augment_batch` in `collate_fn` 498 img/s
  (1 thread) / **1251 img/s** (4 threads, no worker processes).
- **Per kernel** (512x512 RGB, one thread, AVX2 vs scalar fallback): hue shift 0.52 vs 2.1 ms,
  warp 0.6 vs 1.3 ms, resize 0.5 vs 0.56 ms, saturation 0.12 vs 0.32 ms.

Raw numbers are in [`benches/results/coco300.json`](benches/results/coco300.json). To reproduce:

```bash
bash scripts/fetch_coco_subset.sh             # COCO val2017 annotations + 300 images into ./data
RUNS=3 bash scripts/bench_median.sh --repeats 2
bash scripts/torch_example.sh                 # DataLoader comparison (needs torch)
bash scripts/profile_ops.sh                   # per-kernel timings
```

## Differences from Albumentations

- **Keypoint coordinates are continuous**: pixel `i` spans `[i, i+1)` and its centre is at
  `i + 0.5`. Pass `KeypointParams(pixel_index_coords=True)` for the pixel-index convention; flips,
  transposes and 90-degree rotations then match Albumentations exactly.
- **`BboxParams(clip=True)` is the default** (Albumentations: `False`).
- **Reflect padding does not duplicate targets**: Albumentations 2.0.8 mirrors boxes and keypoints
  into reflect-padded borders (`PadIfNeeded`, `pad_if_needed`, and warps with reflecting borders);
  augrs keeps exactly one box / keypoint per input.
- **Box filtering happens once**, at the end of the pipeline: `min_visibility` is the fraction of
  the *input* box still visible after all transforms. `check_each_transform` is accepted for config
  compatibility.
- **Random draws differ**: augrs uses its own RNG (xoshiro256++), so the same seed does not give
  the same samples as Albumentations.
- **`Perspective`**: the jittered quadrilateral is mapped exactly onto the output. Albumentations
  2.0.8 additionally zooms in by `size / quad_size` when `keep_size=True`.
- **`fit_output`** (`Rotate`, `Affine`, `Perspective`) uses whole-pixel floor/ceil bounds;
  Albumentations' canvas is one pixel larger.
- **`GaussNoise`**: normal samples come from a 65536-entry inverse-CDF table (tails cut at about
  ±4.2 sigma) and are rounded rather than truncated.
- **`ElasticTransform`**: for `sigma > 8` the noise is generated on a grid `sigma / 4` times
  coarser and upsampled (same standard deviation) instead of a very large blur at full resolution.
  Boxes are the union of the output pixels that sample inside them and of their mapped boundary;
  keypoints use the exact inverse of the displacement field.
- **`CLAHE` on RGB** converts to Lab in floating point; OpenCV uses fixed-point tables.
- **Masks** default to nearest-pixel-centre sampling (OpenCV `INTER_NEAREST_EXACT`); every
  geometric transform accepts `mask_interpolation` (also settable on `Compose`).
- **Cubic, area and lanczos resizes** use `fast_image_resize` kernels (antialiased, Catmull-Rom for
  cubic), so they differ from OpenCV's.
- **`CoarseDropout`** adds `bbox_handling` (`"shrink"`, the Albumentations behaviour, `"visibility"`
  or `"keep"`) and `keypoint_handling` options.

## Determinism

- A `Compose(..., seed=s)` produces the same sequence of outputs every time it is created with that
  seed. `t.set_seed(s)` resets it.
- `t(..., seed=s)` replays one specific draw; with `Compose(save_applied_params=True)` the output
  also contains `applied_transforms` (the sampled parameters) and the `seed` that produced it.
- `augment_batch(..., seed=s)` gives sample `i` a seed derived from `(s, i)`: the results are the
  same for any `num_threads` (1 vs 4 is tested).
- The AVX2 kernels and the portable scalar kernels produce identical results (every kernel is
  tested against its scalar version; the whole Rust suite also runs with `AUGRS_FORCE_SCALAR=1`).
- The RNG is implemented in augrs (no dependency on platform or library RNGs). Outputs for a given
  seed may still change between augrs releases while the project is pre-1.0.
- **Worker processes.** A process forked after the pipeline was created (PyTorch `DataLoader`
  workers on Linux) re-seeds the pipeline on its first call: from `(seed, worker seed)` inside a
  PyTorch worker (reproducible for a seeded `DataLoader`), from fresh entropy when `seed=None`.
  Without this every worker would replay the same augmentation stream (Albumentations 2.0.8 has
  that problem). Pipelines pickle, so spawned workers (Windows, macOS) work too; an unpickled
  pipeline restarts its stream from its seed. A seeded pipeline used in other kinds of forked
  processes keeps its stream: call `t.set_seed(...)` in each process if needed.

## Input validation and limits

- **Parameters are checked when a transform is created.** Non-finite values (NaN, infinity), `p`
  outside `[0, 1]`, zero or negative sizes, inverted ranges and values beyond the limits below
  raise `ValueError` naming the transform (and `NotImplementedError` for Albumentations options
  augrs does not support), so a bad config fails where it is built, not in a `DataLoader` worker.
- **Images** must be non-empty `(H, W)` or `(H, W, C)` arrays of uint8 or float32 (`ValueError` /
  `TypeError` otherwise); masks and additional images must have the image's height and width.
- **Boxes** with NaN or infinite values, or with `x_max < x_min` / `y_max < y_min`, raise
  `ValueError`. Boxes partly or fully outside the image are clipped (`clip=True`, the default) or
  raise (`clip=False`). Boxes that are empty after clipping or after the pipeline (zero area,
  outside the output) or that fail the `BboxParams` thresholds are dropped together with their
  labels. An empty box list is fine.
- **Keypoints** with a non-finite coordinate, angle or scale raise `ValueError`. With
  `remove_invisible=True` (the default) keypoints outside the output image are dropped, including
  ones that were already outside on input; with `False` they are kept with their coordinates.
- **Limits** (far beyond normal use; they bound memory and time for malformed configs): images up
  to 1,048,576 px per side and 2^31 - 1 elements; `GaussianBlur` kernels up to 1023 (so `sigma`
  up to 146 when the kernel size is derived from it); `CoarseDropout` up to 4096 holes;
  `SomeOf(replace=True)` up to `n = 1024`; `CLAHE` up to 256 tiles per axis; compositions nested
  up to 32 levels. A parameter beyond a limit raises at construction; a transform whose *output*
  would exceed the size limit for a given image (for example `Affine(fit_output=True)` with a
  huge scale) raises `ValueError` when it is called.

## Development

```bash
# one-time: a virtualenv with the test dependencies
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest pyyaml "albumentations==2.0.8" opencv-python-headless

bash scripts/test_rust.sh                        # Rust unit + property tests (release)
AUGRS_FORCE_SCALAR=1 bash scripts/test_rust.sh   # same, on the portable scalar kernels
bash scripts/build_py.sh                         # maturin develop --release into the venv
bash scripts/test_py.sh                          # pytest: parity with Albumentations/OpenCV, API, serialisation
cargo fmt --all --check && cargo clippy --workspace --all-targets --release -- -D warnings
```

The scripts source [`scripts/env.sh`](scripts/env.sh), which activates `$AUGRS_VENV` (default
`./.venv`) and sets `AUGRS_DATA` (default `./data`, git-ignored). Set `CARGO_TARGET_DIR` to keep
build output elsewhere (on WSL, a Linux path is much faster than a Windows drive).

**Windows wheel from WSL/Linux.** `scripts/build_wheel_windows.sh` cross-builds the abi3 Windows
wheel with mingw-w64 (`rustup target add x86_64-pc-windows-gnu`, `apt install gcc-mingw-w64-x86-64`)
into `dist/`. On the Windows host, `scripts/test_wheel_windows.ps1` installs it into a fresh venv
and runs the smoke test and the test suite:

```powershell
powershell -File scripts\test_wheel_windows.ps1 -Python python -VenvDir $env:TEMP\augrs-venv
```

CI (`.github/workflows/ci.yml`) runs rustfmt, clippy and the Rust tests on Linux, Windows and
macOS, plus pytest on Python 3.9, 3.12 and 3.13. `.github/workflows/wheels.yml` (manual or on a
`v*` tag) builds abi3 wheels for Linux (x86_64, aarch64), Windows and macOS (x86_64, arm64) plus
an sdist, runs the smoke test and the full pytest suite on each build machine, then checks the
distributions (`twine check`, license files, metadata) and installs the sdist in a fresh venv. On a
`v*` tag it then publishes everything to PyPI (trusted publishing).

Tested by hand as well: the Windows wheel in a fresh native Python 3.10 venv (412 tests pass), and
on Apple Silicon (macOS, M5 Pro) both a source build and the CI wheel (412 tests pass, Rust tests
pass, and a pipeline runs in spawned worker processes).

**Fuzzing** (nightly Rust and `cargo install cargo-fuzz`; see [`fuzz/`](fuzz)):

```bash
cargo +nightly fuzz run pipeline -- -jobs=2 -rss_limit_mb=2048 -timeout=10 -max_total_time=1200
cargo +nightly fuzz run single_transform -- -jobs=2 -rss_limit_mb=2048 -timeout=10 -max_total_time=1200
cargo +nightly fuzz run from_json -- -dict=fuzz/dict/augrs_json.dict -jobs=2 -max_total_time=1200
python fuzz/python/fuzz_from_dict.py --loop --seconds 1200   # Albumentations config loader
```

## Project layout

```
Cargo.toml                    workspace (rustfmt max_width 120)
pyproject.toml                Python package metadata (maturin build backend)
crates/augrs-core/            Rust core: ndarray HWC buffers, fast_image_resize, rayon
  src/geometry.rs             Affine2 and Homography in continuous pixel coordinates
  src/geo.rs                  one sampled geometric op -> images, masks, boxes, keypoints
  src/targets.rs              box/keypoint formats, clipping, visibility filtering
  src/transforms.rs           the Transform enum (serde) and parameter sampling
  src/pipeline.rs             Pipeline: seeded RNG, apply / apply_batch (rayon)
  src/rng.rs                  xoshiro256++ / splitmix64
  src/ops/                    kernels: flip/crop/pad, resize, warp, remap, color, hsv, clahe,
                              noise, blur, normalize, simd.rs (AVX2, runtime-detected)
  tests/props.rs              proptest property tests (joint-target consistency)
  tests/robustness.rs         invalid parameters / targets, degenerate inputs, limits (fuzz regressions)
  examples/profile_ops.rs     per-kernel timings
crates/augrs-py/              PyO3 (abi3-py39) extension and the `augrs` Python package (maturin)
  python/augrs/               _transforms.py, _compose.py, serialization.py
compat-tests/                 pytest: parity with Albumentations 2.0.8 / OpenCV, API, serialisation
  fixtures/                   configs exported by Albumentations 2.0.8 (JSON and YAML)
examples/torch_dataloader.py  augment_batch in a PyTorch collate_fn
examples/train_parity.py      train the same classifier with augrs and Albumentations (Imagenette)
fuzz/                         cargo-fuzz targets (random pipelines, JSON specs) and a config-loader fuzzer
benches/                      benchmark script and results
scripts/                      build, test, benchmark and wheel scripts
docs/MOTIVATION.md            background and original scope
```

## Roadmap

- AVX2 kernels for Gaussian blur and nearest-neighbour mask warps; NEON kernels for aarch64 and
  Apple silicon (which currently run the correct, deterministic scalar fallback).
- More transforms (`MotionBlur`, `Downscale`, `GridDistortion`, `RandomScale`, ...), `max_size_hw`
  and inpainting fills.
- The core crate on crates.io.
- A Burn integration crate for Rust-native training.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be dual-licensed as above,
without any additional terms or conditions.

## Acknowledgements

- [Albumentations](https://github.com/albumentations-team/albumentations) (MIT), whose API and
  behaviour augrs follows, and whose 2.0.8 release the test suite compares against.
- [OpenCV](https://opencv.org) (Apache-2.0), whose arithmetic several kernels reproduce.
- [PyO3](https://pyo3.rs), [maturin](https://www.maturin.rs), [rust-numpy](https://github.com/PyO3/rust-numpy),
  [ndarray](https://github.com/rust-ndarray/ndarray), [rayon](https://github.com/rayon-rs/rayon),
  [fast_image_resize](https://github.com/Cykooz/fast_image_resize) and
  [mimalloc](https://github.com/microsoft/mimalloc).
- The [COCO](https://cocodataset.org) dataset, used (downloaded on demand) for benchmarks.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details and license notices. augrs
contains no code from AlbumentationsX.
