import os
import platform
import warnings

import numpy as np
import pytest

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning)


def _natural_like(h, w, seed=0):
    """A smooth-ish RGB test image with edges and texture (deterministic)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        128 + 100 * np.sin(x / 13.0) * np.cos(y / 17.0),
        128 + 90 * np.cos((x + y) / 23.0),
        128 + 80 * np.sin(np.hypot(x - w / 2, y - h / 2) / 9.0),
    ], axis=-1)
    base[h // 4: h // 2, w // 3: w // 2] = [250, 30, 30]  # a hard-edged block
    noise = rng.normal(0, 12, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def cv2_hsv_fused() -> bool:
    """Does the installed OpenCV compute HSV->RGB with fused multiply-adds?

    OpenCV's x86-64 Linux wheels (GCC, AVX2 dispatch) contract ``v * (1 - s * h)`` into FMA;
    augrs reproduces that exactly. MSVC builds (Windows wheels) do not, which moves about
    0.01% of values by one level. HSV parity tests are exact when this returns True.
    """
    import cv2

    hsv = np.zeros((1, 32, 3), np.uint8)
    hsv[...] = (12, 235, 255)  # G = 255 * (1 - s * (1 - h)): 114 fused, 113 unfused
    return int(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0, 1]) == 114


def cv2_arm_build() -> bool:
    """Is the installed OpenCV an arm64/aarch64 build (macOS on Apple silicon, Linux aarch64)?

    OpenCV's arm builds use different kernels for a few 8-bit ops than its x86-64 builds:
    INTER_LINEAR resize rounds differently for some sizes (about 22% of pixels move by one
    level when downscaling by ~2x, up to two levels on some single-channel upscales), the
    CLAHE tile blend gives a different f32 rounding (< 0.5% of pixels, one level) and the
    HSV round trip differs on slightly more values than the Windows build.

    augrs itself computes the same results on every CPU (integer / non-contracted IEEE f32
    arithmetic; CI also runs these tests on x86 with ``AUGRS_FORCE_SCALAR=1``, the portable
    kernels that arm uses), so the tests that are exact against x86 OpenCV allow these
    documented differences on arm instead of being skipped.
    """
    return platform.machine().lower() in ("arm64", "aarch64")


def assert_cv2_u8_equal(actual, desired, arm_frac=1e-2):
    """Exact against x86-64 OpenCV; on arm builds at most 1 level on < ``arm_frac`` of values."""
    if not cv2_arm_build():
        np.testing.assert_array_equal(actual, desired)
        return
    diff = np.abs(np.asarray(actual, dtype=np.int64) - np.asarray(desired, dtype=np.int64))
    assert np.asarray(actual).shape == np.asarray(desired).shape
    assert diff.max() <= 1, diff.max()
    assert (diff > 0).mean() < arm_frac, (diff > 0).mean()


def assert_hsv_equal(actual, desired):
    """Exact when OpenCV uses FMA on x86-64 (Linux x86-64 wheels), else at most 1 level on < 0.1% of
    values (< 0.3% on arm builds, see ``cv2_arm_build``). Linux aarch64 OpenCV also fuses
    multiply-adds, but its arm kernels still differ from x86 on ~0.02-0.07% of values, so arm builds
    always take the tolerant branch."""
    if cv2_hsv_fused() and not cv2_arm_build():
        np.testing.assert_array_equal(actual, desired)
    else:
        diff = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(desired, dtype=np.float64))
        scale = 1.0 if np.asarray(desired).dtype == np.uint8 else 1.0 / 255
        limit = 3e-3 if cv2_arm_build() else 1e-3
        assert diff.max() <= scale + 1e-6, diff.max()
        assert (diff > 1e-6).mean() < limit, (diff > 1e-6).mean()


@pytest.fixture
def image():
    return _natural_like(97, 131)


@pytest.fixture
def make_image():
    return _natural_like
