"""Compare deterministic augrs transforms with Albumentations 2.0.8 (last MIT release) / OpenCV."""

import albumentations as A
import cv2
import numpy as np
import pytest

import augrs
from conftest import cv2_arm_build

FORMATS = ["pascal_voc", "coco", "yolo", "albumentations"]


def boxes_in(fmt, h, w):
    voc = np.array([[5, 7, 40, 50], [60.5, 10.25, 130, 90], [0, 0, 131, 97], [20, 30, 21, 31]], dtype=np.float64)
    if fmt == "pascal_voc":
        return voc
    if fmt == "coco":
        return np.c_[voc[:, :2], voc[:, 2:] - voc[:, :2]]
    if fmt == "albumentations":
        return voc / [w, h, w, h]
    cx, cy = (voc[:, 0] + voc[:, 2]) / 2 / w, (voc[:, 1] + voc[:, 3]) / 2 / h
    return np.c_[cx, cy, (voc[:, 2] - voc[:, 0]) / w, (voc[:, 3] - voc[:, 1]) / h]


def run_pair(albu_t, augrs_t, image, fmt=None, mask=None, keypoints=None, kp_index=False):
    akw, rkw = {}, {}
    if fmt:
        akw["bbox_params"] = A.BboxParams(format=fmt, label_fields=["labels"])
        rkw["bbox_params"] = augrs.BboxParams(format=fmt, label_fields=["labels"])
    if keypoints is not None:
        akw["keypoint_params"] = A.KeypointParams(format="xy", remove_invisible=False)
        rkw["keypoint_params"] = augrs.KeypointParams(format="xy", remove_invisible=False, pixel_index_coords=kp_index)
    a = A.Compose([albu_t], **akw)
    r = augrs.Compose([augrs_t], seed=0, **rkw)
    data = {"image": image}
    if fmt:
        h, w = image.shape[:2]
        data["bboxes"] = boxes_in(fmt, h, w)
        data["labels"] = list(range(len(data["bboxes"])))
    if mask is not None:
        data["mask"] = mask
    if keypoints is not None:
        data["keypoints"] = keypoints
    return a(**data), r(**data)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("which", ["h", "v"])
def test_flips_exact(image, fmt, which):
    mask = (image[..., 0] > 128).astype(np.uint8)
    kps = [(0, 0), (10.5, 20), (130, 96)]
    at, rt = (A.HorizontalFlip(p=1), augrs.HorizontalFlip(p=1)) if which == "h" else (A.VerticalFlip(p=1), augrs.VerticalFlip(p=1))
    ao, ro = run_pair(at, rt, image, fmt, mask, kps, kp_index=True)
    np.testing.assert_array_equal(ro["image"], ao["image"])
    np.testing.assert_array_equal(ro["mask"], ao["mask"])
    np.testing.assert_allclose(ro["bboxes"], np.asarray(ao["bboxes"]), rtol=1e-5, atol=1e-5)  # albumentations works in float32
    assert list(ro["labels"]) == list(ao["labels"])
    # with pixel_index_coords=True keypoints follow Albumentations' (W-1)-x convention exactly
    np.testing.assert_allclose(np.asarray(ro["keypoints"]), np.asarray(ao["keypoints"]), atol=1e-6)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("size", [(50, 64), (97, 131), (1, 1), (96, 130)])
def test_center_crop_exact(image, fmt, size):
    mask = (image[..., 1] > 100).astype(np.uint8)
    ao, ro = run_pair(A.CenterCrop(*size, p=1), augrs.CenterCrop(*size), image, fmt, mask)
    np.testing.assert_array_equal(ro["image"], ao["image"])
    np.testing.assert_array_equal(ro["mask"], ao["mask"])
    # same surviving boxes, same coordinates
    assert list(ro["labels"]) == list(ao["labels"])
    np.testing.assert_allclose(np.asarray(ro["bboxes"]).reshape(-1, 4), np.asarray(ao["bboxes"]).reshape(-1, 4), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("size", [(48, 64), (200, 300), (97, 50), (31, 263), (64, 64)])
def test_resize_linear_matches_opencv(image, size):
    ao, ro = run_pair(A.Resize(*size, interpolation=cv2.INTER_LINEAR, p=1), augrs.Resize(*size), image, "pascal_voc")
    diff = np.abs(ro["image"].astype(int) - ao["image"].astype(int))
    assert ro["image"].shape == ao["image"].shape
    assert diff.max() <= 1, diff.max()
    # same fixed-point arithmetic as OpenCV's x86-64 build: (nearly) bit-exact. OpenCV's arm
    # builds round differently for some sizes (about 22% of pixels by one level; see cv2_arm_build)
    limit = 0.3 if cv2_arm_build() else 0.005
    assert (diff > 0).mean() < limit, (diff > 0).mean()
    np.testing.assert_allclose(ro["bboxes"], np.asarray(ao["bboxes"]), atol=1e-4)


@pytest.mark.parametrize("size", [(48, 64), (200, 300), (31, 263)])
def test_resize_nearest_exact_matches_opencv(image, size):
    ref = cv2.resize(image, size[::-1], interpolation=cv2.INTER_NEAREST_EXACT)
    out = augrs.Compose([augrs.Resize(*size, interpolation=cv2.INTER_NEAREST)], seed=0)(image=image)["image"]
    np.testing.assert_array_equal(out, ref)
    # masks always use nearest-exact sampling
    mask = (image[..., 0] // 32).astype(np.uint8)
    mref = cv2.resize(mask, size[::-1], interpolation=cv2.INTER_NEAREST_EXACT)
    mout = augrs.Compose([augrs.Resize(*size)], seed=0)(image=image, mask=mask)["mask"]
    np.testing.assert_array_equal(mout, mref)


def test_resize_area_and_cubic_close_to_opencv(image):
    # INTER_AREA downscale by an integer factor is a box average in both libraries
    big = cv2.resize(image, (524, 388), interpolation=cv2.INTER_CUBIC)
    ref = cv2.resize(big, (131, 97), interpolation=cv2.INTER_AREA)
    out = augrs.Compose([augrs.Resize(97, 131, interpolation=cv2.INTER_AREA)], seed=0)(image=big)["image"]
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    # cubic: fast_image_resize uses Catmull-Rom (a=-0.5) vs OpenCV a=-0.75, and it
    # antialiases when downscaling, so only compare upscaling loosely
    ref = cv2.resize(image, (300, 200), interpolation=cv2.INTER_CUBIC)
    out = augrs.Compose([augrs.Resize(200, 300, interpolation=cv2.INTER_CUBIC)], seed=0)(image=image)["image"]
    assert np.abs(out.astype(int) - ref.astype(int)).mean() < 2.0


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_normalize_matches(image, dtype):
    img = image if dtype == np.uint8 else image.astype(np.float32) / 255.0
    mpv = 255.0 if dtype == np.uint8 else 1.0
    ao, ro = run_pair(A.Normalize(max_pixel_value=mpv, p=1), augrs.Normalize(max_pixel_value=mpv), img)
    assert ro["image"].dtype == np.float32
    np.testing.assert_allclose(ro["image"], ao["image"], atol=1e-5, rtol=1e-5)
    ao, ro = run_pair(A.Normalize(mean=0.5, std=0.25, max_pixel_value=mpv, p=1),
                      augrs.Normalize(mean=0.5, std=0.25, max_pixel_value=mpv), img)
    np.testing.assert_allclose(ro["image"], ao["image"], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("ksize,sigma", [(3, 0.8), (5, 1.5), (7, 2.0), (9, 3.0)])
def test_gaussian_blur_matches_opencv(image, ksize, sigma):
    ref = cv2.GaussianBlur(image, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)
    t = augrs.Compose([augrs.GaussianBlur(blur_limit=(ksize, ksize), sigma_limit=(sigma, sigma), p=1)], seed=0)
    out = t(image=image)["image"]
    diff = np.abs(out.astype(int) - ref.astype(int))
    assert diff.max() <= 1, diff.max()


@pytest.mark.parametrize("angle", [7.0, -33.0, 90.0, 180.0])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101, cv2.BORDER_REPLICATE])
def test_rotate_matches_opencv_warp(image, angle, border):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    ref = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=(0, 0, 0))
    t = augrs.Compose([augrs.Rotate(limit=(angle, angle), border_mode=border, p=1)], seed=0)
    out = t(image=image)["image"]
    diff = np.abs(out.astype(int) - ref.astype(int))
    # OpenCV quantises sub-pixel positions to 1/32 px, so allow small differences
    assert diff.mean() < 0.6, diff.mean()
    assert np.percentile(diff, 99.5) <= 4


def test_brightness_contrast_saturation_match(image):
    cases = [
        dict(brightness=(1.3, 1.3), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)),
        dict(brightness=(1, 1), contrast=(0.6, 0.6), saturation=(1, 1), hue=(0, 0)),
        dict(brightness=(1, 1), contrast=(1, 1), saturation=(0.3, 0.3), hue=(0, 0)),
    ]
    for kw in cases:
        ao, ro = run_pair(A.ColorJitter(**kw, p=1), augrs.ColorJitter(**kw, p=1), image)
        diff = np.abs(ro["image"].astype(int) - ao["image"].astype(int))
        assert diff.max() <= 1, (kw, diff.max())


def test_hue_close_to_albumentations(image):
    kw = dict(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0.2, 0.2))
    ao, ro = run_pair(A.ColorJitter(**kw, p=1), augrs.ColorJitter(**kw, p=1), image)
    diff = np.abs(ro["image"].astype(int) - ao["image"].astype(int))
    # Loose check; exact parity of the 8-bit HSV path is tested in test_new_transforms.py.
    assert diff.mean() < 2.0, diff.mean()
