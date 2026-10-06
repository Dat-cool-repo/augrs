"""Direct tests for the README parity-table rows that had no test of their own: RandomCrop,
CenterCrop / RandomCrop with ``pad_if_needed``, PadIfNeeded (all positions and border modes),
RandomResizedCrop (given its window, and its window distribution), LongestMaxSize /
SmallestMaxSize, Affine and ShiftScaleRotate (vs Albumentations 2.0.8 with fixed parameters)."""

import albumentations as A
import cv2
import numpy as np
import pytest

import augrs
from conftest import cv2_arm_build

FORMATS = ["pascal_voc", "coco", "yolo", "albumentations"]


def voc_boxes(h, w):
    rel = np.array([[0.04, 0.07, 0.3, 0.5], [0.46, 0.1, 0.99, 0.93], [0, 0, 1, 1], [0.15, 0.3, 0.17, 0.33]])
    return rel * [w, h, w, h]


def to_fmt(voc, fmt, h, w):
    if fmt == "pascal_voc":
        return voc
    if fmt == "coco":
        return np.c_[voc[:, :2], voc[:, 2:] - voc[:, :2]]
    if fmt == "albumentations":
        return voc / [w, h, w, h]
    cx, cy = (voc[:, 0] + voc[:, 2]) / 2 / w, (voc[:, 1] + voc[:, 3]) / 2 / h
    return np.c_[cx, cy, (voc[:, 2] - voc[:, 0]) / w, (voc[:, 3] - voc[:, 1]) / h]


def compose_pair(albu_ts, augrs_ts, fmt="pascal_voc", seed=0, record=False):
    a = A.Compose(albu_ts, bbox_params=A.BboxParams(format=fmt, label_fields=["labels"]),
                  keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    r = augrs.Compose(augrs_ts, bbox_params=augrs.BboxParams(format=fmt, label_fields=["labels"]),
                      keypoint_params=augrs.KeypointParams(format="xy", remove_invisible=False,
                                                           pixel_index_coords=True),
                      save_applied_params=record, seed=seed)
    return a, r


def sample(image, fmt):
    h, w = image.shape[:2]
    boxes = to_fmt(voc_boxes(h, w), fmt, h, w)
    return dict(image=image, mask=(image[..., 0] // 40).astype(np.uint8), bboxes=boxes,
                labels=list(range(len(boxes))), keypoints=np.array([[0.0, 0.0], [10.5, 20.0], [w - 1.0, h - 1.0]]))


REFLECT = (cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT)


def assert_targets_equal(ro, ao, exact_image=True, border=cv2.BORDER_CONSTANT):
    if exact_image:
        np.testing.assert_array_equal(ro["image"], ao["image"])
    np.testing.assert_array_equal(ro["mask"], ao["mask"])
    if border in REFLECT:
        # Albumentations 2.0.8 also mirrors boxes and keypoints into reflect-padded areas
        # (a documented difference): augrs keeps one box per object.
        assert len(ro["bboxes"]) <= len(ao["bboxes"])
        return
    assert list(ro["labels"]) == list(ao["labels"])
    np.testing.assert_allclose(np.asarray(ro["bboxes"]).reshape(-1, 4), np.asarray(ao["bboxes"]).reshape(-1, 4),
                               rtol=1e-5, atol=1e-5)  # Albumentations computes boxes in float32
    np.testing.assert_allclose(np.asarray(ro["keypoints"]), np.asarray(ao["keypoints"]), atol=1e-5)


def applied(out, name):
    return next(t["params"] for t in out["applied_transforms"] if t["name"] == name)


# ------------------------------------------------------------------------------------- crops


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("size", [(50, 64), (97, 131), (1, 1), (20, 131), (97, 3)])
@pytest.mark.parametrize("seed", range(3))
def test_random_crop_exact(image, fmt, size, seed):
    """RandomCrop = a crop at the sampled window: identical image, mask, boxes, keypoints."""
    _, r = compose_pair([], [augrs.RandomCrop(*size)], fmt, seed=seed, record=True)
    data = sample(image, fmt)
    ro = r(**data)
    p = applied(ro, "RandomCrop")
    assert (p["y_max"] - p["y_min"], p["x_max"] - p["x_min"]) == size
    a, _ = compose_pair([A.Crop(p["x_min"], p["y_min"], p["x_max"], p["y_max"], p=1)], [], fmt)
    assert_targets_equal(ro, a(**data))


@pytest.mark.parametrize("cls", ["RandomCrop", "CenterCrop"])
@pytest.mark.parametrize("size", [(120, 150), (120, 64), (50, 140)])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101, cv2.BORDER_REPLICATE])
def test_crop_pad_if_needed_exact(image, cls, size, border):
    """pad_if_needed pads to the crop size first (Albumentations: PadIfNeeded, then the crop)."""
    kw = dict(pad_if_needed=True, border_mode=border, fill=7, fill_mask=3)
    _, r = compose_pair([], [getattr(augrs, cls)(*size, **kw)], seed=1, record=True)
    data = sample(image, "pascal_voc")
    ro = r(**data)
    p = applied(ro, cls)
    a, _ = compose_pair([A.PadIfNeeded(min_height=size[0], min_width=size[1], position="center", border_mode=border,
                                       fill=7, fill_mask=3, p=1),
                         A.Crop(p["x_min"], p["y_min"], p["x_max"], p["y_max"], p=1)], [])
    assert_targets_equal(ro, a(**data), border=border)
    if cls == "CenterCrop":  # and the same as Albumentations' own CenterCrop(pad_if_needed=True)
        a, _ = compose_pair([A.CenterCrop(*size, p=1, **kw)], [])
        assert_targets_equal(ro, a(**data), border=border)


@pytest.mark.parametrize("position", ["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101, cv2.BORDER_REPLICATE,
                                    cv2.BORDER_REFLECT])
def test_pad_if_needed_exact(image, position, border):
    kw = dict(min_height=150, min_width=140, position=position, border_mode=border, fill=(1, 2, 3), fill_mask=5)
    _, r = compose_pair([], [augrs.PadIfNeeded(**kw, p=1)], "coco", seed=2, record=True)
    data = sample(image, "coco")
    ro = r(**data)
    p = applied(ro, "PadIfNeeded")
    assert ro["image"].shape[:2] == (150, 140)
    # the same padding through Albumentations' Pad (random positions are sampled by augrs)
    pad = (p["left"], p["top"], p["right"], p["bottom"])
    a, _ = compose_pair([A.Pad(padding=pad, fill=(1, 2, 3), fill_mask=5, border_mode=border, p=1)], [], "coco")
    assert_targets_equal(ro, a(**data), border=border)
    if position != "random":
        a, _ = compose_pair([A.PadIfNeeded(**kw, p=1)], [], "coco")
        assert_targets_equal(ro, a(**data), border=border)


# ---------------------------------------------------------------------------- resizing crops


def tie_positions(src, dst):
    """Output indices whose centre maps exactly halfway between two source pixel centres."""
    x = np.arange(dst)
    return ((2 * x + 1) * src) % (2 * dst) == 0


def assert_resize_close(out, ref):
    """The README's Resize row: max diff 1 level, on < 1% of pixels (x86-64 OpenCV) for images of
    at least 16 x 16 px (tiny upscales have few pixels, so the fraction is not meaningful there)."""
    assert out.shape == ref.shape
    diff = np.abs(out.astype(int) - ref.astype(int))
    assert diff.max() <= (2 if cv2_arm_build() else 1), diff.max()
    if min(out.shape[:2]) >= 16:
        assert (diff > 0).mean() < (0.3 if cv2_arm_build() else 0.01), (diff > 0).mean()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("size", [(64, 64), (160, 120)])
def test_random_resized_crop_given_window(image, seed, size):
    """RandomResizedCrop = crop at the sampled window + Resize (linear image, nearest-exact mask)."""
    _, r = compose_pair([], [augrs.RandomResizedCrop(size=size, scale=(0.2, 1.0))], seed=seed, record=True)
    data = sample(image, "pascal_voc")
    ro = r(**data)
    p = applied(ro, "RandomResizedCrop")
    a, _ = compose_pair([A.Crop(p["x_min"], p["y_min"], p["x_max"], p["y_max"], p=1),
                         A.Resize(*size, interpolation=cv2.INTER_LINEAR, p=1)], [])
    ao = a(**data)
    assert_resize_close(ro["image"], ao["image"])
    crop = data["mask"][p["y_min"]:p["y_max"], p["x_min"]:p["x_max"]]
    ref = cv2.resize(crop, size[::-1], interpolation=cv2.INTER_NEAREST_EXACT)
    # identical except where an output centre lies exactly halfway between two source centres
    # (OpenCV breaks those ties either way, depending on floating-point rounding)
    ry, rx = ~tie_positions(crop.shape[0], size[0]), ~tie_positions(crop.shape[1], size[1])
    np.testing.assert_array_equal(ro["mask"][ry][:, rx], ref[ry][:, rx])
    assert list(ro["labels"]) == list(ao["labels"])
    np.testing.assert_allclose(np.asarray(ro["bboxes"]), np.asarray(ao["bboxes"]), atol=1e-3)


def test_random_resized_crop_window_distribution():
    """The window sampling follows torchvision / Albumentations: compare the distributions of the
    window's area fraction and log aspect ratio over many draws."""
    h, w = 120, 200
    img = np.zeros((h, w, 3), np.uint8)
    kw = dict(size=(32, 32), scale=(0.1, 0.9), ratio=(0.5, 2.0))
    r = augrs.Compose([augrs.RandomResizedCrop(**kw)], save_applied_params=True, seed=0)
    at = A.RandomResizedCrop(**kw, p=1)
    at.set_random_seed(0)
    ours, theirs = [], []
    for _ in range(3000):
        p = r(image=img)["applied_transforms"][0]["params"]
        ours.append((p["x_max"] - p["x_min"], p["y_max"] - p["y_min"]))
        x0, y0, x1, y1 = at.get_params_dependent_on_data({"shape": (h, w)}, {"image": img})["crop_coords"]
        theirs.append((x1 - x0, y1 - y0))
    ours, theirs = np.array(ours, float), np.array(theirs, float)
    for f in (lambda s: s[:, 0] * s[:, 1] / (h * w), lambda s: np.log(s[:, 0] / s[:, 1])):
        a, b = f(ours), f(theirs)
        assert abs(a.mean() - b.mean()) < 0.03, (a.mean(), b.mean())
        for q in (10, 50, 90):
            assert abs(np.percentile(a, q) - np.percentile(b, q)) < 0.06, (q, np.percentile(a, q), np.percentile(b, q))


@pytest.mark.parametrize("cls", ["LongestMaxSize", "SmallestMaxSize"])
@pytest.mark.parametrize("shape,max_size", [((97, 131), 64), ((97, 131), 200), ((3, 4), 6), ((5, 2), 3), ((7, 9), 5),
                                            ((100, 50), 75)])
def test_max_size_transforms_vs_albumentations(make_image, cls, shape, max_size):
    image = make_image(*shape)
    data = sample(image, "pascal_voc")
    data["keypoints"] = np.array([[0.0, 0.0], [shape[1] / 2, shape[0] / 3]])
    a, r = compose_pair([getattr(A, cls)(max_size=max_size, p=1)], [getattr(augrs, cls)(max_size=max_size)])
    ao, ro = a(**data), r(**data)
    assert ro["image"].shape == ao["image"].shape  # same output size (including .5 rounding cases)
    assert_resize_close(ro["image"], ao["image"])
    np.testing.assert_allclose(np.asarray(ro["bboxes"]), np.asarray(ao["bboxes"]), atol=1e-3)


# --------------------------------------------------------------------------------- affine


def assert_warp_close(out, ref):
    """The README's Rotate/Affine rows: mean diff < 0.6 (OpenCV quantises to 1/32 px)."""
    assert out.shape == ref.shape
    diff = np.abs(out.astype(int) - ref.astype(int))
    assert diff.mean() < 0.6, diff.mean()
    assert np.percentile(diff, 99.5) <= 4


@pytest.mark.parametrize("kw", [
    dict(scale=0.8, rotate=30),
    dict(scale=1.25, rotate=-12, translate_px=7),
    dict(scale={"x": 0.9, "y": 1.1}, shear=10),
    dict(translate_percent=0.1, rotate=90),
    dict(scale=0.7, rotate=45, shear={"x": -8, "y": 5}, translate_percent={"x": -0.05, "y": 0.08}),
])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101])
def test_affine_fixed_params_vs_albumentations(image, kw, border):
    """With scalar (fixed) parameters both libraries are deterministic."""
    data = sample(image, "pascal_voc")
    a, r = compose_pair([A.Affine(**kw, border_mode=border, p=1)], [augrs.Affine(**kw, border_mode=border, p=1)])
    ao, ro = a(**data), r(**data)
    assert_warp_close(ro["image"], ao["image"])
    if border in REFLECT:  # Albumentations also mirrors boxes/keypoints into reflected borders
        return
    assert list(ro["labels"]) == list(ao["labels"])
    np.testing.assert_allclose(np.asarray(ro["bboxes"]), np.asarray(ao["bboxes"]), atol=0.05)
    # keypoints: same point map (Albumentations uses the pixel-index convention)
    np.testing.assert_allclose(np.asarray(ro["keypoints"]), np.asarray(ao["keypoints"]), atol=1e-3)


@pytest.mark.parametrize("shift,scale,angle", [(0.05, 0.1, 20), (-0.08, -0.2, -35), (0.0, 0.0, 90)])
def test_shift_scale_rotate_fixed_vs_albumentations(image, shift, scale, angle):
    kw = dict(shift_limit=(shift, shift), scale_limit=(scale, scale), rotate_limit=(angle, angle), p=1)
    data = sample(image, "pascal_voc")
    a, r = compose_pair([A.ShiftScaleRotate(**kw)], [augrs.ShiftScaleRotate(**kw)])
    ao, ro = a(**data), r(**data)
    assert_warp_close(ro["image"], ao["image"])
    assert list(ro["labels"]) == list(ao["labels"])
    np.testing.assert_allclose(np.asarray(ro["bboxes"]), np.asarray(ao["bboxes"]), atol=0.05)
