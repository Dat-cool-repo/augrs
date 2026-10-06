"""Robustness: invalid parameters raise at construction, invalid targets raise (or are filtered,
as documented), degenerate inputs never crash, pipelines pickle, and forked workers get their own
random streams. Several cases are regressions found by the fuzzers in ``fuzz/``."""

import math
import os
import pickle
import sys

import numpy as np
import pytest

import augrs as A

NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("make", [
    lambda: A.HorizontalFlip(p=1.5),
    lambda: A.HorizontalFlip(p=-0.1),
    lambda: A.HorizontalFlip(p=NAN),
    lambda: A.Resize(-5, 10),
    lambda: A.Resize(0, 10),
    lambda: A.Resize(2**21, 10),
    lambda: A.RandomCrop(10**30, 4),
    lambda: A.CenterCrop(4, 0),
    lambda: A.RandomResizedCrop(size=(0, 10)),
    lambda: A.RandomResizedCrop(size=(32, 32), scale=(0.0, 1.0)),
    lambda: A.RandomResizedCrop(size=(32, 32), scale=(NAN, 1.0)),
    lambda: A.LongestMaxSize(max_size=0),
    lambda: A.LongestMaxSize(max_size=-3),
    lambda: A.PadIfNeeded(min_height=-1, min_width=8),
    lambda: A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=0, pad_width_divisor=8),
    lambda: A.Rotate(limit=(NAN, 10)),
    lambda: A.Rotate(limit=(10, -10)),
    lambda: A.Rotate(limit=10, fill=NAN),
    lambda: A.Affine(scale=0.0),
    lambda: A.Affine(scale=(-1, 2)),
    lambda: A.Affine(shear=95),
    lambda: A.Affine(rotate=INF),
    lambda: A.ShiftScaleRotate(scale_limit=(-1.5, 0.1)),
    lambda: A.Perspective(scale=(-0.1, 0.1)),
    lambda: A.Perspective(scale=(0.05, INF)),
    lambda: A.ElasticTransform(sigma=0),
    lambda: A.ElasticTransform(sigma=-1),
    lambda: A.ElasticTransform(alpha=INF),
    lambda: A.ColorJitter(brightness=(-1, 2)),
    lambda: A.ColorJitter(hue=0.7),
    lambda: A.RandomGamma(gamma_limit=(-10, 120)),
    lambda: A.CLAHE(tile_grid_size=(0, 8)),
    lambda: A.CLAHE(tile_grid_size=(257, 8)),
    lambda: A.GaussNoise(std_range=(-0.1, 0.2)),
    lambda: A.GaussNoise(noise_scale_factor=0),
    lambda: A.CoarseDropout(num_holes_range=(5, 1)),
    lambda: A.CoarseDropout(num_holes_range=(1, 10**9)),
    lambda: A.CoarseDropout(hole_height_range=(-0.1, 0.2)),
    lambda: A.Normalize(std=0),
    lambda: A.Normalize(mean=(NAN, 0, 0)),
    lambda: A.Normalize(max_pixel_value=0),
    lambda: A.GaussianBlur(blur_limit=(3, 10**6)),
    lambda: A.GaussianBlur(blur_limit=0, sigma_limit=(1, 10**6)),
    lambda: A.SomeOf([A.HorizontalFlip()], n=10**9, replace=True),
    lambda: A.OneOf([A.HorizontalFlip(p=2.0)]),
    lambda: A.BboxParams(format="coco", min_visibility=2),
    lambda: A.BboxParams(format="coco", min_area=NAN),
    lambda: A.BboxParams(format="coco", max_accept_ratio=INF),
    lambda: A.BboxParams(format="bogus"),
])
def test_invalid_parameters_raise_at_construction(make):
    with pytest.raises(ValueError):
        make()


def test_construction_errors_name_the_transform():
    with pytest.raises(ValueError, match="ElasticTransform"):
        A.ElasticTransform(sigma=0)
    with pytest.raises(ValueError, match="Rotate.*finite"):
        A.Rotate(limit=(NAN, 1))
    with pytest.raises(ValueError, match="GaussianBlur"):
        A.GaussianBlur(blur_limit=(3, 10**6))


def test_extreme_but_valid_parameters_work(image):
    """Fuzz-style extremes that are valid must run and give sane outputs."""
    t = A.Compose([
        A.Rotate(limit=(-1e9, 1e9), p=1),
        A.Affine(translate_px=(-10**6, 10**6), p=1),
        A.Perspective(scale=1.0, p=1),
        A.ElasticTransform(alpha=1e6, sigma=1e-6, p=1),
        A.CoarseDropout(num_holes_range=(3, 3), hole_height_range=(500, 900), hole_width_range=(500, 900), p=1),
        A.GaussianBlur(blur_limit=(3, 1023), p=1),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"), seed=0)
    boxes = [(10, 10, 60, 50), (0, 0, 131, 97)]
    out = t(image=image, bboxes=boxes, labels=[1, 2], keypoints=[(5, 5), (100, 80)])
    assert out["image"].shape[:2] == image.shape[:2]
    assert len(out["bboxes"]) == len(out["labels"])
    for b in out["bboxes"]:
        assert all(math.isfinite(v) for v in b)


def test_deep_nesting():
    t = A.HorizontalFlip()
    for _ in range(20):
        t = A.OneOf([t], p=1)
    A.Compose([t])  # fine
    with pytest.raises(ValueError):
        for _ in range(20):
            t = A.SomeOf([t], n=1)


@pytest.mark.parametrize("shape", [(0, 0, 3), (0, 5, 3), (5, 0), (5, 5, 0)])
def test_empty_images_raise(shape):
    t = A.Compose([A.HorizontalFlip(p=1)])
    with pytest.raises(ValueError):
        t(image=np.zeros(shape, np.uint8))


@pytest.mark.parametrize("hw", [(1, 1), (1, 300), (300, 1), (2, 2)])
def test_tiny_and_thin_images(hw):
    img = np.arange(hw[0] * hw[1] * 3, dtype=np.uint8).reshape(hw + (3,))
    for tr in [A.RandomResizedCrop(size=(8, 8)), A.Rotate(p=1), A.Affine(scale=(0.5, 2), rotate=30, p=1),
               A.Perspective(p=1), A.ElasticTransform(p=1), A.CLAHE(p=1), A.GaussianBlur(p=1),
               A.CoarseDropout(p=1), A.HueSaturationValue(p=1), A.Normalize()]:
        out = A.Compose([tr], bbox_params=A.BboxParams(format="pascal_voc"), seed=1)(
            image=img, bboxes=[(0, 0, hw[1], hw[0])])
        assert out["image"].ndim == 3 and min(out["image"].shape[:2]) >= 1


def test_wrong_dtype_raises():
    t = A.Compose([A.HorizontalFlip(p=1)])
    with pytest.raises(TypeError):
        t(image=np.zeros((4, 4, 3), np.int64))


def test_invalid_boxes_raise_and_degenerate_boxes_are_filtered():
    t = A.Compose([A.HorizontalFlip(p=1)], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["l"]))
    img = np.zeros((10, 20, 3), np.uint8)
    for bad in ([(NAN, 0, 1, 1)], [(0, 0, INF, 1)], [(5, 0, 1, 1)]):
        with pytest.raises(ValueError):
            t(image=img, bboxes=bad, l=[0])
    assert t(image=img, bboxes=[], l=[])["bboxes"] == []
    out = t(image=img, bboxes=[(3, 3, 3, 8), (30, 0, 40, 5), (-5, 2, 4, 6)], l=["zero", "outside", "partial"])
    assert out["l"] == ["partial"]
    assert out["bboxes"] == [(16.0, 2.0, 20.0, 6.0)]
    strict = A.Compose([A.HorizontalFlip(p=1)], bbox_params=A.BboxParams(format="pascal_voc", clip=False))
    with pytest.raises(ValueError, match="outside"):
        strict(image=img, bboxes=[(-5, 2, 4, 6)])


def test_invalid_keypoints_raise_and_outside_ones_are_dropped():
    t = A.Compose([A.RandomBrightnessContrast(p=1)], keypoint_params=A.KeypointParams(format="xya"))
    img = np.zeros((10, 20, 3), np.uint8)
    with pytest.raises(ValueError):
        t(image=img, keypoints=[(1, 1, NAN)])
    with pytest.raises(ValueError):
        t(image=img, keypoints=[(INF, 1, 0)])
    # remove_invisible: outside points are dropped even when no geometric transform moved them
    out = t(image=img, keypoints=[(1, 1, 0), (0.75, -0.4, 0), (25, 3, 0)])
    assert len(out["keypoints"]) == 1


def test_parameters_reach_rust_exactly():
    """22.65 used to arrive in Rust as 22.650000000000002 (inexact JSON float parsing)."""
    t = A.Compose([A.Rotate(limit=(-7.35, 13.3), p=0.735)],
                  bbox_params=A.BboxParams(format="coco", min_area=22.65, min_visibility=0.199))
    import json
    rust_spec = json.loads(t._pipe.to_json())  # what the Rust core parsed, serialised back
    assert rust_spec["bbox_params"]["min_area"] == 22.65
    assert rust_spec["bbox_params"]["min_visibility"] == 0.199
    assert rust_spec["transforms"][0]["limit"] == [-7.35, 13.3]
    assert rust_spec["transforms"][0]["p"] == 0.735


def test_pickle_round_trip(image):
    t = A.Compose([A.RandomResizedCrop(size=(32, 32)), A.HorizontalFlip(), A.ColorJitter(p=1)],
                  bbox_params=A.BboxParams(format="coco", label_fields=["labels"]), seed=3)
    t2 = pickle.loads(pickle.dumps(t))
    a = t(image=image, bboxes=[(5, 5, 20, 20)], labels=[1], seed=11)
    b = t2(image=image, bboxes=[(5, 5, 20, 20)], labels=[1], seed=11)
    np.testing.assert_array_equal(a["image"], b["image"])
    assert a["bboxes"] == b["bboxes"]
    # the unpickled pipeline restarts its stream from the seed, like a fresh Compose(seed=3)
    fresh = A.Compose([A.RandomResizedCrop(size=(32, 32)), A.HorizontalFlip(), A.ColorJitter(p=1)],
                      bbox_params=A.BboxParams(format="coco", label_fields=["labels"]), seed=3)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(t))(image=image, bboxes=[], labels=[])["image"],
                                  fresh(image=image, bboxes=[], labels=[])["image"])


def _draw_in_child(t, img):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        out = t(image=img)["image"]
        os.write(w, out.tobytes()[:4096])
        os._exit(0)
    os.close(w)
    data = b""
    while chunk := os.read(r, 65536):
        data += chunk
    os.waitpid(pid, 0)
    return data


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "darwin", reason="needs fork")
def test_forked_workers_get_their_own_stream(image):
    """A forked process (DataLoader worker) must not replay the parent's random stream."""
    ts = [A.RandomResizedCrop(size=(32, 32)), A.HorizontalFlip(), A.ColorJitter(p=1)]
    t = A.Compose(ts)  # seed=None: each process re-seeds from entropy
    t(image=image)  # use it in the parent first
    assert _draw_in_child(t, image) != _draw_in_child(t, image)
    # an explicitly seeded pipeline outside a torch worker keeps its (reproducible) stream
    t = A.Compose(ts, seed=5)
    assert _draw_in_child(t, image) == _draw_in_child(t, image)
