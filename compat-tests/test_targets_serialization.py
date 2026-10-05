"""additional_targets and Albumentations-format (de)serialisation."""

import io
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest

import augrs

FIX = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- additional targets


def test_additional_image_and_mask_follow_primary(make_image):
    img = make_image(90, 120, seed=1)
    mask = (img[..., 0] > 120).astype(np.uint8)
    t = augrs.Compose(
        [augrs.RandomResizedCrop(size=(64, 80), scale=(0.3, 1)), augrs.HorizontalFlip(), augrs.Affine(rotate=(-30, 30), p=0.7),
         augrs.Perspective(p=0.3), augrs.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.8), augrs.HueSaturationValue(p=0.5),
         augrs.GaussNoise(p=0.5), augrs.CoarseDropout(fill="random", fill_mask=3, p=0.5), augrs.ToGray(p=0.2)],
        additional_targets={"image2": "image", "mask2": "mask"}, seed=0)
    for _ in range(10):
        out = t(image=img, image2=img.copy(), mask=mask, mask2=mask.copy())
        np.testing.assert_array_equal(out["image"], out["image2"])
        np.testing.assert_array_equal(out["mask"], out["mask2"])
    # a different second image gets the same geometry: a constant image stays constant inside the warp
    out = t(image=img, image2=np.full_like(img, 77), seed=5)
    assert out["image2"].shape == out["image"].shape


def test_additional_bboxes_and_keypoints(make_image):
    img = make_image(90, 120, seed=2)
    geo = [augrs.RandomResizedCrop(size=(64, 80), scale=(0.5, 1)), augrs.HorizontalFlip(), augrs.Rotate(limit=20, p=1)]
    t = augrs.Compose(geo, bbox_params=augrs.BboxParams("pascal_voc", label_fields=["labels"]),
                      keypoint_params=augrs.KeypointParams("xy", label_fields=["kl"]),
                      additional_targets={"bboxes2": "bboxes", "keypoints2": "keypoints"}, seed=0)
    b1 = [(10, 10, 50, 40), (60, 20, 110, 80)]
    b2 = [(5, 50, 30, 85, "tag-a"), (70, 5, 100, 30, "tag-b"), (0, 0, 2, 2, "tiny")]
    k1, k2 = [(20, 20), (100, 70)], [(60.5, 45.5, ), (1, 1)]
    out = t(image=img, bboxes=b1, labels=["x", "y"], bboxes2=b2, keypoints=k1, kl=[1, 2], keypoints2=k2, seed=123)
    # same as running each set alone with the same seed
    solo1 = t(image=img, bboxes=b1, labels=["x", "y"], keypoints=k1, kl=[1, 2], seed=123)
    solo2 = t(image=img, bboxes=[b[:4] for b in b2], keypoints=k2, seed=123)
    assert out["bboxes"] == solo1["bboxes"] and out["labels"] == solo1["labels"]
    assert [b[:4] for b in out["bboxes2"]] == solo2["bboxes"]
    assert all(isinstance(b[4], str) for b in out["bboxes2"])
    np.testing.assert_allclose(np.asarray(out["keypoints2"]).reshape(-1, 2), np.asarray(solo2["keypoints"]).reshape(-1, 2))
    np.testing.assert_allclose(np.asarray(out["keypoints"]), np.asarray(solo1["keypoints"]))


def test_additional_targets_match_albumentations(image):
    mask = (image[..., 2] > 100).astype(np.uint8)
    img2 = image[::-1].copy()
    kw = dict(additional_targets={"image2": "image", "mask2": "mask"})
    a = A.Compose([A.HorizontalFlip(p=1), A.CenterCrop(60, 70)], **kw)
    r = augrs.Compose([augrs.HorizontalFlip(p=1), augrs.CenterCrop(60, 70)], **kw)
    ao = a(image=image, image2=img2, mask=mask, mask2=1 - mask)
    ro = r(image=image, image2=img2, mask=mask, mask2=1 - mask)
    for k in ("image", "image2", "mask", "mask2"):
        np.testing.assert_array_equal(ro[k], ao[k])


def test_additional_targets_batch_and_errors(make_image):
    imgs = [make_image(40 + i, 50, seed=i) for i in range(6)]
    t = augrs.Compose([augrs.RandomCrop(32, 32), augrs.ColorJitter(p=1)], additional_targets={"img2": "image"}, seed=0)
    outs = t.augment_batch(imgs, img2=[i.copy() for i in imgs], seed=3, num_threads=2)
    for o in outs:
        np.testing.assert_array_equal(o["image"], o["img2"])
    with pytest.raises(ValueError):
        augrs.Compose([augrs.HorizontalFlip()], additional_targets={"x": "points"})
    with pytest.raises(ValueError):
        t(image=imgs[0], img2=imgs[1])  # different size


# ---------------------------------------------------------------- serialisation


def _all_transforms(lib):
    L = lib
    return [
        L.HorizontalFlip(), L.VerticalFlip(p=0.3), L.Transpose(), L.RandomRotate90(),
        L.RandomCrop(32, 40), L.CenterCrop(30, 30, pad_if_needed=True), L.RandomResizedCrop(size=(48, 64), scale=(0.3, 0.9)),
        L.Resize(50, 60, interpolation=cv2.INTER_AREA), L.LongestMaxSize(max_size=64), L.SmallestMaxSize(max_size=40),
        L.PadIfNeeded(min_height=64, min_width=70, border_mode=cv2.BORDER_REFLECT_101),
        L.Rotate(limit=20, border_mode=cv2.BORDER_REPLICATE, crop_border=False),
        L.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.1, 0.1), "y": 0}, rotate=(-10, 10), shear=5,
                 balanced_scale=True),
        L.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
        L.Perspective(scale=(0.02, 0.08), keep_size=True), L.ElasticTransform(alpha=20, sigma=5, approximate=True),
        L.OneOf([L.ColorJitter(0.2, 0.2, 0.2, 0.05), L.RandomBrightnessContrast(0.1, 0.2), L.HueSaturationValue(10, 15, 10)]),
        L.SomeOf([L.RandomGamma(gamma_limit=(90, 110)), L.CLAHE(clip_limit=2), L.GaussNoise(std_range=(0.01, 0.05))], n=2),
        L.Sequential([L.ToGray(p=0.1), L.GaussianBlur(blur_limit=(3, 5))], p=0.7),
        L.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(4, 10), hole_width_range=(4, 10), fill=0, fill_mask=0),
        L.Normalize(),
    ]


def test_albumentations_dict_loads_and_roundtrips(make_image):
    a = A.Compose(_all_transforms(A), bbox_params=A.BboxParams("coco", label_fields=["labels"], min_visibility=0.2),
                  keypoint_params=A.KeypointParams("xy", remove_invisible=False))
    d = A.to_dict(a)
    t = augrs.from_dict(d, seed=0)
    assert isinstance(t, augrs.Compose) and len(t.transforms) == len(a.transforms)
    assert t.bbox_params.min_visibility == 0.2 and t.bbox_params.clip is False
    img = make_image(70, 90)
    out = t(image=img, bboxes=[(5, 5, 30, 20)], labels=[1], keypoints=[(10, 10)])
    assert out["image"].dtype == np.float32
    # our export is loadable by Albumentations and by augrs again (same behaviour)
    d2 = augrs.to_dict(t)
    assert d2["transform"]["transforms"][6]["size"] == [48, 64]
    A.from_dict(d2)
    t2 = augrs.from_dict(d2, seed=0)
    for s in range(5):
        o1 = t(image=img, bboxes=[(5, 5, 30, 20)], labels=[1], keypoints=[(10, 10)], seed=s)
        o2 = t2(image=img, bboxes=[(5, 5, 30, 20)], labels=[1], keypoints=[(10, 10)], seed=s)
        np.testing.assert_array_equal(o1["image"], o2["image"])
        assert o1["bboxes"] == o2["bboxes"]


@pytest.mark.parametrize("name", ["detection", "segmentation", "keypoints"])
@pytest.mark.parametrize("ext", ["json", "yaml"])
def test_load_albumentations_exported_files(make_image, name, ext):
    t = augrs.load(FIX / f"albu_2.0.8_{name}.{ext}", seed=1)
    img = make_image(300, 400, seed=4)
    if name == "detection":
        out = t(image=img, bboxes=[(10, 10, 100, 80), (200, 150, 50, 60)], labels=["a", "b"])
        assert out["image"].shape == (512, 512, 3) and out["image"].dtype == np.float32
        assert len(out["bboxes"]) == len(out["labels"])
    elif name == "segmentation":
        m = (img[..., 0] > 128).astype(np.uint8)
        out = t(image=img, image2=img.copy(), mask=m, mask2=m.copy())
        assert out["image"].shape == (224, 224, 3)
        np.testing.assert_array_equal(out["image"], out["image2"])
        np.testing.assert_array_equal(out["mask"], out["mask2"])
    else:
        a = A.load(str(FIX / f"albu_2.0.8_{name}.{ext}"), data_format=ext)
        kps = [(10.0, 20.0), (300.0, 100.0)]
        ao = a(image=img, keypoints=kps, kp_labels=[0, 1])
        ro = t(image=img, keypoints=kps, kp_labels=[0, 1])
        assert np.abs(ro["image"].astype(int) - ao["image"].astype(int)).max() <= 1
        # Albumentations flips in pixel-index coordinates; augrs is continuous (<= 1 px apart here)
        np.testing.assert_allclose(np.asarray(ro["keypoints"]), np.asarray(ao["keypoints"]), atol=1.0)
        assert list(ro["kp_labels"]) == list(ao["kp_labels"])


def test_deterministic_config_matches_albumentations(image):
    a = A.Compose([A.HorizontalFlip(p=1), A.CenterCrop(50, 60), A.Transpose(p=1), A.Normalize(mean=0.5, std=0.2)],
                  bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]))
    t = augrs.from_dict(A.to_dict(a))
    boxes = [(5, 7, 40, 50), (60.5, 10.25, 130, 90)]
    ao, ro = a(image=image, bboxes=boxes, labels=[1, 2]), t(image=image, bboxes=boxes, labels=[1, 2])
    np.testing.assert_allclose(ro["image"], ao["image"], atol=1e-5)
    np.testing.assert_allclose(np.asarray(ro["bboxes"]), np.asarray(ao["bboxes"]), atol=1e-4)


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_augrs_save_load_roundtrip(tmp_path, make_image, fmt):
    t = augrs.Compose(
        [augrs.Rotate(limit=30, fit_output=True, p=1), augrs.CoarseDropout(bbox_handling="visibility", p=1),
         augrs.Affine(rotate=10, fit_output=True, p=0.5), augrs.RandomResizedCrop(40, 50)],
        bbox_params=augrs.BboxParams("yolo", label_fields=["c"], min_area=4),
        keypoint_params=augrs.KeypointParams("xya", pixel_index_coords=True),
        additional_targets={"depth": "image"}, seed=7)
    path = tmp_path / f"cfg.{fmt}"
    augrs.save(t, path)
    t2 = augrs.load(path, seed=7)
    d = augrs.to_dict(t2)["transform"]
    assert d["transforms"][0]["fit_output"] is True and d["transforms"][1]["bbox_handling"] == "visibility"
    assert d["additional_targets"] == {"depth": "image"} and d["keypoint_params"]["pixel_index_coords"] is True
    img = make_image(60, 70)
    for s in range(5):
        kw = dict(image=img, depth=img[..., :1].copy(), bboxes=[(0.5, 0.5, 0.3, 0.4)], c=[3], keypoints=[(10, 12, 30)], seed=s)
        o1, o2 = t(**kw), t2(**kw)
        np.testing.assert_array_equal(o1["image"], o2["image"])
        np.testing.assert_array_equal(o1["depth"], o2["depth"])
        assert o1["bboxes"] == o2["bboxes"] and o1["keypoints"] == o2["keypoints"]
    # buffers and from_json with an Albumentations-format string
    buf = io.StringIO()
    augrs.save(t, buf, data_format="json")
    t3 = augrs.Compose.from_json(buf.getvalue(), seed=7)
    np.testing.assert_array_equal(t3(image=img, seed=1)["image"], t(image=img, seed=1)["image"])


def test_unsupported_configs_raise():
    with pytest.raises(NotImplementedError, match="MotionBlur"):
        augrs.from_dict({"__class_fullname__": "MotionBlur", "p": 0.5})
    with pytest.raises(NotImplementedError, match="pca"):
        augrs.from_dict({"__class_fullname__": "ToGray", "p": 0.5, "method": "pca"})
    with pytest.raises(NotImplementedError, match="area_for_downscale"):
        augrs.from_dict({"__class_fullname__": "Resize", "height": 4, "width": 4, "area_for_downscale": "image"})
    with pytest.raises(NotImplementedError, match="bogus"):
        augrs.from_dict({"__class_fullname__": "HorizontalFlip", "p": 0.5, "bogus": 1})
    with pytest.raises(NotImplementedError, match="inpaint"):
        augrs.from_dict({"__class_fullname__": "CoarseDropout", "fill": "inpaint_telea"})
    # Albumentations 1.x full class paths work
    t = augrs.from_dict({"__class_fullname__": "albumentations.augmentations.geometric.transforms.HorizontalFlip",
                         "p": 1.0, "always_apply": False})
    assert isinstance(t, augrs.HorizontalFlip)


def test_internal_json_still_roundtrips(image):
    t = augrs.Compose([augrs.HorizontalFlip(p=1), augrs.CLAHE(p=1)], seed=1)
    t2 = augrs.Compose.from_json(t.to_json(), seed=1)
    np.testing.assert_array_equal(t(image=image)["image"], t2(image=image)["image"])
    assert json.loads(t.to_json())["transforms"][1]["type"] == "CLAHE"
