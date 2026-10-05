"""New transforms vs Albumentations 2.0.8 / OpenCV (exact where Albumentations is deterministic)."""

import albumentations as A
import albumentations.augmentations.dropout.functional as fdrop
import albumentations.augmentations.geometric.functional as fgeo
import cv2
import numpy as np
import pytest

import augrs
from conftest import assert_cv2_u8_equal, assert_hsv_equal, cv2_arm_build

FORMATS = ["pascal_voc", "coco", "yolo", "albumentations"]


def one(t_albu, t_augrs, **data):
    a = A.Compose([t_albu])(**data)
    r = augrs.Compose([t_augrs], seed=0)(**data)
    return a, r


# ---------------------------------------------------------------- photometric


@pytest.mark.parametrize("shifts", [(10, 0, 0), (-37.5, 0, 0), (0, 25.7, 0), (0, -40.2, 0), (0, 0, 31.9),
                                    (0, 0, -18.4), (13.3, -22.8, 17.1), (179.9, 50, -50)])
@pytest.mark.parametrize("width", [131, 64, 33])
def test_hue_saturation_value_exact(make_image, shifts, width):
    img = make_image(41, width, seed=width)
    h, s, v = shifts
    kw = dict(hue_shift_limit=(h, h), sat_shift_limit=(s, s), val_shift_limit=(v, v), p=1)
    a, r = one(A.HueSaturationValue(**kw), augrs.HueSaturationValue(**kw), image=img)
    assert_hsv_equal(r["image"], a["image"])


def test_hsv_round_trip_all_colors_exact():
    # every 24-bit colour through OpenCV-exact RGB->HSV->RGB (hue shift 180 = identity LUT);
    # width 4096 is a multiple of 32 and 4099 exercises OpenCV's per-row scalar tail
    allc = np.arange(1 << 24, dtype=np.uint32)
    img = np.stack([(allc >> 16) & 255, (allc >> 8) & 255, allc & 255], -1).astype(np.uint8)
    kw = dict(hue_shift_limit=(180, 180), sat_shift_limit=(0, 0), val_shift_limit=(0, 0), p=1)
    for w in (4096, 4099):
        n = (len(img) // w) * w
        im = img[:n].reshape(-1, w, 3)
        a, r = one(A.HueSaturationValue(**kw), augrs.HueSaturationValue(**kw), image=im)
        assert_hsv_equal(r["image"], a["image"])


def test_hue_saturation_value_gray_and_float(make_image):
    gray = make_image(30, 45)[..., 0]
    kw = dict(hue_shift_limit=(5, 5), sat_shift_limit=(5, 5), val_shift_limit=(12.5, 12.5), p=1)
    a, r = one(A.HueSaturationValue(**kw), augrs.HueSaturationValue(**kw), image=gray)
    assert_hsv_equal(r["image"], a["image"])
    f = make_image(30, 45).astype(np.float32) / 255
    a, r = one(A.HueSaturationValue(**kw), augrs.HueSaturationValue(**kw), image=f)
    assert_hsv_equal(r["image"], a["image"])


@pytest.mark.parametrize("hue", [0.2, -0.13, 0.5, 0.02])
def test_color_jitter_hue_exact(image, hue):
    kw = dict(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(hue, hue), p=1)
    a, r = one(A.ColorJitter(**kw), augrs.ColorJitter(**kw), image=image)
    assert_hsv_equal(r["image"], a["image"])


@pytest.mark.parametrize("b,c,by_max", [(0.1, 0.2, True), (-0.15, -0.3, True), (0.2, 0.0, False), (0.33, -0.1, False),
                                        (0.0, 0.7, True)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_random_brightness_contrast_exact(image, b, c, by_max, dtype):
    img = image if dtype == np.uint8 else image.astype(np.float32) / 255
    kw = dict(brightness_limit=(b, b), contrast_limit=(c, c), brightness_by_max=by_max, p=1)
    a, r = one(A.RandomBrightnessContrast(**kw), augrs.RandomBrightnessContrast(**kw), image=img)
    if dtype == np.uint8:
        np.testing.assert_array_equal(r["image"], a["image"])
    else:
        np.testing.assert_allclose(r["image"], a["image"], atol=1e-6)
    kw["ensure_safe_range"] = True
    a, r = one(A.RandomBrightnessContrast(**kw), augrs.RandomBrightnessContrast(**kw), image=img)
    np.testing.assert_allclose(r["image"].astype(float), a["image"].astype(float), atol=1e-6)


@pytest.mark.parametrize("gamma", [50, 80, 100, 117.3, 200])
def test_random_gamma_exact(image, gamma):
    kw = dict(gamma_limit=(gamma, gamma), p=1)
    a, r = one(A.RandomGamma(**kw), augrs.RandomGamma(**kw), image=image)
    np.testing.assert_array_equal(r["image"], a["image"])
    f = image.astype(np.float32) / 255
    a, r = one(A.RandomGamma(**kw), augrs.RandomGamma(**kw), image=f)
    np.testing.assert_allclose(r["image"], a["image"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["weighted_average", "average", "max", "desaturation"])
@pytest.mark.parametrize("nch", [1, 3])
def test_to_gray_exact(image, method, nch):
    kw = dict(num_output_channels=nch, method=method, p=1)
    a, r = one(A.ToGray(**kw), augrs.ToGray(**kw), image=image)
    np.testing.assert_array_equal(r["image"], a["image"].reshape(r["image"].shape))
    f = image.astype(np.float32) / 255
    a, r = one(A.ToGray(**kw), augrs.ToGray(**kw), image=f)
    np.testing.assert_allclose(r["image"], a["image"].reshape(r["image"].shape), atol=1e-6)


@pytest.mark.parametrize("shape,tiles", [((97, 131), (8, 8)), ((64, 64), (8, 8)), ((64, 96), (4, 2)), ((50, 64), (3, 5))])
@pytest.mark.parametrize("clip", [1.0, 2.5, 4.0, 40.0])
def test_clahe_gray_exact(make_image, shape, tiles, clip):
    img = make_image(*shape)[..., 1]
    kw = dict(clip_limit=(clip, clip), tile_grid_size=tiles, p=1)
    a, r = one(A.CLAHE(**kw), augrs.CLAHE(**kw), image=img)
    assert_cv2_u8_equal(r["image"], a["image"])  # exact on x86-64 OpenCV


@pytest.mark.parametrize("clip", [1.0, 2.0, 4.0])
def test_clahe_rgb_close(image, clip):
    kw = dict(clip_limit=(clip, clip), tile_grid_size=(4, 4), p=1)
    a, r = one(A.CLAHE(**kw), augrs.CLAHE(**kw), image=image)
    diff = np.abs(r["image"].astype(int) - a["image"].astype(int))
    # OpenCV's 8-bit Lab uses fixed-point tables; augrs converts in float
    assert diff.mean() < 1.0, diff.mean()
    assert np.percentile(diff, 99) <= 6, np.percentile(diff, 99)


def test_gauss_noise_statistics(make_image):
    img = np.full((256, 256, 3), 128, np.uint8)
    t = augrs.Compose([augrs.GaussNoise(std_range=(0.05, 0.05), mean_range=(0.02, 0.02), p=1)], seed=1)
    out = t(image=img)["image"].astype(float) - 128
    assert abs(out.mean() - 0.02 * 255) < 0.3
    assert abs(out.std() - 0.05 * 255) < 0.3
    # shared noise: identical across channels
    t = augrs.Compose([augrs.GaussNoise(std_range=(0.05, 0.05), per_channel=False, p=1)], seed=1)
    out = t(image=img)["image"]
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()
    # low-resolution noise is spatially smooth: neighbours are correlated
    t = augrs.Compose([augrs.GaussNoise(std_range=(0.1, 0.1), noise_scale_factor=0.25, p=1)], seed=1)
    n = t(image=img)["image"][..., 0].astype(float) - 128
    corr = np.corrcoef(n[:, :-1].ravel(), n[:, 1:].ravel())[0, 1]
    assert corr > 0.5, corr
    f = augrs.Compose([augrs.GaussNoise(p=1)], seed=0)(image=img.astype(np.float32) / 255)["image"]
    assert f.dtype == np.float32 and f.min() >= 0 and f.max() <= 1


# ---------------------------------------------------------------- geometric


@pytest.mark.parametrize("fmt", FORMATS)
def test_transpose_exact(image, fmt):
    from test_compat_albumentations import boxes_in, run_pair

    mask = (image[..., 0] > 128).astype(np.uint8)
    kps = [(0, 0), (10.5, 20), (130, 96)]
    ao, ro = run_pair(A.Transpose(p=1), augrs.Transpose(p=1), image, fmt, mask, kps, kp_index=True)
    np.testing.assert_array_equal(ro["image"], ao["image"])
    np.testing.assert_array_equal(ro["mask"], ao["mask"])
    np.testing.assert_allclose(ro["bboxes"], np.asarray(ao["bboxes"]), rtol=1e-5, atol=1e-5)
    assert list(ro["labels"]) == list(ao["labels"])
    np.testing.assert_allclose(np.asarray(ro["keypoints"]), np.asarray(ao["keypoints"]), atol=1e-5)


@pytest.mark.parametrize("seed", range(8))
def test_random_rotate90_matches_numpy_and_albumentations(image, seed):
    h, w = image.shape[:2]
    mask = (image[..., 0] > 128).astype(np.uint8)
    voc = np.array([[5, 7, 40, 50], [60.5, 10.25, 130, 90]], dtype=np.float64)
    kps = np.array([[3.0, 4.0], [100.0, 50.0]])
    t = augrs.Compose([augrs.RandomRotate90(p=1)], bbox_params=augrs.BboxParams("pascal_voc"),
                      keypoint_params=augrs.KeypointParams("xy", pixel_index_coords=True, remove_invisible=False),
                      save_applied_params=True, seed=seed)
    r = t(image=image, mask=mask, bboxes=voc, keypoints=kps)
    k = r["applied_transforms"][0]["params"]["factor"]
    np.testing.assert_array_equal(r["image"], np.rot90(image, k))
    np.testing.assert_array_equal(r["mask"], np.rot90(mask, k))
    nb = fgeo.bboxes_rot90(voc / [w, h, w, h], k)
    oh, ow = r["image"].shape[:2]
    np.testing.assert_allclose(r["bboxes"], nb * [ow, oh, ow, oh], atol=1e-9)
    kp5 = np.c_[kps, np.zeros((2, 3))]
    nk = fgeo.keypoints_rot90(kp5, k, image.shape)
    np.testing.assert_allclose(r["keypoints"], nk[:, :2], atol=1e-9)


@pytest.mark.parametrize("keep_size,fit_output", [(True, False), (False, False), (True, True), (False, True)])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101])
def test_perspective_matches_cv2_warp(image, keep_size, fit_output, border):
    t = augrs.Compose([augrs.Perspective(scale=(0.05, 0.12), keep_size=keep_size, fit_output=fit_output,
                                         border_mode=border, p=1)],
                      bbox_params=augrs.BboxParams("pascal_voc"), save_applied_params=True, seed=3)
    voc = [(20, 15, 60, 50)]
    r = t(image=image, bboxes=voc)
    prm = r["applied_transforms"][0]["params"]
    hm = np.array(prm["matrix"]).reshape(3, 3)
    oh, ow = r["image"].shape[:2]
    assert (oh, ow) == (prm["height"], prm["width"])
    if keep_size:
        assert (oh, ow) == image.shape[:2]
    # continuous -> pixel-index convention
    T = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    Ti = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    m_cv = Ti @ hm @ T
    ref = cv2.warpPerspective(image, m_cv, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=border)
    diff = np.abs(r["image"].astype(int) - ref.astype(int))
    assert diff.mean() < 0.6, diff.mean()
    # boxes: bounding box of the projected corners
    x0, y0, x1, y1 = voc[0]
    pts = cv2.perspectiveTransform(np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float64), hm)[0]
    exp = [max(pts[:, 0].min(), 0), max(pts[:, 1].min(), 0), min(pts[:, 0].max(), ow), min(pts[:, 1].max(), oh)]
    np.testing.assert_allclose(r["bboxes"][0], exp, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_elastic_targets_consistent(make_image, seed):
    img = make_image(96, 128, seed=seed)
    mask = np.zeros((96, 128), np.uint8)
    mask[30:60, 40:90] = 1
    t = augrs.Compose([augrs.ElasticTransform(alpha=60, sigma=6, p=1)],
                      bbox_params=augrs.BboxParams("pascal_voc"),
                      keypoint_params=augrs.KeypointParams("xy", remove_invisible=False), seed=seed)
    r = t(image=img, mask=mask, bboxes=[(40, 30, 90, 60)], keypoints=[(40, 30), (90, 60)])
    ys, xs = np.nonzero(r["mask"])
    bx0, by0, bx1, by1 = r["bboxes"][0]
    assert (xs + 0.5 >= bx0 - 1.5).all() and (xs + 0.5 <= bx1 + 1.5).all()
    assert (ys + 0.5 >= by0 - 1.5).all() and (ys + 0.5 <= by1 + 1.5).all()
    # the field actually moved something
    assert not np.array_equal(r["image"], img)
    # corners tracked as keypoints stay on the box boundary extremes
    k = np.asarray(r["keypoints"])
    assert (k[:, 0] >= bx0 - 1e-6).all() and (k[:, 0] <= bx1 + 1e-6).all()


def test_affine_fit_output_vs_albumentations(image):
    a, r = one(A.Affine(rotate=30, scale=0.8, fit_output=True, p=1), augrs.Affine(rotate=30, scale=0.8, fit_output=True, p=1),
               image=image)
    # same floor/ceil bounds; Albumentations adds one extra row/column
    assert 0 <= a["image"].shape[0] - r["image"].shape[0] <= 1 and 0 <= a["image"].shape[1] - r["image"].shape[1] <= 1
    t = augrs.Compose([augrs.Rotate(limit=(45, 45), fit_output=True, p=1)], seed=0)
    out = t(image=np.full((40, 40, 3), 255, np.uint8))["image"]
    assert out.shape[0] == out.shape[1] == 58  # floor/ceil of 20 -+ 20 * sqrt(2)


def test_rotate_crop_border(image):
    t = augrs.Compose([augrs.Rotate(limit=(20, 20), crop_border=True, interpolation=cv2.INTER_NEAREST, p=1)], seed=0)
    out = t(image=np.full((97, 131, 3), 200, np.uint8))["image"]
    assert out.shape[0] < 97 and out.shape[1] < 131
    assert (out == 200).mean() > 0.999
    a = A.Compose([A.Rotate(limit=(20, 20), crop_border=True, p=1)])(image=image)["image"]
    assert abs(a.shape[0] - out.shape[0]) <= 2 and abs(a.shape[1] - out.shape[1]) <= 2


# ---------------------------------------------------------------- dropout


@pytest.mark.parametrize("seed", range(10))
def test_coarse_dropout_matches_albumentations_given_holes(make_image, seed):
    img = make_image(80, 100, seed=seed)
    mask = np.zeros((80, 100), np.uint8)
    boxes = np.array([[5, 5, 40, 30], [30, 20, 90, 70], [60, 50, 70, 60]], dtype=np.float64)
    kps = np.array([[10.0, 10.0], [50.0, 40.0], [65.0, 55.0], [99.0, 79.0]])
    t = augrs.Compose([augrs.CoarseDropout(num_holes_range=(2, 5), hole_height_range=(10, 30), hole_width_range=(10, 30),
                                           fill=(1, 2, 3), fill_mask=9, p=1)],
                      bbox_params=augrs.BboxParams("pascal_voc", label_fields=["labels"]),
                      keypoint_params=augrs.KeypointParams("xy", pixel_index_coords=True),
                      save_applied_params=True, seed=seed)
    r = t(image=img, mask=mask, bboxes=boxes, labels=[0, 1, 2], keypoints=kps)
    holes = np.array(r["applied_transforms"][0]["params"]["holes"])
    ref = fdrop.cutout(img, holes, (1, 2, 3), np.random.default_rng(0))
    np.testing.assert_array_equal(r["image"], ref)
    np.testing.assert_array_equal(r["mask"], fdrop.cutout(mask, holes, 9, np.random.default_rng(0)))
    exp_boxes = fdrop.filter_bboxes_by_holes(np.c_[boxes, np.arange(3)], holes, (80, 100), 0.0, 0.0)
    np.testing.assert_allclose(r["bboxes"], exp_boxes[:, :4])
    assert list(r["labels"]) == exp_boxes[:, 4].astype(int).tolist()
    exp_kps = fdrop.filter_keypoints_in_holes(kps, holes)
    # augrs keypoints are pixel-index here, so the hole test uses the same convention
    np.testing.assert_allclose(r["keypoints"], exp_kps)


def test_coarse_dropout_options(image):
    h, w = image.shape[:2]
    t = augrs.Compose([augrs.CoarseDropout(num_holes_range=(3, 3), hole_height_range=(0.3, 0.3),
                                           hole_width_range=(0.3, 0.3), fill="random", p=1)], seed=0)
    out = t(image=image, mask=np.zeros((h, w), np.uint8))["image"]
    assert not np.array_equal(out, image)
    # fill_mask=None leaves the mask alone
    out = t(image=image, mask=np.zeros((h, w), np.uint8))
    assert out["mask"].max() == 0
    t = augrs.Compose([augrs.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(97, 97), hole_width_range=(131, 131),
                                           bbox_handling="keep", keypoint_handling="keep", p=1)],
                      bbox_params=augrs.BboxParams("pascal_voc"), keypoint_params=augrs.KeypointParams("xy"))
    out = t(image=image, bboxes=[(1, 1, 5, 5)], keypoints=[(3, 3)])
    assert out["image"].max() == 0 and len(out["bboxes"]) == 1 and len(out["keypoints"]) == 1
    t = augrs.Compose([augrs.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(97, 97), hole_width_range=(131, 131),
                                           p=1)],
                      bbox_params=augrs.BboxParams("pascal_voc"), keypoint_params=augrs.KeypointParams("xy"))
    out = t(image=image, bboxes=[(1, 1, 5, 5)], keypoints=[(3, 3)])
    assert len(out["bboxes"]) == 0 and len(out["keypoints"]) == 0


# ---------------------------------------------------------------- composition


def test_some_of_and_sequential(image):
    t = augrs.Compose([augrs.SomeOf([augrs.HorizontalFlip(p=1), augrs.VerticalFlip(p=1), augrs.Transpose(p=1)], n=2)],
                      save_applied_params=True, seed=0)
    pairs = set()
    for _ in range(40):
        out = t(image=image)
        names = tuple(a["name"] for a in out["applied_transforms"][1:])
        assert len(names) == 2
        pairs.add(names)
    assert len(pairs) == 3
    out = augrs.Compose([augrs.Sequential([augrs.HorizontalFlip(p=1), augrs.VerticalFlip(p=1)], p=1)], seed=0)(image=image)
    np.testing.assert_array_equal(out["image"], image[::-1, ::-1])
    # SomeOf with n > len clamps like Albumentations
    assert augrs.SomeOf([augrs.HorizontalFlip()], n=3).n == 1


# ---------------------------------------------------------------- misc options


@pytest.mark.parametrize("size", [(48, 64), (200, 300)])
def test_mask_interpolation_linear(image, size):
    soft = (image[..., 0].astype(np.float32) / 255)
    t = augrs.Compose([augrs.Resize(*size, mask_interpolation=cv2.INTER_LINEAR)], seed=0)
    out = t(image=image, mask=soft)["mask"]
    ref = cv2.resize(soft, size[::-1], interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(out, ref, atol=1e-4)
    # Compose-level override, like Albumentations
    t = augrs.Compose([augrs.Resize(*size)], mask_interpolation=cv2.INTER_LINEAR, seed=0)
    np.testing.assert_allclose(t(image=image, mask=soft)["mask"], ref, atol=1e-4)
    u8 = image[..., 1]
    out = augrs.Compose([augrs.Resize(*size, mask_interpolation=1)], seed=0)(image=image, mask=u8)["mask"]
    # OpenCV's arm builds use a lower-precision kernel for some single-channel upscales
    max_diff = 2 if cv2_arm_build() else 1
    assert np.abs(out.astype(int) - cv2.resize(u8, size[::-1], interpolation=cv2.INTER_LINEAR).astype(int)).max() <= max_diff


def test_pad_if_needed_divisor_and_crop_padding(image):
    kw = dict(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=48, p=1)
    a, r = one(A.PadIfNeeded(**kw), augrs.PadIfNeeded(**kw), image=image)
    np.testing.assert_array_equal(r["image"], a["image"])
    assert r["image"].shape[:2] == (128, 144)
    kw = dict(height=120, width=150, pad_if_needed=True, border_mode=cv2.BORDER_CONSTANT, fill=7, p=1)
    a, r = one(A.CenterCrop(**kw), augrs.CenterCrop(**kw), image=image)
    np.testing.assert_array_equal(r["image"], a["image"])


def test_longest_max_size_list_and_max_accept_ratio(image):
    t = augrs.Compose([augrs.LongestMaxSize(max_size=[32, 64])], seed=0)
    sizes = {max(t(image=image)["image"].shape[:2]) for _ in range(20)}
    assert sizes == {32, 64}
    t = augrs.Compose([augrs.HorizontalFlip(p=0)], bbox_params=augrs.BboxParams("pascal_voc", max_accept_ratio=3))
    out = t(image=image, bboxes=[(0, 0, 10, 10), (0, 0, 40, 10), (0, 0, 10, 31)])
    assert out["bboxes"] == [(0.0, 0.0, 10.0, 10.0)]


@pytest.mark.parametrize("mode", ["image", "image_per_channel", "min_max", "min_max_per_channel"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_normalize_per_image_modes(image, mode, dtype):
    img = image if dtype == np.uint8 else image.astype(np.float32) / 255
    a, r = one(A.Normalize(normalization=mode, p=1), augrs.Normalize(normalization=mode), image=img)
    np.testing.assert_allclose(r["image"], a["image"], atol=2e-4, rtol=1e-4)
