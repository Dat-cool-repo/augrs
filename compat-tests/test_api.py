"""Python API behaviour: determinism, batching, GIL release, targets, dtypes, errors."""

import threading
import time

import numpy as np
import pytest

import augrs as A


def detection_pipeline(seed=0, fmt="coco", **kw):
    return A.Compose(
        [
            A.RandomResizedCrop(64, 80, scale=(0.3, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-20, 20), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.7),
            A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(),
        ],
        bbox_params=A.BboxParams(format=fmt, label_fields=["labels"], min_visibility=0.1),
        seed=seed,
        **kw,
    )


BOXES = [(10, 10, 40, 30), (50, 20, 60, 70), (0, 0, 131, 97)]


def test_same_seed_same_output(image):
    a, b = detection_pipeline(5), detection_pipeline(5)
    for _ in range(5):
        oa = a(image=image, bboxes=BOXES, labels=["a", "b", "c"])
        ob = b(image=image, bboxes=BOXES, labels=["a", "b", "c"])
        np.testing.assert_array_equal(oa["image"], ob["image"])
        assert oa["bboxes"] == ob["bboxes"] and oa["labels"] == ob["labels"]
    c = detection_pipeline(6)
    oc = [c(image=image)["image"] for _ in range(3)]
    oa = [detection_pipeline(5)(image=image)["image"] for _ in range(1)]
    assert not np.array_equal(oc[0], oa[0])


def test_set_seed_restarts_stream(image):
    t = detection_pipeline(1)
    first = [t(image=image)["image"] for _ in range(3)]
    t.set_seed(1)
    again = [t(image=image)["image"] for _ in range(3)]
    for x, y in zip(first, again):
        np.testing.assert_array_equal(x, y)


def test_batch_independent_of_threads(make_image):
    t = detection_pipeline(0)
    imgs = [make_image(90 + i, 120 - i, seed=i) for i in range(12)]
    boxes = [BOXES[:2]] * 12
    labels = [[1, 2]] * 12
    r1 = t.augment_batch(imgs, bboxes=boxes, labels=labels, seed=42, num_threads=1)
    r4 = t.augment_batch(imgs, bboxes=boxes, labels=labels, seed=42, num_threads=4)
    for a, b in zip(r1, r4):
        np.testing.assert_array_equal(a["image"], b["image"])
        assert a["bboxes"] == b["bboxes"] and a["labels"] == b["labels"]
    # module-level alias
    r = A.augment_batch(t, imgs, seed=42, num_threads=2)
    np.testing.assert_array_equal(r[3]["image"], r1[3]["image"])


def test_batch_releases_gil(make_image):
    t = A.Compose([A.Resize(512, 512), A.Rotate(limit=30, p=1), A.GaussianBlur(sigma_limit=(2, 3), p=1)], seed=0)
    imgs = [make_image(400, 400, seed=i) for i in range(48)]

    def count_for(seconds):
        n, end = 0, time.perf_counter() + seconds
        while time.perf_counter() < end:
            n += 1
        return n

    t0 = time.perf_counter()
    t.augment_batch(imgs[:8], num_threads=1)
    solo_rate = count_for(0.2) / 0.2
    done = threading.Event()
    counter = [0]

    def spin():
        while not done.is_set():
            counter[0] += 1

    th = threading.Thread(target=spin)
    th.start()
    t0 = time.perf_counter()
    t.augment_batch(imgs, num_threads=1)
    dt = time.perf_counter() - t0
    done.set()
    th.join()
    # If the GIL were held during augmentation the spinning thread would be frozen.
    assert counter[0] / dt > 0.25 * solo_rate, (counter[0] / dt, solo_rate)


def test_labels_and_extras_follow_boxes(image):
    t = A.Compose([A.CenterCrop(40, 40)], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels", "ids"]))
    boxes = [(0, 0, 10, 10, "corner"), (40, 30, 80, 60, "middle"), (50, 40, 60, 50, "inner")]
    out = t(image=image, bboxes=boxes, labels=np.array([7, 8, 9]), ids=["x", "y", "z"])
    # crop window is x in [45, 85), y in [28, 68)
    assert [b[4] for b in out["bboxes"]] == ["middle", "inner"]
    assert out["labels"].tolist() == [8, 9] and out["ids"] == ["y", "z"]
    assert out["bboxes"][0][:4] == (0.0, 2.0, 35.0, 32.0)
    # ndarray input with a class column -> ndarray output
    arr = np.array([[0, 0, 10, 10, 1], [40, 30, 80, 60, 2]], dtype=np.float32)
    out = t(image=image, bboxes=arr)
    assert isinstance(out["bboxes"], np.ndarray) and out["bboxes"].dtype == np.float32
    np.testing.assert_allclose(out["bboxes"], [[0, 2, 35, 32, 2]])


def test_yolo_roundtrip_identity(image):
    t = A.Compose([A.HorizontalFlip(p=0)], bbox_params=A.BboxParams(format="yolo"))
    boxes = np.array([[0.5, 0.5, 0.2, 0.3], [0.1, 0.9, 0.2, 0.2]])
    out = t(image=image, bboxes=boxes)
    np.testing.assert_allclose(out["bboxes"], boxes, atol=1e-12)


def test_min_visibility_and_area(image):
    t = A.Compose([A.CenterCrop(20, 20)], bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.5))
    # crop is x in [55, 75), y in [38, 58)
    out = t(image=image, bboxes=[(45, 38, 65, 58), (50, 38, 65, 58)])  # 50% and 66% visible
    assert len(out["bboxes"]) == 2
    out = t(image=image, bboxes=[(44, 38, 65, 58)])  # 10/21 visible
    assert len(out["bboxes"]) == 0
    t = A.Compose([A.CenterCrop(20, 20)], bbox_params=A.BboxParams(format="pascal_voc", min_area=50))
    out = t(image=image, bboxes=[(70, 38, 80, 58)])  # 5x20 = 100 px^2 visible
    assert len(out["bboxes"]) == 1


def test_mask_variants(image):
    h, w = image.shape[:2]
    t = A.Compose([A.HorizontalFlip(p=1), A.Rotate(limit=(90, 90), p=1)], seed=0)
    m_bool = np.zeros((h, w), bool)
    m_bool[5:20, 10:30] = True
    m_i64 = np.arange(h * w, dtype=np.int64).reshape(h, w)
    out = t(image=image, mask=m_bool, masks=[m_i64, m_bool.astype(np.float32)])
    assert out["mask"].dtype == np.bool_ and out["mask"].shape == (h, w)
    assert out["masks"][0].dtype == np.int64 and out["masks"][1].dtype == np.float32
    assert out["mask"].sum() > 0
    stacked = np.stack([m_bool.astype(np.uint8)] * 3)
    out = t(image=image, masks=stacked)
    assert out["masks"].shape == (3, h, w)
    # nearest sampling: mask values are preserved (no new values)
    assert set(np.unique(out["masks"])) <= {0, 1}


def test_grayscale_and_float_images(image):
    t = A.Compose([A.RandomResizedCrop(32, 48), A.HorizontalFlip(), A.ColorJitter(0.2, 0.2), A.Normalize(mean=0.5, std=0.2)], seed=0)
    out = t(image=image[..., 0])
    assert out["image"].shape == (32, 48) and out["image"].dtype == np.float32
    t2 = A.Compose([A.Rotate(limit=10, p=1), A.ColorJitter(p=1)], seed=0)
    f = t2(image=image.astype(np.float32) / 255)["image"]
    assert f.dtype == np.float32 and f.min() >= 0 and f.max() <= 1


def test_keypoints_with_labels(image):
    t = A.Compose([A.CenterCrop(50, 50)], keypoint_params=A.KeypointParams(format="xy", label_fields=["kl"]))
    out = t(image=image, keypoints=[(1, 1), (70, 50), (60.5, 30.25)], kl=["a", "b", "c"])
    # crop starts at x=40, y=23
    assert out["kl"] == ["b", "c"]
    np.testing.assert_allclose(np.asarray(out["keypoints"]), [[30, 27], [20.5, 7.25]])


def test_boxes_contain_mask_after_random_geometry(make_image):
    img = make_image(120, 160, seed=3)
    rng = np.random.default_rng(0)
    t = A.Compose(
        [A.RandomResizedCrop(96, 96, scale=(0.4, 1)), A.HorizontalFlip(), A.VerticalFlip(),
         A.Affine(rotate=(-45, 45), scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), p=1)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["k"]), seed=11)
    for _ in range(30):
        x0, y0 = rng.integers(0, 120), rng.integers(0, 80)
        x1, y1 = x0 + rng.integers(5, 40), y0 + rng.integers(5, 40)
        mask = np.zeros((120, 160), np.uint8)
        mask[y0:y1, x0:x1] = 1
        out = t(image=img, mask=mask, bboxes=[(x0, y0, x1, y1)], k=[0])
        ys, xs = np.nonzero(out["mask"])
        if len(out["bboxes"]) == 0:
            assert len(xs) <= 96  # at most a sliver of nearest-sampled pixels
            continue
        bx0, by0, bx1, by1 = out["bboxes"][0]
        assert (xs + 0.5 >= bx0 - 1.5).all() and (xs + 0.5 <= bx1 + 1.5).all()
        assert (ys + 0.5 >= by0 - 1.5).all() and (ys + 0.5 <= by1 + 1.5).all()


def test_serialization_roundtrip(image):
    t = A.Compose([A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=1)], p=1),
                   A.Compose([A.Rotate(limit=15, p=1)]), A.ShiftScaleRotate(p=1)],
                  bbox_params=A.BboxParams(format="coco"), seed=3)
    t2 = A.Compose.from_json(t.to_json(), seed=3)
    a = t(image=image, bboxes=[(5, 5, 20, 20)])
    b = t2(image=image, bboxes=[(5, 5, 20, 20)])
    np.testing.assert_array_equal(a["image"], b["image"])
    assert a["bboxes"] == b["bboxes"]


def test_save_applied_params_and_replay(image):
    t = A.Compose([A.RandomResizedCrop(32, 32), A.HorizontalFlip()], seed=0, save_applied_params=True)
    out = t(image=image)
    names = [x["name"] for x in out["applied_transforms"]]
    assert names[0] == "RandomResizedCrop"
    replay = t(image=image, seed=out["seed"])
    np.testing.assert_array_equal(replay["image"], out["image"])


def test_errors(image):
    t = A.Compose([A.HorizontalFlip()])
    with pytest.raises(ValueError):
        t(image=image, bboxes=[(0, 0, 1, 1)])
    with pytest.raises(ValueError):
        A.Compose([A.CenterCrop(500, 500)])(image=image)
    with pytest.raises(ValueError):
        A.Compose([A.HorizontalFlip(p=3)])
    with pytest.raises(TypeError):
        t(image)
    with pytest.raises(ValueError):
        A.Compose([A.HorizontalFlip()])(image=image, mask=np.zeros((5, 5), np.uint8))
    t = A.Compose([A.HorizontalFlip()], bbox_params=A.BboxParams(format="pascal_voc", clip=False))
    with pytest.raises(ValueError):
        t(image=image, bboxes=[(0, 0, 500, 10)])
