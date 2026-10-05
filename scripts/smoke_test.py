"""Smoke test of an installed augrs wheel (run with the wheel installed, not the source tree)."""
import io
import platform
import sys
import threading
import time

import numpy as np

import augrs as A

print("python", sys.version.split()[0], platform.platform(), "augrs", A.__version__, "from", A.__file__)
assert "site-packages" in A.__file__, "must test the installed wheel"

rng = np.random.default_rng(0)
img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
mask = (img[..., 0] > 128).astype(np.uint8)
boxes = [(10, 20, 200, 150), (300, 100, 120, 200)]
t = A.Compose(
    [A.RandomResizedCrop(size=(320, 320), scale=(0.3, 1)), A.HorizontalFlip(), A.Affine(rotate=(-20, 20), p=0.7),
     A.Perspective(p=0.3), A.ElasticTransform(alpha=30, sigma=5, p=0.2), A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.8),
     A.HueSaturationValue(p=0.5), A.CLAHE(p=0.2), A.GaussNoise(p=0.3),
     A.CoarseDropout(num_holes_range=(1, 3), fill_mask=0, p=0.5), A.Normalize()],
    bbox_params=A.BboxParams("coco", label_fields=["labels"]), additional_targets={"image2": "image"}, seed=0)
out = t(image=img, image2=img.copy(), mask=mask, bboxes=boxes, labels=["a", "b"])
assert out["image"].shape == (320, 320, 3) and out["image"].dtype == np.float32
np.testing.assert_array_equal(out["image"], out["image2"])
print("single call ok:", len(out["bboxes"]), "boxes")

# determinism + thread-count independence
imgs = [img[i * 10: 400 + i * 10] for i in range(16)]
r1 = t.augment_batch(imgs, bboxes=[boxes] * 16, labels=[["a", "b"]] * 16, seed=5, num_threads=1)
r4 = t.augment_batch(imgs, bboxes=[boxes] * 16, labels=[["a", "b"]] * 16, seed=5, num_threads=4)
for a, b in zip(r1, r4):
    np.testing.assert_array_equal(a["image"], b["image"])
    assert a["bboxes"] == b["bboxes"]
print("batch determinism ok")

# GIL released: a Python thread keeps running during augment_batch
done, counter = threading.Event(), [0]


def spin():
    while not done.is_set():
        counter[0] += 1


th = threading.Thread(target=spin)
th.start()
t0 = time.perf_counter()
t.augment_batch(imgs * 4, num_threads=2)
dt = time.perf_counter() - t0
done.set()
th.join()
print(f"batch of 64 in {dt * 1000:.0f} ms, spinner iterations during it: {counter[0]}")
assert counter[0] > 1000

# serialisation round trip (YAML + JSON)
for fmt in ("yaml", "json"):
    buf = io.StringIO()
    A.save(t, buf, data_format=fmt)
    buf.seek(0)
    t2 = A.load(buf, data_format=fmt, seed=0)
    o1 = t(image=img, image2=img, bboxes=boxes, labels=[1, 2], seed=9)
    o2 = t2(image=img, image2=img, bboxes=boxes, labels=[1, 2], seed=9)
    np.testing.assert_array_equal(o1["image"], o2["image"])
print("serialization ok")

# throughput (single thread) on this machine
tt = A.Compose([A.RandomResizedCrop(size=(512, 512), scale=(0.25, 1)), A.HorizontalFlip(),
                A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=1), A.Normalize()], seed=0)
n = 100
t0 = time.perf_counter()
for i in range(n):
    tt(image=img)
print(f"single-thread: {n / (time.perf_counter() - t0):.0f} img/s (640x480 -> 512x512)")
print("SMOKE OK")
