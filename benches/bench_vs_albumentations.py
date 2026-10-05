"""Throughput benchmark: augrs vs Albumentations 2.0.8 on real COCO val2017 images.

Images are decoded once up front; only augmentation is timed. Usage:

    bash scripts/fetch_coco_subset.sh        # downloads the data into $AUGRS_DATA (default ./data)
    python benches/bench_vs_albumentations.py --data data --threads 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import augrs  # noqa: E402


def load(data: Path, n: int):
    sub = json.loads((data / "subset300.json").read_text())
    anns: dict[int, list] = {}
    for a in sub["annotations"]:
        x, y, w, h = a["bbox"]
        if w > 1 and h > 1:
            anns.setdefault(a["image_id"], []).append(((x, y, w, h), a["category_id"]))
    imgs, boxes, labels, masks = [], [], [], []
    for info in sub["images"][:n]:
        img = cv2.cvtColor(cv2.imread(str(data / "val2017" / info["file_name"])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        bl = anns.get(info["id"], [])
        b = np.array([bb for bb, _ in bl], dtype=np.float64).reshape(-1, 4)
        # clip to the image (a few COCO boxes overshoot by a fraction of a pixel)
        if len(b):
            x0 = np.clip(b[:, 0], 0, W)
            y0 = np.clip(b[:, 1], 0, H)
            x1 = np.clip(b[:, 0] + b[:, 2], 0, W)
            y1 = np.clip(b[:, 1] + b[:, 3], 0, H)
            b = np.stack([x0, y0, x1 - x0, y1 - y0], 1)
            keep = (b[:, 2] > 1) & (b[:, 3] > 1)
            b = b[keep]
            lab = [c for (_, c), k in zip(bl, keep) if k]
        else:
            lab = []
        # a synthetic instance-style mask from the boxes (for the segmentation pipeline)
        m = np.zeros((H, W), np.uint8)
        for i, (x, y, w, h) in enumerate(b):
            m[int(y): int(y + h), int(x): int(x + w)] = (i % 250) + 1
        imgs.append(img)
        boxes.append(b)
        labels.append(lab)
        masks.append(m)
    return imgs, boxes, labels, masks


def pipelines(lib, kind: str, seed=0):
    L = A if lib == "albu" else augrs
    common = [
        L.RandomResizedCrop(size=(512, 512), scale=(0.25, 1.0)),
        L.HorizontalFlip(p=0.5),
        L.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), p=0.5),
        L.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        L.GaussianBlur(blur_limit=(3, 5), p=0.2),
        L.Normalize(),
    ]
    if kind == "detection":
        if lib == "albu":
            bp = A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.1, clip=True)
            return A.Compose(common, bbox_params=bp, seed=seed)
        bp = augrs.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.1)
        return augrs.Compose(common, bbox_params=bp, seed=seed)
    return (A if lib == "albu" else augrs).Compose(common, seed=seed)


def sample_kwargs(kind, imgs, boxes, labels, masks, i):
    d = {"image": imgs[i]}
    if kind == "detection":
        d["bboxes"] = boxes[i]
        d["labels"] = labels[i]
    else:
        d["mask"] = masks[i]
    return d


def best_of(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# -- albumentations multiprocessing (fork; inputs inherited, outputs not shipped back)
_G: dict = {}


def _albu_worker_init(kind):
    cv2.setNumThreads(1)
    _G["t"] = pipelines("albu", kind, seed=os.getpid())


def _albu_worker(i):
    out = _G["t"](**sample_kwargs(_G["kind"], _G["imgs"], _G["boxes"], _G["labels"], _G["masks"], i))
    return out["image"].shape[0]


def main():
    ap = argparse.ArgumentParser()
    default_data = Path(__file__).resolve().parent.parent / "data"
    ap.add_argument("--data", default=os.environ.get("AUGRS_DATA", str(default_data)))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "coco300.json"))
    args = ap.parse_args()

    imgs, boxes, labels, masks = load(Path(args.data), args.n)
    n = len(imgs)
    mean_hw = np.mean([i.shape[:2] for i in imgs], axis=0)
    print(f"{n} COCO val2017 images, mean size {mean_hw[0]:.0f}x{mean_hw[1]:.0f}, "
          f"{sum(len(b) for b in boxes)} boxes; threads={args.threads}, batch={args.batch}")
    T = args.threads
    results = {"n_images": n, "threads": T, "batch": args.batch, "cpu_count": os.cpu_count(),
               "platform": platform.platform(), "albumentations": A.__version__, "opencv": cv2.__version__,
               "augrs": augrs.__version__, "rows": []}

    for kind in ("detection", "segmentation"):
        _G.update(kind=kind, imgs=imgs, boxes=boxes, labels=labels, masks=masks)
        rows = []

        def rec(name, secs):
            rows.append({"pipeline": kind, "config": name, "images_per_s": n / secs})
            print(f"  {kind:12s} {name:48s} {n / secs:8.1f} img/s")

        # --- Albumentations, 1 thread
        cv2.setNumThreads(1)
        ta = pipelines("albu", kind)
        for i in range(8):
            ta(**sample_kwargs(kind, imgs, boxes, labels, masks, i))
        rec("albumentations, 1 thread (cv2 threads=1)",
            best_of(lambda: [ta(**sample_kwargs(kind, imgs, boxes, labels, masks, i)) for i in range(n)], args.repeats))

        # --- Albumentations, T Python threads (GIL-bound except inside OpenCV calls)
        with ThreadPoolExecutor(T) as ex:
            tas = [pipelines("albu", kind, seed=k) for k in range(T)]

            def albu_threads():
                list(ex.map(lambda i: tas[i % T](**sample_kwargs(kind, imgs, boxes, labels, masks, i)), range(n)))

            albu_threads()
            rec(f"albumentations, {T} Python threads", best_of(albu_threads, args.repeats))

        # --- Albumentations, T processes (like DataLoader(num_workers=T)); no result transfer
        ctx = mp.get_context("fork")
        with ctx.Pool(T, initializer=_albu_worker_init, initargs=(kind,)) as pool:
            pool.map(_albu_worker, range(T * 4))
            rec(f"albumentations, {T} processes (fork pool)",
                best_of(lambda: pool.map(_albu_worker, range(n), chunksize=8), args.repeats))

        # --- augrs, 1 thread, per-image calls
        tr = pipelines("augrs", kind)
        for i in range(8):
            tr(**sample_kwargs(kind, imgs, boxes, labels, masks, i))
        rec("augrs, 1 thread (per-image calls)",
            best_of(lambda: [tr(**sample_kwargs(kind, imgs, boxes, labels, masks, i)) for i in range(n)], args.repeats))

        # --- augrs, per-image calls from T Python threads (GIL released in Rust)
        with ThreadPoolExecutor(T) as ex:
            def augrs_threads():
                list(ex.map(lambda i: tr(**sample_kwargs(kind, imgs, boxes, labels, masks, i)), range(n)))

            augrs_threads()
            rec(f"augrs, {T} Python threads (per-image calls)", best_of(augrs_threads, args.repeats))

        # --- augrs batch API (rayon), batches of args.batch
        def augrs_batch(threads):
            def run():
                for s in range(0, n, args.batch):
                    sl = slice(s, s + args.batch)
                    if kind == "detection":
                        tr.augment_batch(imgs[sl], bboxes=boxes[sl], labels=labels[sl], num_threads=threads)
                    else:
                        tr.augment_batch(imgs[sl], mask=masks[sl], num_threads=threads)
            return run

        for th in sorted({1, 2, T}):
            augrs_batch(th)()
            rec(f"augrs, augment_batch num_threads={th}", best_of(augrs_batch(th), args.repeats))
        results["rows"].extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
