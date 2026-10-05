"""augrs in a PyTorch DataLoader: augment whole batches in ``collate_fn`` with ``augment_batch``.

The usual pattern (one sample per ``__getitem__``, parallelism through worker processes)
works with augrs too, but augrs releases the GIL and runs a batch on its own thread pool, so
the augmentation can live in ``collate_fn``: no worker processes, no pickling of outputs,
reproducible results independent of the thread count.

    python examples/torch_dataloader.py                       # synthetic images
    python examples/torch_dataloader.py --data data   # COCO subset (scripts/fetch_coco_subset.sh)

Images are decoded once up front so the comparison measures augmentation + batching, not JPEG
decoding. CPU-only torch is enough (pip install torch --index-url https://download.pytorch.org/whl/cpu).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import augrs  # noqa: E402

SIZE = 384


def load_images(data: str | None, n: int):
    """Decoded RGB images + COCO boxes (or synthetic images with random boxes)."""
    rng = np.random.default_rng(0)
    if data and (Path(data) / "subset300.json").exists():
        import cv2

        sub = json.loads((Path(data) / "subset300.json").read_text())
        anns: dict[int, list] = {}
        for a in sub["annotations"]:
            anns.setdefault(a["image_id"], []).append((a["bbox"], a["category_id"]))
        imgs, boxes, labels = [], [], []
        for info in sub["images"][:n]:
            img = cv2.cvtColor(cv2.imread(str(Path(data) / "val2017" / info["file_name"])), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            bl = [(b, c) for b, c in anns.get(info["id"], []) if b[2] > 1 and b[3] > 1]
            b = np.array([bb for bb, _ in bl], dtype=np.float64).reshape(-1, 4)
            b[:, 2] = np.minimum(b[:, 0] + b[:, 2], w) - b[:, 0]
            b[:, 3] = np.minimum(b[:, 1] + b[:, 3], h) - b[:, 1]
            imgs.append(img)
            boxes.append(b)
            labels.append([c for _, c in bl])
        return imgs, boxes, labels
    imgs, boxes, labels = [], [], []
    for _ in range(n):
        h, w = rng.integers(360, 640), rng.integers(400, 640)
        imgs.append(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        k = rng.integers(1, 8)
        xy = rng.uniform(0, 0.6, (k, 2)) * [w, h]
        wh = rng.uniform(0.05, 0.4, (k, 2)) * [w, h]
        boxes.append(np.c_[xy, wh])
        labels.append(rng.integers(0, 80, k).tolist())
    return imgs, boxes, labels


def pipeline(lib, seed=0):
    L = lib
    return L.Compose(
        [
            L.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.25, 1.0)),
            L.HorizontalFlip(p=0.5),
            L.Affine(rotate=(-15, 15), scale=(0.9, 1.1), p=0.5),
            L.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
            L.Normalize(),
        ],
        bbox_params=L.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.1, clip=True),
        seed=seed,
    )


class Raw(Dataset):
    """Returns decoded, un-augmented samples (augmentation happens in collate_fn)."""

    def __init__(self, imgs, boxes, labels):
        self.imgs, self.boxes, self.labels = imgs, boxes, labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.boxes[i], self.labels[i]


class PerSample(Raw):
    """Classic pattern: augment one sample in __getitem__ (Albumentations or augrs)."""

    def __init__(self, imgs, boxes, labels, transform):
        super().__init__(imgs, boxes, labels)
        self.t = transform

    def __getitem__(self, i):
        out = self.t(image=self.imgs[i], bboxes=self.boxes[i], labels=self.labels[i])
        return torch.from_numpy(out["image"]).permute(2, 0, 1), torch.as_tensor(np.asarray(out["bboxes"]).reshape(-1, 4))


def collate_per_sample(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class AugrsCollate:
    """collate_fn that augments the whole batch with augrs (GIL released, rayon threads)."""

    def __init__(self, transform: augrs.Compose, num_threads: int = 4):
        self.t = transform
        self.num_threads = num_threads

    def __call__(self, batch):
        imgs, boxes, labels = zip(*batch)
        outs = self.t.augment_batch(list(imgs), bboxes=list(boxes), labels=list(labels), num_threads=self.num_threads)
        x = torch.from_numpy(np.stack([o["image"] for o in outs])).permute(0, 3, 1, 2)
        targets = [{"boxes": torch.as_tensor(np.asarray(o["bboxes"]).reshape(-1, 4)), "labels": torch.as_tensor(o["labels"])}
                   for o in outs]
        return x, targets


def run(loader, epochs: int) -> float:
    for _ in loader:  # warm-up (workers start, caches)
        break
    n, t0 = 0, time.perf_counter()
    for _ in range(epochs):
        for x, _ in loader:
            n += x.shape[0]
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("AUGRS_DATA"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader worker processes / augrs threads")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--skip-albumentations", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    imgs, boxes, labels = load_images(args.data, args.n)
    print(f"{len(imgs)} images ({'COCO' if args.data else 'synthetic'}), batch {args.batch}, output {SIZE}x{SIZE}, "
          f"{args.workers} workers/threads; torch {torch.__version__}, augrs {augrs.__version__}")
    W, B = args.workers, args.batch
    rows = []
    if not args.skip_albumentations:
        import albumentations as A
        import cv2

        cv2.setNumThreads(1)
        for w in (0, W):
            dl = DataLoader(PerSample(imgs, boxes, labels, pipeline(A)), batch_size=B, num_workers=w,
                            collate_fn=collate_per_sample, persistent_workers=w > 0)
            rows.append((f"albumentations in __getitem__, num_workers={w}", run(dl, args.epochs)))
    dl = DataLoader(PerSample(imgs, boxes, labels, pipeline(augrs)), batch_size=B, num_workers=0,
                    collate_fn=collate_per_sample)
    rows.append(("augrs in __getitem__, num_workers=0", run(dl, args.epochs)))
    for th in sorted({1, W}):
        dl = DataLoader(Raw(imgs, boxes, labels), batch_size=B, num_workers=0, collate_fn=AugrsCollate(pipeline(augrs), th))
        rows.append((f"augrs augment_batch in collate_fn, num_workers=0, {th} thread(s)", run(dl, args.epochs)))
    for name, ips in rows:
        print(f"  {name:62s} {ips:8.1f} img/s")


if __name__ == "__main__":
    main()
