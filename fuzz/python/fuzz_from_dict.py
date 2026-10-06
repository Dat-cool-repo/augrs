"""Fuzz the Albumentations-format config loader (``augrs.from_dict`` / ``augrs.load``).

Starts from real Albumentations 2.0.8 exports (``compat-tests/fixtures``) and from configs saved
by augrs, applies random structural mutations (replace any value with an odd one: NaN, huge or
negative numbers, wrong types, empty or nested containers; delete or add keys; rename transforms;
wrap nodes in deep OneOf/SomeOf chains), then loads the result and, if it loads, runs it on a
small sample and checks that ``to_dict`` round-trips.

Contract: loading or running may raise ``ValueError``, ``TypeError`` or ``NotImplementedError``
(clean, documented errors); anything else (``KeyError``, ``AttributeError``, ``IndexError``,
``OverflowError``, ``RecursionError``, a crash or a hang) is a bug.

    # coverage-guided, with atheris (pip install atheris):
    python fuzz/python/fuzz_from_dict.py -max_total_time=1200 corpus_dir/
    # or a seeded random-mutation loop (no extra dependency):
    python fuzz/python/fuzz_from_dict.py --loop --seconds 1200
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np

import augrs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLEAN = (ValueError, TypeError, NotImplementedError)

ODD_VALUES = [None, True, False, 0, 1, -1, 2, 3, 0.5, -0.5, 255, 1e9, -1e9, 2**31, 2**63, 2**64, 2**70, -2**70,
              1e308, -1e308, float("nan"), float("inf"), float("-inf"), "", "abc", "random", "random_uniform",
              "inpaint_telea", "largest_box", "coco", "yolo", "nearest", [], [0], [1, 2], [2, 1], [0, 0], [-1, -1],
              [1, 2, 3], [[1, 2]], [None, None], [1e308, 1e308], ["a", "b"], {}, {"x": 1}, {"x": [1, 2], "y": 3},
              {"__class_fullname__": "HorizontalFlip"}]


def base_configs() -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "compat-tests", "fixtures", "*.json"))):
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    every = augrs.Compose([
        augrs.OneOf([augrs.HorizontalFlip(), augrs.VerticalFlip(), augrs.Transpose()]),
        augrs.SomeOf([augrs.RandomRotate90(), augrs.Rotate(limit=20), augrs.Affine(scale=(0.9, 1.1), shear=5)], n=2),
        augrs.Sequential([augrs.RandomCrop(16, 16, pad_if_needed=True), augrs.CenterCrop(12, 12)]),
        augrs.RandomResizedCrop(size=(20, 20)), augrs.Resize(24, 24), augrs.LongestMaxSize(max_size=[16, 32]),
        augrs.SmallestMaxSize(max_size=20), augrs.PadIfNeeded(min_height=32, min_width=32),
        augrs.ShiftScaleRotate(), augrs.Perspective(), augrs.ElasticTransform(alpha=20, sigma=4),
        augrs.ColorJitter(), augrs.RandomBrightnessContrast(), augrs.HueSaturationValue(), augrs.RandomGamma(),
        augrs.CLAHE(), augrs.GaussNoise(), augrs.ToGray(), augrs.CoarseDropout(fill_mask=0), augrs.GaussianBlur(),
        augrs.Normalize(),
    ], bbox_params=augrs.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.1),
        keypoint_params=augrs.KeypointParams(format="xyas", label_fields=["kp_labels"]))
    out.append(augrs.to_dict(every))
    return out


BASES = base_configs()
NAMES = sorted(augrs._transforms.REGISTRY) + ["Compose", "ReplayCompose", "Nope", "albumentations.Foo"]


def paths(node, prefix=()):
    """All (container, key) positions in a nested dict/list."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield node, k
            yield from paths(v)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield node, i
            yield from paths(v)


def mutate(d: dict, rnd: random.Random, n: int) -> dict:
    d = copy.deepcopy(d)
    for _ in range(n):
        pos = list(paths(d))
        if not pos:
            break
        cont, key = rnd.choice(pos)
        op = rnd.randrange(8)
        if op <= 2:
            cont[key] = copy.deepcopy(rnd.choice(ODD_VALUES))
        elif op == 3 and isinstance(cont, dict):
            del cont[key]
        elif op == 4 and isinstance(cont, dict):
            cont[rnd.choice(["foo", "p", "transforms", "bbox_params", "__class_fullname__", "always_apply", "n",
                             "seed", "strict", "label_fields"])] = copy.deepcopy(rnd.choice(ODD_VALUES))
        elif op == 5 and isinstance(cont, dict) and "__class_fullname__" in cont:
            cont["__class_fullname__"] = rnd.choice(NAMES)
        elif op == 6 and isinstance(cont[key], dict) and "__class_fullname__" in cont[key]:
            node = cont[key]
            for _ in range(rnd.choice([1, 2, 5, 40, 200])):
                node = {"__class_fullname__": rnd.choice(["OneOf", "SomeOf", "Sequential", "Compose"]),
                        "transforms": [node], "p": 1.0}
            cont[key] = node
        elif isinstance(cont[key], (int, float)) and not isinstance(cont[key], bool):
            v = cont[key]
            cont[key] = rnd.choice([-v, v * 1e6, v + 1, v - 1, 0, int(v) if math.isfinite(v) else 0])
    return d


IMG = (np.arange(23 * 29 * 3) % 251).astype(np.uint8).reshape(23, 29, 3)


def run_one(d: dict) -> None:
    try:
        t = augrs.from_dict(d, seed=1)
    except CLEAN:
        return
    if not isinstance(t, augrs.Compose):
        try:
            t = augrs.Compose([t], seed=1)
        except CLEAN:
            return
    # a loaded pipeline must re-export and reload to the same config
    try:
        d2 = augrs.to_dict(t)
    except CLEAN:
        d2 = None
    if d2 is not None:
        t2 = augrs.from_dict(json.loads(json.dumps(d2)), seed=1)
        assert augrs.to_dict(t2) == d2, "to_dict -> from_dict -> to_dict changed the config"
        # and the JSON / YAML file paths too
        buf = io.StringIO()
        augrs.save(t, buf, data_format="json")
        augrs.load(io.StringIO(buf.getvalue()), data_format="json")
    data = {"image": IMG, "mask": IMG[..., 0] // 100}
    if t.bbox_params is not None:
        fmt = t.bbox_params.format
        data["bboxes"] = [(2, 3, 10, 12)] if fmt in ("pascal_voc", "coco") else [(0.3, 0.4, 0.2, 0.2)]
        for f in t.bbox_params.label_fields:
            if isinstance(f, str) and f.isidentifier():
                data[f] = [0]
    if t.keypoint_params is not None:
        ncol = {"xy": 2, "yx": 2, "xya": 3, "xys": 3, "xyas": 4, "xysa": 4}[t.keypoint_params.format]
        data["keypoints"] = [(5.0, 6.0, 0.0, 1.0)[:ncol]]
        for f in t.keypoint_params.label_fields:
            if isinstance(f, str) and f.isidentifier():
                data[f] = [0]
    for k, v in t.additional_targets.items():
        if not (isinstance(k, str) and k.isidentifier()) or k in data:
            continue
        data[k] = {"image": IMG, "mask": IMG[..., 0], "masks": [IMG[..., 0]], "bboxes": [], "keypoints": []}[v]
    try:
        t(**data)
    except CLEAN:
        pass


class ByteChoices:
    """``random.Random``-like choices that consume the fuzzer's bytes in order, so small input
    mutations make small config mutations (what coverage guidance needs). Exhausted input = 0."""

    def __init__(self, data: bytes):
        self.data, self.i = data, 0

    def _byte(self) -> int:
        b = self.data[self.i] if self.i < len(self.data) else 0
        self.i += 1
        return b

    def randrange(self, n: int) -> int:
        v = self._byte() | (self._byte() << 8) if n > 256 else self._byte()
        return v % n

    def choice(self, seq):
        return seq[self.randrange(len(seq))]


def one_input(data: bytes) -> None:
    rnd = ByteChoices(data)
    base = BASES[rnd.randrange(len(BASES))]
    run_one(mutate(base, rnd, rnd.choice([1, 1, 2, 3, 5, 8])))


def loop(seconds: float, seed: int) -> int:
    rnd = random.Random(seed)
    t_end = time.time() + seconds
    n = fails = 0
    while time.time() < t_end:
        data = rnd.getrandbits(1024).to_bytes(128, "little")
        try:
            one_input(data)
        except Exception:  # noqa: BLE001 - report every unclean failure
            fails += 1
            print("FAIL", data.hex(), file=sys.stderr)
            traceback.print_exc()
            if fails > 20:
                break
        n += 1
    print(f"{n} inputs, {fails} failures in {seconds:.0f} s")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--loop" in sys.argv:
        secs = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 60
        sys.exit(loop(secs, int(os.environ.get("SEED", "0"))))
    if "--replay" in sys.argv:  # --replay <hex>
        one_input(bytes.fromhex(sys.argv[sys.argv.index("--replay") + 1]))
        sys.exit(0)
    import atheris

    with atheris.instrument_imports():
        pass
    atheris.instrument_all()
    atheris.Setup(sys.argv, one_input)
    atheris.Fuzz()
