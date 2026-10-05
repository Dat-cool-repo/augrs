"""``Compose``: target marshalling (numpy <-> Rust), additional targets, batches."""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from . import _util as U
from ._augrs import _Pipeline
from ._transforms import BaseCompose, BasicTransform

_BBOX_FORMATS = ("pascal_voc", "coco", "yolo", "albumentations")
_KP_COLS = {"xy": 2, "yx": 2, "xya": 3, "xys": 3, "xyas": 4, "xysa": 4}
_TARGET_TYPES = ("image", "mask", "masks", "bboxes", "keypoints")


class BboxParams:
    """Bounding-box settings (Albumentations names).

    Differences: ``clip`` defaults to ``True``; ``min_visibility`` uses the fraction of the
    *input* box still visible after the whole pipeline; ``check_each_transform`` and
    ``filter_invalid_bboxes`` are accepted for config compatibility (augrs filters once, at the end).
    """

    def __init__(self, format: str, label_fields: Sequence[str] | None = None, min_area: float = 0.0,
                 min_visibility: float = 0.0, min_width: float = 0.0, min_height: float = 0.0,
                 check_each_transform: bool = True, clip: bool = True, filter_invalid_bboxes: bool = False,
                 max_accept_ratio: float | None = None):
        if format not in _BBOX_FORMATS:
            raise ValueError(f"unknown bbox format {format!r}")
        self.format = format
        self.label_fields = list(label_fields or [])
        self.min_area = float(min_area)
        self.min_visibility = float(min_visibility)
        self.min_width = float(min_width)
        self.min_height = float(min_height)
        self.check_each_transform = bool(check_each_transform)
        self.clip = bool(clip)
        self.filter_invalid_bboxes = bool(filter_invalid_bboxes)
        self.max_accept_ratio = None if max_accept_ratio is None else float(max_accept_ratio)

    def _spec(self) -> dict:
        return {"format": self.format, "min_area": self.min_area, "min_visibility": self.min_visibility,
                "min_width": self.min_width, "min_height": self.min_height, "clip": self.clip,
                "max_accept_ratio": self.max_accept_ratio}

    def to_dict_private(self) -> dict:
        return {"format": self.format, "label_fields": self.label_fields, "min_area": self.min_area,
                "min_visibility": self.min_visibility, "min_width": self.min_width, "min_height": self.min_height,
                "check_each_transform": self.check_each_transform, "clip": self.clip,
                "max_accept_ratio": self.max_accept_ratio}

    to_dict = to_dict_private


class KeypointParams:
    """Keypoint settings (Albumentations names). augrs extension: ``pixel_index_coords=True`` uses
    the pixel-index convention (pixel ``i`` at ``i``) instead of continuous coordinates."""

    def __init__(self, format: str = "xy", label_fields: Sequence[str] | None = None, remove_invisible: bool = True,
                 angle_in_degrees: bool = True, check_each_transform: bool = True, pixel_index_coords: bool = False):
        if format not in _KP_COLS:
            raise ValueError(f"unknown keypoint format {format!r}")
        self.format = format
        self.label_fields = list(label_fields or [])
        self.remove_invisible = bool(remove_invisible)
        self.angle_in_degrees = bool(angle_in_degrees)
        self.check_each_transform = bool(check_each_transform)
        self.pixel_index_coords = bool(pixel_index_coords)

    def _spec(self) -> dict:
        return {"format": self.format, "remove_invisible": self.remove_invisible,
                "angle_in_degrees": self.angle_in_degrees, "pixel_index_coords": self.pixel_index_coords}

    def to_dict_private(self) -> dict:
        d = {"format": self.format, "label_fields": self.label_fields, "remove_invisible": self.remove_invisible,
             "angle_in_degrees": self.angle_in_degrees, "check_each_transform": self.check_each_transform}
        if self.pixel_index_coords:
            d["pixel_index_coords"] = True
        return d

    to_dict = to_dict_private


# ---------------------------------------------------------------------------
# data marshalling


def _as_image(img: Any, name: str = "image") -> tuple[np.ndarray, bool]:
    a = np.asarray(img)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"{name} must have shape (H, W) or (H, W, C), got {a.shape}")
    if a.dtype == np.uint8 or a.dtype == np.float32:
        pass
    elif np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32)
    else:
        raise TypeError(f"{name} dtype must be uint8 or float32, got {a.dtype}")
    return a, squeeze


_MASK_NATIVE = (np.uint8, np.uint16, np.int32, np.float32)


def _as_mask(m: Any) -> tuple[np.ndarray, bool, np.dtype]:
    a = np.asarray(m)
    orig = a.dtype
    squeeze = a.ndim == 2
    if squeeze:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"mask must have shape (H, W) or (H, W, C), got {a.shape}")
    if a.dtype == np.bool_:
        a = a.view(np.uint8)
    elif a.dtype.type in _MASK_NATIVE:
        pass
    elif np.issubdtype(a.dtype, np.integer):
        if a.size and (a.min() < np.iinfo(np.int32).min or a.max() > np.iinfo(np.int32).max):
            raise ValueError("integer mask values must fit in int32")
        a = a.astype(np.int32)
    elif np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32)
    else:
        raise TypeError(f"unsupported mask dtype {a.dtype}")
    return a, squeeze, orig


def _mask_out(a: np.ndarray, squeeze: bool, orig: np.dtype) -> np.ndarray:
    if squeeze:
        a = a[:, :, 0]
    if a.dtype != orig:
        a = a.view(np.bool_) if orig == np.bool_ else a.astype(orig)
    return a


def _split_rows(rows: Any, ncoord: int, what: str):
    """Split boxes/keypoints into an (N, ncoord) float64 array plus extras."""
    if rows is None:
        return np.zeros((0, ncoord)), None, "list"
    if isinstance(rows, np.ndarray):
        a = rows
        if a.ndim != 2 or (a.shape[0] and a.shape[1] < ncoord):
            if a.size == 0:
                return np.zeros((0, ncoord)), a.reshape(0, max(0, a.shape[-1] - ncoord) if a.ndim == 2 else 0), "array"
            raise ValueError(f"{what} array must have shape (N, >={ncoord}), got {a.shape}")
        return np.ascontiguousarray(a[:, :ncoord], dtype=np.float64), a[:, ncoord:], "array"
    rows = list(rows)
    coords = np.zeros((len(rows), ncoord), dtype=np.float64)
    extras = []
    for i, r in enumerate(rows):
        r = tuple(r)
        if len(r) < ncoord:
            raise ValueError(f"{what}[{i}] needs at least {ncoord} values, got {r!r}")
        coords[i] = r[:ncoord]
        extras.append(r[ncoord:])
    return coords, extras, "list"


def _join_rows(coords: np.ndarray, ids: list[int], extras, kind: str, orig: Any):
    if kind == "array":
        ex = extras[ids] if extras is not None and extras.ndim == 2 and extras.shape[1] else None
        dtype = orig.dtype if np.issubdtype(orig.dtype, np.floating) else np.float64
        out = coords.astype(dtype, copy=False)
        if ex is not None:
            out = np.concatenate([out, ex.astype(dtype)], axis=1)
        return out
    return [tuple(float(v) for v in c) + tuple(extras[i]) for c, i in zip(coords, ids)]


def _reindex(values: Any, ids: list[int]):
    if isinstance(values, np.ndarray):
        return values[np.asarray(ids, dtype=np.intp)]
    values = list(values)
    return [values[i] for i in ids]


def _set_mask_interpolation(transforms: Sequence[BasicTransform], value: Any) -> None:
    for t in transforms:
        if isinstance(t, BaseCompose):
            _set_mask_interpolation(t.transforms, value)
        elif hasattr(t, "mask_interpolation"):
            t.mask_interpolation = U.interp(value)


class Compose(BaseCompose):
    """An augmentation pipeline (compiled into a Rust ``_Pipeline``).

    Call it with keyword arguments: ``image`` (required), ``mask``, ``masks``, ``bboxes``,
    ``keypoints``, any label fields named in the params, and any ``additional_targets``
    (e.g. ``{"image2": "image", "mask2": "mask", "bboxes2": "bboxes"}``): additional images
    and masks get exactly the same geometry and sampled colour parameters; additional box and
    keypoint sets are filtered with the same params (label fields apply to the primary set; extra
    tuple columns are kept for every set). Other keyword arguments are passed through unchanged.
    """

    def __init__(self, transforms: Sequence[BasicTransform], bbox_params: BboxParams | dict | None = None,
                 keypoint_params: KeypointParams | dict | None = None,
                 additional_targets: dict[str, str] | None = None, p: float = 1.0, is_check_shapes: bool = True,
                 strict: bool = False, mask_interpolation: Any = None, seed: int | None = None,
                 save_applied_params: bool = False):
        super().__init__(transforms, p)
        if isinstance(bbox_params, dict):
            bbox_params = BboxParams(**bbox_params)
        if isinstance(keypoint_params, dict):
            keypoint_params = KeypointParams(**keypoint_params)
        self.bbox_params = bbox_params
        self.keypoint_params = keypoint_params
        self.additional_targets = dict(additional_targets or {})
        for k, v in self.additional_targets.items():
            if v not in _TARGET_TYPES:
                raise ValueError(f"additional target {k!r} has unknown type {v!r} (use one of {_TARGET_TYPES})")
            if k in _TARGET_TYPES:
                raise ValueError(f"additional target name {k!r} clashes with a built-in target")
        if mask_interpolation is not None:
            _set_mask_interpolation(self.transforms, mask_interpolation)
        self.save_applied_params = bool(save_applied_params)
        self._internal_spec: dict | None = None
        self._build(seed)

    def _build(self, seed: int | None) -> None:
        spec = {
            "transforms": [t._spec() for t in self.transforms],
            "p": self.p,
            "bbox_params": self.bbox_params._spec() if self.bbox_params else None,
            "keypoint_params": self.keypoint_params._spec() if self.keypoint_params else None,
        }
        self._pipe = _Pipeline(json.dumps(spec), None if seed is None else int(seed) & (2**64 - 1))
        self._json = json.dumps(spec)

    # -- serialisation -------------------------------------------------------
    def _spec(self) -> dict:  # nested use: Compose inside Compose / OneOf
        if self._internal_spec is not None:
            return {"type": "Compose", "transforms": self._internal_spec["transforms"], "p": self.p}
        return {"type": "Compose", "transforms": [t._spec() for t in self.transforms], "p": self.p}

    def to_dict_private(self) -> dict:
        if self._internal_spec is not None:
            raise ValueError("this pipeline was loaded from an augrs internal spec (from_json); it has no "
                             "Albumentations-format description. Load it with augrs.from_dict / augrs.load instead.")
        d = {"__class_fullname__": "Compose", "p": self.p, "transforms": [t.to_dict_private() for t in self.transforms],
             "bbox_params": self.bbox_params.to_dict_private() if self.bbox_params else None,
             "keypoint_params": self.keypoint_params.to_dict_private() if self.keypoint_params else None,
             "additional_targets": dict(self.additional_targets), "is_check_shapes": True}
        return d

    def to_json(self) -> str:
        """The internal (Rust) pipeline spec as JSON. For a portable config use :func:`augrs.save`."""
        return self._json

    @classmethod
    def from_json(cls, s: str, seed: int | None = None) -> "Compose":
        """Load from :meth:`to_json` output, or from an Albumentations-format JSON string."""
        spec = json.loads(s)
        if "transform" in spec or "__class_fullname__" in spec:
            from .serialization import from_dict

            return from_dict(spec, seed=seed)
        obj = cls.__new__(cls)
        obj.transforms = []
        obj._init_args = {}
        bp, kp = spec.get("bbox_params"), spec.get("keypoint_params")
        obj.bbox_params = BboxParams(**bp) if bp else None
        obj.keypoint_params = KeypointParams(**kp) if kp else None
        obj.additional_targets = {}
        obj.p = float(spec.get("p", 1.0))
        obj.save_applied_params = False
        obj._internal_spec = spec
        obj._json = json.dumps(spec)
        obj._pipe = _Pipeline(obj._json, seed)
        return obj

    def set_seed(self, seed: int | None) -> None:
        """Restart the random stream (``None`` = non-deterministic)."""
        self._pipe.set_seed(None if seed is None else int(seed) & (2**64 - 1))

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {inner}\n], p={self.p})"

    # -- marshalling -------------------------------------------------------
    def _names(self, kind: str) -> list[str]:
        return [kind] + [k for k, v in self.additional_targets.items() if v == kind]

    def _prepare(self, data: dict) -> dict:
        if "image" not in data or data["image"] is None:
            raise TypeError("image is required: call t(image=...)")
        prep: dict[str, Any] = {}
        prep["image"], prep["image_squeeze"] = _as_image(data["image"])
        extra, extra_meta = [], []
        for name in self._names("image")[1:]:
            if data.get(name) is not None:
                a, sq = _as_image(data[name], name)
                extra.append(a)
                extra_meta.append((name, sq))
        prep["extra"], prep["extra_meta"] = extra, extra_meta
        masks, mask_meta = [], []
        for name in self._names("mask"):
            if data.get(name) is not None:
                a, sq, dt = _as_mask(data[name])
                masks.append(a)
                mask_meta.append((name, "mask", sq, dt))
        for name in self._names("masks"):
            ms = data.get(name)
            if ms is not None:
                kind = "masks_array" if isinstance(ms, np.ndarray) else "masks"
                for m in ms:
                    a, sq, dt = _as_mask(m)
                    masks.append(a)
                    mask_meta.append((name, kind, sq, dt))
        prep["masks"], prep["mask_meta"] = masks, mask_meta
        for kind, params, ncol in (("bboxes", self.bbox_params, 4),
                                   ("keypoints", self.keypoint_params,
                                    _KP_COLS[self.keypoint_params.format] if self.keypoint_params else 2)):
            groups, coords, start = [], [], 0
            for name in self._names(kind):
                if data.get(name) is None:
                    continue
                if params is None:
                    raise ValueError(f"{name} were passed but Compose has no {'bbox' if kind == 'bboxes' else 'keypoint'}_params")
                c, extras, rkind = _split_rows(data[name], ncol, name)
                groups.append((name, start, len(c), extras, rkind))
                coords.append(c)
                start += len(c)
            if groups:
                prep[kind] = (np.ascontiguousarray(np.concatenate(coords, axis=0)), groups)
        return prep

    def _finish(self, data: dict, prep: dict, res: tuple) -> dict:
        image, masks, bboxes, bbox_ids, kps, kp_ids, applied, seed, extra = res
        out = dict(data)
        out["image"] = image[:, :, 0] if prep["image_squeeze"] else image
        for a, (name, sq) in zip(extra, prep["extra_meta"]):
            out[name] = a[:, :, 0] if sq else a
        lists: dict[str, list] = {}
        kinds: dict[str, str] = {}
        for m, (name, kind, sq, dt) in zip(masks, prep["mask_meta"]):
            m = _mask_out(m, sq, dt)
            if kind == "mask":
                out[name] = m
            else:
                lists.setdefault(name, []).append(m)
                kinds[name] = kind
        for name in self._names("masks"):
            if name in lists:
                out[name] = np.stack(lists[name]) if kinds[name] == "masks_array" else lists[name]
            elif data.get(name) is not None:  # empty list / array of masks
                ms = data[name]
                out[name] = list(ms) if not isinstance(ms, np.ndarray) else np.zeros(
                    (0,) + tuple(out["image"].shape[:2]), dtype=np.asarray(ms).dtype)
        for kind, coords, ids, params in (("bboxes", bboxes, bbox_ids, self.bbox_params),
                                          ("keypoints", kps, kp_ids, self.keypoint_params)):
            if kind not in prep:
                continue
            ids = np.asarray(ids, dtype=np.intp)
            for name, start, n, extras, rkind in prep[kind][1]:
                sel = (ids >= start) & (ids < start + n)
                local = (ids[sel] - start).tolist()
                out[name] = _join_rows(coords[sel], local, extras, rkind, data[name])
                if name == kind:
                    for f in params.label_fields:
                        if f in data and data[f] is not None:
                            out[f] = _reindex(data[f], local)
        if applied is not None:
            out["applied_transforms"] = json.loads(applied)
            out["seed"] = seed
        return out

    def __call__(self, *args, seed: int | None = None, **data) -> dict:
        """Augment one sample. ``seed`` (optional) replays a specific draw."""
        if args:
            raise TypeError("pass data as keyword arguments, e.g. t(image=img, bboxes=boxes)")
        data.pop("force_apply", None)
        prep = self._prepare(data)
        bb = prep["bboxes"][0] if "bboxes" in prep else None
        kp = prep["keypoints"][0] if "keypoints" in prep else None
        res = self._pipe.apply(prep["image"], prep["masks"], bb, kp, seed, self.save_applied_params, prep["extra"])
        return self._finish(data, prep, res)

    def augment_batch(self, images: Sequence[Any], *, seed: int | None = None, num_threads: int | None = None,
                      **per_image: Sequence[Any] | None) -> list[dict]:
        """Augment many images in parallel with the GIL released.

        Every other keyword (``bboxes``, ``mask``, ``masks``, ``keypoints``, label fields,
        additional targets) must be a sequence with one entry per image. Sample ``i`` uses a seed
        derived from ``(seed, i)``, so results are reproducible and independent of ``num_threads``.
        """
        n = len(images)
        for k, v in per_image.items():
            if v is not None and len(v) != n:
                raise ValueError(f"{k} has {len(v)} entries but there are {n} images")
        datas, preps = [], []
        for i in range(n):
            d = {"image": images[i]}
            for k, v in per_image.items():
                if v is not None:
                    d[k] = v[i]
            datas.append(d)
            preps.append(self._prepare(d))
        res = self._pipe.apply_batch(
            [p["image"] for p in preps],
            [p["masks"] for p in preps],
            [p["bboxes"][0] if "bboxes" in p else None for p in preps],
            [p["keypoints"][0] if "keypoints" in p else None for p in preps],
            None if seed is None else int(seed) & (2**64 - 1),
            num_threads,
            self.save_applied_params,
            [p["extra"] for p in preps],
        )
        return [self._finish(d, p, r) for d, p, r in zip(datas, preps, res)]


def augment_batch(transform: Compose, images: Sequence[Any], **kwargs) -> list[dict]:
    """Module-level alias for :meth:`Compose.augment_batch`."""
    return transform.augment_batch(images, **kwargs)
