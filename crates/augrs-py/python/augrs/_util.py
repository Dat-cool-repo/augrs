"""Argument normalisation helpers (Albumentations conventions)."""

from __future__ import annotations

import numbers
from typing import Any

# OpenCV constants (so cv2.INTER_* / cv2.BORDER_* can be passed directly)
_INTERP = {0: "nearest", 1: "linear", 2: "cubic", 3: "area", 4: "lanczos", 5: "linear", 6: "nearest"}
_INTERP_NAMES = {"nearest", "linear", "cubic", "area", "lanczos"}
_BORDER = {0: "constant", 1: "replicate", 2: "reflect", 3: "wrap", 4: "reflect101"}
_BORDER_NAMES = {"constant", "replicate", "reflect", "wrap", "reflect101", "reflect_101"}


def interp(v: Any) -> str:
    if isinstance(v, str):
        v = v.lower()
        if v not in _INTERP_NAMES:
            raise ValueError(f"unknown interpolation {v!r}")
        return v
    try:
        return _INTERP[int(v)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unsupported interpolation {v!r}") from None


def border(v: Any) -> str:
    if isinstance(v, str):
        v = v.lower()
        if v not in _BORDER_NAMES:
            raise ValueError(f"unknown border mode {v!r}")
        return "reflect101" if v == "reflect_101" else v
    try:
        return _BORDER[int(v)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unsupported border mode {v!r}") from None


_INTERP_CODE = {"nearest": 0, "linear": 1, "cubic": 2, "area": 3, "lanczos": 4}
_BORDER_CODE = {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect101": 4}


def interp_code(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else _INTERP_CODE[interp(v)]


def border_code(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else _BORDER_CODE[border(v)]


def fill(v: Any) -> list[float]:
    if isinstance(v, numbers.Number):
        return [float(v)]
    return [float(x) for x in v]


def pair(v: Any) -> tuple[float, float]:
    a, b = v
    return (float(a), float(b))


def sym(v: Any, bias: float = 0.0) -> tuple[float, float]:
    """Scalar ``v`` -> ``(bias - v, bias + v)``; a pair is shifted by ``bias``."""
    if isinstance(v, numbers.Number):
        return (bias - abs(float(v)), bias + abs(float(v)))
    a, b = pair(v)
    return (bias + a, bias + b)


def from_lo(v: Any, lo: float) -> tuple[float, float]:
    """Scalar ``v`` -> ``(lo, v)``; pairs unchanged."""
    if isinstance(v, numbers.Number):
        return (float(lo), float(v))
    return pair(v)


def fixed(v: Any) -> tuple[float, float]:
    """Scalar ``v`` -> ``(v, v)`` (Albumentations Affine semantics)."""
    if isinstance(v, numbers.Number):
        return (float(v), float(v))
    return pair(v)


def xy(v: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(v, dict):
        return fixed(v.get("x", 0.0)), fixed(v.get("y", 0.0))
    r = fixed(v)
    return r, r


def size(height: Any, width: Any, size_: Any, name: str) -> tuple[int, int]:
    if size_ is None and isinstance(height, (tuple, list)):
        size_ = height
    if size_ is not None:
        h, w = size_
    elif height is None or width is None:
        raise TypeError(f"{name} needs height and width (or size=(h, w))")
    else:
        h, w = height, width
    h, w = int(h), int(w)
    if h <= 0 or w <= 0:
        raise ValueError(f"{name}: height and width must be positive, got {h}x{w}")
    return h, w


def jsonable(v: Any) -> Any:
    """Tuples -> lists, numpy scalars -> Python scalars (recursively)."""
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if hasattr(v, "item") and callable(v.item) and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (TypeError, ValueError):
            return v
    return v


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)
