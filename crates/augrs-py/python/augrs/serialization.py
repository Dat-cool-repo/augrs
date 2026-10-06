"""Pipeline (de)serialisation in the Albumentations format.

``augrs.to_dict(t)`` / ``augrs.from_dict(d)`` and ``augrs.save(t, path)`` / ``augrs.load(path)``
use the same layout as Albumentations' ``A.to_dict`` / ``A.save`` (``{"__version__": ...,
"transform": {"__class_fullname__": "Compose", ...}}``), as JSON or YAML. So:

* configs exported by Albumentations (2.0.8, and 1.x for the shared transforms) load into augrs,
  as long as every transform and argument used is supported (otherwise a ``NotImplementedError``
  names what is missing);
* configs saved by augrs load back into augrs exactly, and into Albumentations when only
  Albumentations arguments are used (augrs-only options are written only when they differ from
  their defaults).
"""

from __future__ import annotations

import inspect
import json
import os
from typing import IO, Any

from ._compose import BboxParams, Compose, KeypointParams
from ._transforms import REGISTRY, BasicTransform

__all__ = ["to_dict", "from_dict", "save", "load"]

# Keys that may appear in dicts written by some Albumentations versions and carry no meaning here.
_IGNORED_KEYS = {"__class_fullname__", "id", "params", "always_apply", "strict"}


def to_dict(transform: BasicTransform) -> dict:
    """Albumentations-format dict: ``{"__version__": ..., "transform": {...}}``."""
    from . import __version__

    return {"__version__": __version__, "transform": transform.to_dict_private()}


# Albumentations configs nest a few levels; this bounds the recursion for malformed input.
_MAX_DEPTH = 64


def _class_name(d: dict) -> str:
    if not isinstance(d, dict):
        raise TypeError(f"a serialised transform must be a dict, got {type(d).__name__}")
    full = d.get("__class_fullname__")
    if not isinstance(full, str):
        raise ValueError(f"not a serialised transform (no __class_fullname__ string): {d!r:.200}")
    return full.rsplit(".", 1)[-1]


def _build(d: dict, seed: int | None = None, top: bool = False, depth: int = 0) -> BasicTransform:
    if depth > _MAX_DEPTH:
        raise ValueError(f"config nested more than {_MAX_DEPTH} levels deep")
    name = _class_name(d)
    args = {k: v for k, v in d.items() if k not in _IGNORED_KEYS}
    if name == "ReplayCompose":
        name = "Compose"
        args.pop("save_key", None)
    if name == "Compose":
        args.pop("save_applied_params", None)
        for k, cls in (("bbox_params", BboxParams), ("keypoint_params", KeypointParams)):
            v = args.get(k)
            if isinstance(v, dict):
                args[k] = _make(cls, {kk: vv for kk, vv in v.items() if kk not in _IGNORED_KEYS}, k)
        if top and seed is not None:
            args["seed"] = seed
    cls = REGISTRY.get(name)
    if cls is None:
        raise NotImplementedError(f"augrs does not implement the {name} transform")
    if "transforms" in args:
        ts = args["transforms"]
        if not isinstance(ts, (list, tuple)):
            raise TypeError(f"{name}.transforms must be a list, got {type(ts).__name__}")
        args["transforms"] = [_build(t, depth=depth + 1) for t in ts]
    if name in ("Compose", "OneOf", "SomeOf", "Sequential"):
        return _make(cls, args, name)
    return _make(cls, args, name)


def _make(cls: type, args: dict, name: str):
    sig = inspect.signature(cls.__init__)
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    unknown = [k for k in args if k not in sig.parameters and not has_var_kw]
    if unknown:
        raise NotImplementedError(f"augrs {name} does not support argument(s) {', '.join(sorted(unknown))}")
    # list -> tuple for range-like values (cosmetic: keeps repr close to Albumentations)
    clean = {k: (tuple(v) if isinstance(v, list) and k != "transforms" and k != "label_fields" else v)
             for k, v in args.items()}
    return cls(**clean)


def from_dict(d: dict, seed: int | None = None) -> BasicTransform:
    """Build a transform/pipeline from :func:`to_dict` output or Albumentations' ``A.to_dict``.

    ``seed`` seeds the top-level ``Compose``.
    """
    if isinstance(d, dict) and "transform" in d:
        d = d["transform"]
    return _build(d, seed=seed, top=True)


def _format(path: Any, data_format: str | None) -> str:
    if data_format is not None:
        if data_format not in ("json", "yaml"):
            raise ValueError("data_format must be 'json' or 'yaml'")
        return data_format
    if isinstance(path, (str, os.PathLike)):
        ext = os.path.splitext(os.fspath(path))[1].lower()
        if ext in (".yaml", ".yml"):
            return "yaml"
    return "json"


def _yaml():
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("YAML support needs PyYAML: pip install pyyaml") from e
    return yaml


def save(transform: BasicTransform, filepath_or_buffer: str | os.PathLike | IO[str], data_format: str | None = None) -> None:
    """Save as JSON or YAML (``data_format`` defaults to the file extension, else JSON)."""
    fmt = _format(filepath_or_buffer, data_format)
    d = to_dict(transform)
    if fmt == "yaml":
        text = _yaml().safe_dump(d, default_flow_style=False, sort_keys=False)
    else:
        text = json.dumps(d, indent=2)
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        with open(filepath_or_buffer, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        filepath_or_buffer.write(text)


def load(filepath_or_buffer: str | os.PathLike | IO[str], data_format: str | None = None,
         seed: int | None = None) -> BasicTransform:
    """Load a pipeline saved by :func:`save` or by Albumentations' ``A.save``."""
    fmt = _format(filepath_or_buffer, data_format)
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        with open(filepath_or_buffer, encoding="utf-8") as f:
            text = f.read()
    else:
        text = filepath_or_buffer.read()
    d = _yaml().safe_load(text) if fmt == "yaml" else json.loads(text)
    return from_dict(d, seed=seed)


def _attach() -> None:
    Compose.from_dict = staticmethod(from_dict)  # type: ignore[attr-defined]


_attach()
