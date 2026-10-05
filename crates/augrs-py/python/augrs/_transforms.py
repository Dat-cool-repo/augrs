"""Transform classes. Constructors mirror Albumentations 2.0.8 (names, defaults,
scalar conventions); each instance records its constructor arguments so it can
be exported to (and re-created from) the Albumentations dict format."""

from __future__ import annotations

import functools
import inspect
import numbers
from typing import Any, Sequence

from . import _util as U

__all__ = [
    "BasicTransform",
    "BaseCompose",
    "OneOf",
    "SomeOf",
    "Sequential",
    "HorizontalFlip",
    "VerticalFlip",
    "Transpose",
    "RandomRotate90",
    "RandomCrop",
    "CenterCrop",
    "RandomResizedCrop",
    "Resize",
    "LongestMaxSize",
    "SmallestMaxSize",
    "PadIfNeeded",
    "Rotate",
    "Affine",
    "ShiftScaleRotate",
    "Perspective",
    "ElasticTransform",
    "ColorJitter",
    "RandomBrightnessContrast",
    "HueSaturationValue",
    "RandomGamma",
    "CLAHE",
    "GaussNoise",
    "ToGray",
    "CoarseDropout",
    "Normalize",
    "GaussianBlur",
]

#: class name -> class, for deserialisation
REGISTRY: dict[str, type] = {}


class BasicTransform:
    """Base class: a transform is a description that is compiled into the Rust pipeline.

    The arguments passed to ``__init__`` (with defaults filled in) are recorded in
    ``self._init_args``; :meth:`to_dict_private` exports them in the Albumentations format.
    """

    _type = ""
    #: augrs-only arguments, omitted from exported dicts while they keep these defaults
    _augrs_only: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if cls.__name__.startswith("_"):
            return
        REGISTRY[cls.__name__] = cls
        init = cls.__dict__.get("__init__")
        if init is None or getattr(init, "_augrs_wrapped", False):
            return
        sig = inspect.signature(init)

        @functools.wraps(init)
        def wrapped(self, *args: Any, **kw: Any) -> None:
            if "_init_args" not in self.__dict__:
                bound = sig.bind(self, *args, **kw)
                bound.apply_defaults()
                rec = dict(bound.arguments)
                rec.pop("self", None)
                self.__dict__["_init_args"] = rec
            init(self, *args, **kw)

        wrapped._augrs_wrapped = True  # type: ignore[attr-defined]
        cls.__init__ = wrapped  # type: ignore[method-assign]

    def __init__(self, p: float):
        self.p = float(p)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"{type(self).__name__}: p must be in [0, 1], got {p}")

    # -- internal Rust spec ------------------------------------------------
    def _params(self) -> dict:
        return {}

    def _spec(self) -> dict:
        d = {"type": self._type or type(self).__name__, "p": self.p}
        d.update(self._params())
        return d

    # -- Albumentations-format serialisation ------------------------------
    def _export_args(self) -> dict:
        args = dict(getattr(self, "_init_args", {}))
        for k, default in self._augrs_only.items():
            if k in args and args[k] == default:
                del args[k]
        return args

    def to_dict_private(self) -> dict:
        d: dict[str, Any] = {"__class_fullname__": type(self).__name__}
        args = self._export_args()
        d["p"] = self.p
        for k, v in args.items():
            if k == "p":
                continue
            if k == "transforms":
                d[k] = [t.to_dict_private() for t in v]
            elif k in ("interpolation", "mask_interpolation") and v is not None:
                d[k] = U.interp_code(v)  # OpenCV ints, as Albumentations expects
            elif k == "border_mode":
                d[k] = U.border_code(v)
            else:
                d[k] = U.jsonable(v)
        return d

    def to_dict(self) -> dict:
        from . import __version__

        return {"__version__": __version__, "transform": self.to_dict_private()}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._export_args().items() if k != "transforms")
        return f"{type(self).__name__}({args})"


# ---------------------------------------------------------------------------
# composition


class BaseCompose(BasicTransform):
    def __init__(self, transforms: Sequence[Any], p: float):
        super().__init__(p)
        if isinstance(transforms, BasicTransform):
            transforms = [transforms]
        self.transforms = list(transforms)

    def _params(self):
        return {"transforms": [t._spec() for t in self.transforms]}

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self):
        return iter(self.transforms)

    def __getitem__(self, i: int):
        return self.transforms[i]


class OneOf(BaseCompose):
    """Apply exactly one child, chosen with probability proportional to the children's ``p``."""

    def __init__(self, transforms: Sequence[BasicTransform], p: float = 0.5):
        super().__init__(transforms, p)


class SomeOf(BaseCompose):
    """With probability ``p``, pick ``n`` children uniformly (``replace`` = with replacement),
    keep their original order, and apply each with its own probability (Albumentations 2.x)."""

    def __init__(self, transforms: Sequence[BasicTransform], n: int = 1, replace: bool = False, p: float = 1.0):
        super().__init__(transforms, p)
        self.n = int(n)
        if not replace and self.n > len(self.transforms):
            self.n = len(self.transforms)  # Albumentations clamps (with a warning)
        self.replace = bool(replace)

    def _params(self):
        return {**super()._params(), "n": self.n, "replace": self.replace}


class Sequential(BaseCompose):
    """With probability ``p``, apply every child in order (each with its own ``p``)."""

    def __init__(self, transforms: Sequence[BasicTransform], p: float = 0.5):
        super().__init__(transforms, p)


# ---------------------------------------------------------------------------
# geometric


class HorizontalFlip(BasicTransform):
    def __init__(self, p: float = 0.5):
        super().__init__(p)


class VerticalFlip(BasicTransform):
    def __init__(self, p: float = 0.5):
        super().__init__(p)


class Transpose(BasicTransform):
    """Swap rows and columns (output is ``W x H``)."""

    def __init__(self, p: float = 0.5):
        super().__init__(p)


class RandomRotate90(BasicTransform):
    """Rotate by ``k * 90`` degrees counter-clockwise, ``k`` uniform in ``{0, 1, 2, 3}`` (like ``np.rot90``)."""

    def __init__(self, p: float = 1.0):
        super().__init__(p)


def _pad_params(pad_position, border_mode, fill, fill_mask) -> dict:
    return {"pad_position": pad_position, "border_mode": U.border(border_mode), "fill": U.fill(fill),
            "fill_mask": U.fill(fill_mask)}


class RandomCrop(BasicTransform):
    _augrs_only = {"size": None}

    def __init__(self, height=None, width=None, pad_if_needed: bool = False, pad_position: str = "center",
                 border_mode=0, fill=0, fill_mask=0, *, size=None, p: float = 1.0):
        super().__init__(p)
        self.height, self.width = U.size(height, width, size, type(self).__name__)
        self.pad_if_needed = bool(pad_if_needed)
        self._pad = _pad_params(pad_position, border_mode, fill, fill_mask)

    def _export_args(self):
        args = super()._export_args()
        args.pop("size", None)
        args["height"], args["width"] = self.height, self.width
        return args

    def _params(self):
        return {"height": self.height, "width": self.width, "pad_if_needed": self.pad_if_needed, **self._pad}


class CenterCrop(RandomCrop):
    pass


def _no_area_downscale(v):
    U.require(v is None, "area_for_downscale is not supported by augrs (use interpolation='area')")


class RandomResizedCrop(BasicTransform):
    """``RandomResizedCrop(size=(h, w), ...)`` (Albumentations 2.x) or ``RandomResizedCrop(h, w, ...)``."""

    def __init__(self, *hw, size=None, scale=(0.08, 1.0), ratio=(0.75, 4 / 3), interpolation=1,
                 mask_interpolation=0, area_for_downscale=None, height=None, width=None, p: float = 1.0):
        super().__init__(p)
        if len(hw) == 2:
            height, width = hw
        elif len(hw) == 1:
            size = hw[0]
        elif hw:
            raise TypeError("RandomResizedCrop takes (height, width) or size=(height, width) positionally")
        self.height, self.width = U.size(height, width, size, "RandomResizedCrop")
        self.scale = U.pair(scale)
        self.ratio = U.pair(ratio)
        self.interpolation = U.interp(interpolation)
        self.mask_interpolation = U.interp(mask_interpolation)
        _no_area_downscale(area_for_downscale)

    def _export_args(self):
        args = super()._export_args()
        for k in ("hw", "height", "width"):
            args.pop(k, None)
        args["size"] = (self.height, self.width)
        return args

    def _params(self):
        return {"height": self.height, "width": self.width, "scale": self.scale, "ratio": self.ratio,
                "interpolation": self.interpolation, "mask_interpolation": self.mask_interpolation}


class Resize(BasicTransform):
    _augrs_only = {"size": None}

    def __init__(self, height=None, width=None, interpolation=1, mask_interpolation=0, area_for_downscale=None, *,
                 size=None, p: float = 1.0):
        super().__init__(p)
        self.height, self.width = U.size(height, width, size, "Resize")
        self.interpolation = U.interp(interpolation)
        self.mask_interpolation = U.interp(mask_interpolation)
        _no_area_downscale(area_for_downscale)

    def _export_args(self):
        args = super()._export_args()
        args.pop("size", None)
        args["height"], args["width"] = self.height, self.width
        return args

    def _params(self):
        return {"height": self.height, "width": self.width, "interpolation": self.interpolation,
                "mask_interpolation": self.mask_interpolation}


class LongestMaxSize(BasicTransform):
    """Rescale so the longest side equals ``max_size`` (a list = pick one at random)."""

    def __init__(self, max_size=1024, max_size_hw=None, interpolation=1, mask_interpolation=0, area_for_downscale=None,
                 p: float = 1.0):
        super().__init__(p)
        U.require(max_size_hw is None, "max_size_hw is not supported by augrs")
        sizes = [max_size] if isinstance(max_size, numbers.Number) else list(max_size)
        self.max_size = [int(s) for s in sizes]
        self.interpolation = U.interp(interpolation)
        self.mask_interpolation = U.interp(mask_interpolation)
        _no_area_downscale(area_for_downscale)

    def _params(self):
        return {"max_size": self.max_size, "interpolation": self.interpolation,
                "mask_interpolation": self.mask_interpolation}


class SmallestMaxSize(LongestMaxSize):
    pass


class PadIfNeeded(BasicTransform):
    def _export_args(self):
        args = super()._export_args()
        args.pop("padding", None)  # appears in Albumentations' dicts, but its __init__ rejects it
        return args

    def __init__(self, min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None,
                 position: str = "center", border_mode=0, fill=0, fill_mask=0, padding=None, p: float = 1.0):
        super().__init__(p)
        U.require(padding in (None, 0), "PadIfNeeded.padding is not supported by augrs")
        if (min_height is None) == (pad_height_divisor is None) or (min_width is None) == (pad_width_divisor is None):
            raise ValueError("PadIfNeeded: give exactly one of min_height / pad_height_divisor (and of min_width / "
                             "pad_width_divisor)")
        self.min_height = int(min_height or 0)
        self.min_width = int(min_width or 0)
        self.pad_height_divisor = None if pad_height_divisor is None else int(pad_height_divisor)
        self.pad_width_divisor = None if pad_width_divisor is None else int(pad_width_divisor)
        self.position = position
        self.border_mode = U.border(border_mode)
        self.fill = U.fill(fill)
        self.fill_mask = U.fill(fill_mask)

    def _params(self):
        return {"min_height": self.min_height, "min_width": self.min_width,
                "pad_height_divisor": self.pad_height_divisor, "pad_width_divisor": self.pad_width_divisor,
                "position": self.position, "border_mode": self.border_mode, "fill": self.fill,
                "fill_mask": self.fill_mask}


class _AffineLike(BasicTransform):
    def _common(self, interpolation, mask_interpolation, border_mode, fill, fill_mask, rotate_method=None):
        self.interpolation = U.interp(interpolation)
        self.mask_interpolation = U.interp(mask_interpolation)
        self.border_mode = U.border(border_mode)
        self.fill = U.fill(fill)
        self.fill_mask = U.fill(fill_mask)
        if rotate_method is not None:
            if rotate_method not in ("largest_box", "ellipse"):
                raise ValueError("rotate_method must be 'largest_box' or 'ellipse'")
            self.rotate_method = rotate_method

    def _common_params(self):
        d = {"interpolation": self.interpolation, "mask_interpolation": self.mask_interpolation,
             "border_mode": self.border_mode, "fill": self.fill, "fill_mask": self.fill_mask}
        if hasattr(self, "rotate_method"):
            d["rotate_method"] = self.rotate_method
        return d


class Rotate(_AffineLike):
    """Rotate by an angle drawn from ``limit`` (degrees, counter-clockwise).

    ``fit_output=True`` (augrs extension) enlarges the canvas so nothing is cut off;
    ``crop_border=True`` crops the largest rectangle without border pixels.
    """

    _augrs_only = {"fit_output": False}

    def __init__(self, limit=(-90, 90), interpolation=1, border_mode=0, rotate_method: str = "largest_box",
                 crop_border: bool = False, mask_interpolation=0, fill=0, fill_mask=0, fit_output: bool = False,
                 p: float = 0.5):
        super().__init__(p)
        self.limit = U.sym(limit)
        self.crop_border = bool(crop_border)
        self.fit_output = bool(fit_output)
        self._common(interpolation, mask_interpolation, border_mode, fill, fill_mask, rotate_method)

    def _params(self):
        return {"limit": self.limit, "crop_border": self.crop_border, "fit_output": self.fit_output,
                **self._common_params()}


class Affine(_AffineLike):
    """Scale / translate / rotate / shear about the image centre.

    Scalars mean a fixed value (as in Albumentations); pairs are ranges; dicts
    ``{"x": ..., "y": ...}`` set the axes separately.
    """

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0,
                 interpolation=1, mask_interpolation=0, fit_output: bool = False, keep_ratio: bool = False,
                 rotate_method: str = "largest_box", balanced_scale: bool = False, border_mode=0, fill=0, fill_mask=0,
                 p: float = 0.5):
        super().__init__(p)
        if translate_percent is not None and translate_px is not None:
            raise ValueError("pass only one of translate_percent / translate_px")
        self.scale_x, self.scale_y = U.xy(scale)
        self.translate_px = translate_px is not None
        self.translate_x, self.translate_y = U.xy(translate_px if self.translate_px else (translate_percent or 0.0))
        self.rotate = U.fixed(rotate)
        self.shear_x, self.shear_y = U.xy(shear)
        self.keep_ratio = bool(keep_ratio)
        self.balanced_scale = bool(balanced_scale)
        self.fit_output = bool(fit_output)
        self._common(interpolation, mask_interpolation, border_mode, fill, fill_mask, rotate_method)

    def _params(self):
        return {"scale_x": self.scale_x, "scale_y": self.scale_y, "keep_ratio": self.keep_ratio,
                "balanced_scale": self.balanced_scale, "translate_x": self.translate_x,
                "translate_y": self.translate_y, "translate_px": self.translate_px, "rotate": self.rotate,
                "shear_x": self.shear_x, "shear_y": self.shear_y, "fit_output": self.fit_output,
                **self._common_params()}


class ShiftScaleRotate(_AffineLike):
    def __init__(self, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-45, 45), interpolation=1,
                 border_mode=0, shift_limit_x=None, shift_limit_y=None, rotate_method: str = "largest_box",
                 mask_interpolation=0, fill=0, fill_mask=0, p: float = 0.5):
        super().__init__(p)
        self.shift_limit_x = U.sym(shift_limit if shift_limit_x is None else shift_limit_x)
        self.shift_limit_y = U.sym(shift_limit if shift_limit_y is None else shift_limit_y)
        self.scale_limit = U.sym(scale_limit, bias=1.0)
        self.rotate_limit = U.sym(rotate_limit)
        self._common(interpolation, mask_interpolation, border_mode, fill, fill_mask, rotate_method)

    def _params(self):
        return {"shift_limit_x": self.shift_limit_x, "shift_limit_y": self.shift_limit_y,
                "scale_limit": self.scale_limit, "rotate_limit": self.rotate_limit, **self._common_params()}


class Perspective(_AffineLike):
    """Random 4-point perspective. Each corner moves inwards by ``|N(0, scale)| mod 0.32`` of the
    image size and the jittered quadrilateral is mapped onto the output; boxes follow their 4
    corners. (Albumentations 2.0.8 additionally zooms in by ``size / quad_size`` when
    ``keep_size=True``; augrs maps the quadrilateral exactly onto the output.)"""

    def __init__(self, scale=(0.05, 0.1), keep_size: bool = True, fit_output: bool = False, interpolation=1,
                 mask_interpolation=0, border_mode=0, fill=0, fill_mask=0, p: float = 0.5):
        super().__init__(p)
        self.scale = U.from_lo(scale, 0.0)
        self.keep_size = bool(keep_size)
        self.fit_output = bool(fit_output)
        self._common(interpolation, mask_interpolation, border_mode, fill, fill_mask)

    def _params(self):
        return {"scale": self.scale, "keep_size": self.keep_size, "fit_output": self.fit_output,
                **self._common_params()}


class ElasticTransform(_AffineLike):
    """Elastic deformation by smooth random displacement fields (noise in ``[-1, 1]`` blurred
    with ``sigma``, times ``alpha``). Boxes and keypoints use the exact inverse of the field,
    so ``keypoint_remapping_method`` is accepted but not needed."""

    def __init__(self, alpha: float = 1, sigma: float = 50, interpolation=1, approximate: bool = False,
                 same_dxdy: bool = False, mask_interpolation=0, noise_distribution: str = "gaussian",
                 keypoint_remapping_method: str = "mask", border_mode=0, fill=0, fill_mask=0, p: float = 0.5):
        super().__init__(p)
        if keypoint_remapping_method not in ("mask", "direct"):
            raise ValueError("keypoint_remapping_method must be 'mask' or 'direct'")
        if noise_distribution not in ("gaussian", "uniform"):
            raise ValueError("noise_distribution must be 'gaussian' or 'uniform'")
        self.alpha, self.sigma = float(alpha), float(sigma)
        self.approximate, self.same_dxdy = bool(approximate), bool(same_dxdy)
        self.noise_distribution = noise_distribution
        self._common(interpolation, mask_interpolation, border_mode, fill, fill_mask)

    def _params(self):
        return {"alpha": self.alpha, "sigma": self.sigma, "approximate": self.approximate,
                "same_dxdy": self.same_dxdy, "noise_distribution": self.noise_distribution, **self._common_params()}


# ---------------------------------------------------------------------------
# photometric


class ColorJitter(BasicTransform):
    """Brightness / contrast / saturation factors and hue shift, applied in random order.

    A scalar ``b`` means the factor range ``[max(0, 1 - b), 1 + b]``; a scalar hue ``h``
    means ``[-h, h]`` (fraction of a full turn, at most 0.5). On uint8 images the hue
    shift goes through OpenCV-exact 8-bit HSV, like Albumentations.
    """

    def __init__(self, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.5, 0.5),
                 p: float = 0.5):
        super().__init__(p)

        def fac(v):
            if isinstance(v, numbers.Number):
                return (max(0.0, 1.0 - float(v)), 1.0 + float(v))
            return U.pair(v)

        self.brightness = fac(brightness)
        self.contrast = fac(contrast)
        self.saturation = fac(saturation)
        self.hue = U.sym(hue)

    def _params(self):
        return {"brightness": self.brightness, "contrast": self.contrast, "saturation": self.saturation,
                "hue": self.hue}


class RandomBrightnessContrast(BasicTransform):
    """``img * (1 + c) + b * max_value`` (``b * mean(img)`` when ``brightness_by_max=False``)."""

    def __init__(self, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max: bool = True,
                 ensure_safe_range: bool = False, p: float = 0.5):
        super().__init__(p)
        self.brightness_limit = U.sym(brightness_limit)
        self.contrast_limit = U.sym(contrast_limit)
        self.brightness_by_max = bool(brightness_by_max)
        self.ensure_safe_range = bool(ensure_safe_range)

    def _params(self):
        return {"brightness_limit": self.brightness_limit, "contrast_limit": self.contrast_limit,
                "brightness_by_max": self.brightness_by_max, "ensure_safe_range": self.ensure_safe_range}


class HueSaturationValue(BasicTransform):
    """Shift hue (OpenCV units: 180 = full turn), saturation and value (0..255) in 8-bit HSV."""

    def __init__(self, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20),
                 p: float = 0.5):
        super().__init__(p)
        self.hue_shift_limit = U.sym(hue_shift_limit)
        self.sat_shift_limit = U.sym(sat_shift_limit)
        self.val_shift_limit = U.sym(val_shift_limit)

    def _params(self):
        return {"hue_shift_limit": self.hue_shift_limit, "sat_shift_limit": self.sat_shift_limit,
                "val_shift_limit": self.val_shift_limit}


class RandomGamma(BasicTransform):
    """``img ** (gamma / 100)``; a scalar ``g`` means ``(1, g)``."""

    def __init__(self, gamma_limit=(80, 120), p: float = 0.5):
        super().__init__(p)
        self.gamma_limit = U.from_lo(gamma_limit, 1.0)

    def _params(self):
        return {"gamma_limit": self.gamma_limit}


class CLAHE(BasicTransform):
    """Contrast-limited adaptive histogram equalisation (on L of Lab for RGB images)."""

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p: float = 0.5):
        super().__init__(p)
        self.clip_limit = U.from_lo(clip_limit, 1.0)
        self.tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))

    def _params(self):
        return {"clip_limit": self.clip_limit, "tile_grid_size": self.tile_grid_size}


class GaussNoise(BasicTransform):
    """Additive Gaussian noise; ``std_range``/``mean_range`` are fractions of the max value."""

    def __init__(self, std_range=(0.2, 0.44), mean_range=(0.0, 0.0), per_channel: bool = True,
                 noise_scale_factor: float = 1, p: float = 0.5):
        super().__init__(p)
        self.std_range = U.pair(std_range)
        self.mean_range = U.pair(mean_range)
        self.per_channel = bool(per_channel)
        self.noise_scale_factor = float(noise_scale_factor)

    def _params(self):
        return {"std_range": self.std_range, "mean_range": self.mean_range, "per_channel": self.per_channel,
                "noise_scale_factor": self.noise_scale_factor}


class ToGray(BasicTransform):
    def __init__(self, num_output_channels: int = 3, method: str = "weighted_average", p: float = 0.5):
        super().__init__(p)
        U.require(method in ("weighted_average", "desaturation", "average", "max"),
                  f"ToGray(method={method!r}) is not supported by augrs")
        self.num_output_channels = int(num_output_channels)
        self.method = method

    def _params(self):
        return {"num_output_channels": self.num_output_channels, "method": self.method}


class CoarseDropout(BasicTransform):
    """Drop rectangular holes (sizes in pixels if ``hole_height_range[1] >= 1``, else fractions).

    ``fill``: value(s), ``"random"`` or ``"random_uniform"``; ``fill_mask=None`` leaves masks alone.
    augrs extensions: ``bbox_handling`` = ``"shrink"`` (Albumentations: lower visibility, shrink
    to the visible part, drop fully covered boxes), ``"visibility"`` or ``"keep"``;
    ``keypoint_handling`` = ``"auto"`` (drop keypoints in holes if ``remove_invisible``),
    ``"remove"`` or ``"keep"``.
    """

    _augrs_only = {"bbox_handling": "shrink", "keypoint_handling": "auto"}

    def __init__(self, num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0,
                 fill_mask=None, bbox_handling: str = "shrink", keypoint_handling: str = "auto", p: float = 0.5):
        super().__init__(p)
        self.num_holes_range = (int(num_holes_range[0]), int(num_holes_range[1]))
        self.hole_height_range = U.pair(hole_height_range)
        self.hole_width_range = U.pair(hole_width_range)
        if isinstance(fill, str):
            U.require(fill in ("random", "random_uniform"), f"CoarseDropout(fill={fill!r}) is not supported by augrs")
            self.fill: Any = fill
        else:
            self.fill = U.fill(fill)
        self.fill_mask = None if fill_mask is None else U.fill(fill_mask)
        if bbox_handling not in ("shrink", "visibility", "keep"):
            raise ValueError("bbox_handling must be 'shrink', 'visibility' or 'keep'")
        if keypoint_handling not in ("auto", "remove", "keep"):
            raise ValueError("keypoint_handling must be 'auto', 'remove' or 'keep'")
        self.bbox_handling, self.keypoint_handling = bbox_handling, keypoint_handling

    def _params(self):
        return {"num_holes_range": self.num_holes_range, "hole_height_range": self.hole_height_range,
                "hole_width_range": self.hole_width_range, "fill": self.fill, "fill_mask": self.fill_mask,
                "bbox_handling": self.bbox_handling, "keypoint_handling": self.keypoint_handling}


class Normalize(BasicTransform):
    """``(img - mean * max_pixel_value) / (std * max_pixel_value)``; output is float32.
    ``normalization`` may also be ``image``, ``image_per_channel``, ``min_max`` or
    ``min_max_per_channel`` (Albumentations per-image modes)."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value: float = 255.0,
                 normalization: str = "standard", p: float = 1.0):
        super().__init__(p)
        if normalization not in ("standard", "image", "image_per_channel", "min_max", "min_max_per_channel"):
            raise ValueError(f"unknown normalization {normalization!r}")
        self.mean = U.fill(mean if mean is not None else 0.0)
        self.std = U.fill(std if std is not None else 1.0)
        self.max_pixel_value = float(max_pixel_value if max_pixel_value is not None else 1.0)
        self.normalization = normalization

    def _params(self):
        return {"mean": self.mean, "std": self.std, "max_pixel_value": self.max_pixel_value,
                "normalization": self.normalization}


class GaussianBlur(BasicTransform):
    """Gaussian blur. ``blur_limit=(0, 0)`` derives the kernel size from sigma."""

    def __init__(self, blur_limit=(0, 0), sigma_limit=(0.5, 3.0), p: float = 0.5):
        super().__init__(p)
        if isinstance(blur_limit, numbers.Number):
            blur_limit = (0, int(blur_limit))
        self.blur_limit = (int(blur_limit[0]), int(blur_limit[1]))
        self.sigma_limit = U.from_lo(sigma_limit, 0.0)

    def _params(self):
        return {"blur_limit": self.blur_limit, "sigma_limit": self.sigma_limit}
