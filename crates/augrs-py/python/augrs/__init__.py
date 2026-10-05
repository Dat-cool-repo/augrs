"""augrs: fast image augmentation with a Rust core and an Albumentations-like API.

    import augrs as A
    t = A.Compose([A.RandomResizedCrop(size=(512, 512)), A.HorizontalFlip(p=0.5), A.ColorJitter(0.2, 0.2, 0.2)],
                  bbox_params=A.BboxParams(format="coco", label_fields=["labels"]), seed=0)
    out = t(image=img, bboxes=boxes, labels=labels, mask=mask)

    # many images at once, GIL released, rayon threads:
    outs = t.augment_batch(images, bboxes=list_of_boxes, labels=list_of_labels, num_threads=4)

    # configs: Albumentations-format JSON/YAML (also reads files written by A.save)
    A.save(t, "aug.yaml"); t2 = A.load("aug.yaml")

Coordinates: boxes use the usual formats (pascal_voc / coco / yolo / albumentations).
Keypoints are continuous pixel coordinates (pixel ``i`` spans ``[i, i+1)``, its centre is
``i + 0.5``); pass ``KeypointParams(pixel_index_coords=True)`` if your keypoints use the
pixel-index convention (pixel ``i`` is at ``i``).
"""

from __future__ import annotations

from ._augrs import __version__
from ._compose import BboxParams, Compose, KeypointParams, augment_batch
from ._transforms import *  # noqa: F401,F403
from ._transforms import __all__ as _transform_names
from .serialization import from_dict, load, save, to_dict

__all__ = [
    "__version__",
    "Compose",
    "BboxParams",
    "KeypointParams",
    "augment_batch",
    "to_dict",
    "from_dict",
    "save",
    "load",
    *_transform_names,
]
