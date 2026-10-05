"""Regenerate the Albumentations-exported config fixtures (run with albumentations==2.0.8).

    python compat-tests/fixtures/make_albu_fixtures.py
"""

import os
import warnings
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

import albumentations as A  # noqa: E402
import cv2  # noqa: E402

HERE = Path(__file__).parent


def detection():
    return A.Compose(
        [
            A.RandomResizedCrop(size=(512, 512), scale=(0.25, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), p=0.5),
            A.OneOf([A.RandomBrightnessContrast(p=1), A.HueSaturationValue(p=1), A.RandomGamma(p=1)], p=0.8),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.SomeOf([A.GaussianBlur(blur_limit=(3, 5), p=1), A.GaussNoise(p=1), A.CLAHE(p=1)], n=2, p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), p=0.3),
            A.ToGray(p=0.05),
            A.Normalize(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.1, clip=True),
    )


def segmentation():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255),
            A.RandomCrop(height=512, width=512),
            A.Sequential([A.RandomRotate90(p=1), A.Transpose(p=0.5)], p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_REFLECT_101,
                               p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ElasticTransform(alpha=40, sigma=6, p=0.2),
            A.Rotate(limit=10, p=0.3),
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.SmallestMaxSize(max_size=224),
            A.Normalize(mean=0.5, std=0.25),
        ],
        additional_targets={"image2": "image", "mask2": "mask"},
    )


def keypoints():
    return A.Compose(
        [A.HorizontalFlip(p=1), A.Transpose(p=1), A.Resize(64, 80)],
        keypoint_params=A.KeypointParams(format="xy", label_fields=["kp_labels"], remove_invisible=False),
    )


if __name__ == "__main__":
    for name, t in [("detection", detection()), ("segmentation", segmentation()), ("keypoints", keypoints())]:
        A.save(t, str(HERE / f"albu_2.0.8_{name}.json"), data_format="json")
        A.save(t, str(HERE / f"albu_2.0.8_{name}.yaml"), data_format="yaml")
        print("wrote", name)
