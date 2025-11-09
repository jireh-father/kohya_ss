import albumentations as A
import cv2
TARGET_SIZE = 256

# Example adding OneOf dropout transforms
train_pipeline_step3 = A.Compose([
    A.Perspective(
        scale=[0, 0.02],
        keep_size=True,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT,
        fill=0,
        fill_mask=0,
        p=0.5,
    ),
    A.Affine(
        scale=[0.9, 1.1],
        translate_percent=[-0.05, 0.05],
        rotate=[-15, 15],
        shear=[-2, 2],
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        fit_output=False,
        keep_ratio=True,
        rotate_method="ellipse",
        balanced_scale=True,
        border_mode=cv2.BORDER_REFLECT,
        fill=0,
        fill_mask=0,
        p=0.5,
    ),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(
        brightness=[0.9, 1.1],
        contrast=[0.9, 1.1],
        saturation=[0.9, 1.1],
        hue=[-0.1, 0.1],
        p=0.5,
    ),
    # A.Illumination(
    #     mode="linear",
    #     intensity_range=[0.1, 0.2],
    #     effect_type="both",
    #     angle_range=[0, 360],
    #     center_range=[0.1, 0.9],
    #     sigma_range=[0.2, 1],
    #     p=0.5,
    # ),
    A.FancyPCA(
        alpha=1,
        p=0.5,
    ),
    A.RandomGamma(
        gamma_limit=[80, 100],
        p=0.5,
    ),
    A.RandomToneCurve(
        scale=0.1,
        per_channel=False,
        p=0.5,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=[0.1, 0.2],
        contrast_limit=[0.1, 0.2],
        p=0.5,
    ),
    # A.OneOf([
    #     # Use ranges for number/size of holes
    #     A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
    #                     hole_width_range=(0.1, 0.25), fill_value=0, p=1.0),
    #     # Use ratio and unit size range for grid
    #     A.GridDropout(ratio=0.5, unit_size_range=(0.1, 0.2), p=1.0)
    # ], p=0.5), # Apply one of the dropouts 50% of the time
    # ... other transforms ...
])

