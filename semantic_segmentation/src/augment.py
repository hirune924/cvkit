import albumentations as A
import numpy as np

def build_augment(conf):
    if conf.get("augment", True) or conf.get("augment", None) == 'v1':
        train_transform = A.Compose([
                    A.RandomResizedCrop(conf.image_size, conf.image_size, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=1.0),
                    A.Flip(p=0.5),
                    A.ShiftScaleRotate(p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                    A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3, 0.3), p=0.8),
                    A.CLAHE(clip_limit=(1,4), p=0.5),
                    A.OneOf([
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.),
                        A.ElasticTransform(alpha=3),
                    ], p=0.50),
                    A.OneOf([
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        A.MedianBlur(),
                    ], p=0.20),
                    A.OneOf([
                        A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                        A.Downscale(scale_min=0.75, scale_max=0.95),
                    ], p=0.5),
                    A.Cutout(max_h_size=int(conf.image_size * 0.1), max_w_size=int(conf.image_size * 0.1), num_holes=5, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                    ])

    valid_transform = A.Compose([
                #A.Resize(height=conf.image_size, width=conf.image_size, always_apply=False, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                ])

    return train_transform, valid_transform

####################
# Utils
####################
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2