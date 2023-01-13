import albumentations as A

def build_augment(conf):
    train_strong_transform = A.Compose([
                A.Resize(height=conf.image_size, width=conf.image_size, p=1), 
                A.Flip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3, 0.3), p=0.8),
                A.CLAHE(clip_limit=(1,4), p=0.5),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.),
                    A.ElasticTransform(alpha=3),
                ], p=0.20),
                A.OneOf([
                    A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                    A.MedianBlur(),
                ], p=0.20),
                A.OneOf([
                    A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                    A.Downscale(scale_min=0.75, scale_max=0.95),
                ], p=0.2),
                A.Cutout(max_h_size=int(conf.image_size * 0.1), max_w_size=int(conf.image_size * 0.1), num_holes=5, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                ])

    train_weak_transform = A.Compose([
                A.Resize(height=conf.image_size, width=conf.image_size, always_apply=False, p=1.0),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                ])

    valid_transform = A.Compose([
                A.Resize(height=conf.image_size, width=conf.image_size, always_apply=False, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                ])

    return train_strong_transform, train_weak_transform, valid_transform