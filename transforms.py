import albumentations as A
import elasticdeform
import numpy as np

def get_training_augmentation_pipeline():
    """
    Creates an augmentation pipeline for training data with rotations and flips.
    Includes random rotations by specific angles and random horizontal/vertical flips.

    Returns:
        albumentations.Compose: A pipeline with training augmentations.
    """
    augmentation_pipeline = A.Compose([
        A.Rotate(limit=(-15, 15), p=0.3),

        A.HorizontalFlip(p=0.5),  

        A.OneOf([
            A.Affine(scale=(0.8, 1.2), balanced_scale=True, p=0.33),  # Zoom
            A.Affine(translate_px=(-10, 10), p=0.33),  # Translation
            A.Affine(shear=(-5, 5), p=0.33),  # Shearing
        ], p=0.5),

        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),

        # A.GaussNoise(var_limit=0.01, p=0.1),

        A.Lambda(image=elastic_deform, p=0.1),  # Elastic deformation
    ], seed=42)

    return augmentation_pipeline

def elastic_deform(image, **kwargs):
    deformed = elasticdeform.deform_random_grid(image.astype(np.float32).squeeze(), sigma=15, points=2)
    deformed = np.clip(deformed, 0.0, 1.0)
    return np.expand_dims(deformed, axis=-1).astype(np.float32)
