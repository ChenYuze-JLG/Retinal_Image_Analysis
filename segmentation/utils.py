import torch
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import os


def dice_score(pred, target, smooth=1e-5):
    """Compute Dice score."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def iou_score(pred, target, smooth=1e-5):
    """Compute IoU (Intersection over Union)."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def calculate_metrics(pred, target):
    """Return Dice and IoU in a single call."""
    dice = dice_score(pred, target)
    iou = iou_score(pred, target)
    return {'dice': dice.item(), 'iou': iou.item()}


def calculate_biomarkers(pred_mask_np, fov_mask_np):
    """
    Calculate vascular biomarkers inside the field of view (FOV).

    Args:
        pred_mask_np: binary prediction mask (0 or 1)
        fov_mask_np: binary FOV mask (0 or 1)

    Returns:
        dict with vessel density and skeleton length
    """

    # only count pixels inside the FOV
    total_pixels_in_fov = np.sum(fov_mask_np)
    if total_pixels_in_fov == 0:
        return {'vessel_density': 0, 'skeleton_length': 0}

    vessel_pixels_in_fov = np.sum(pred_mask_np * fov_mask_np)

    # vessel density
    density = vessel_pixels_in_fov / total_pixels_in_fov

    # skeleton length
    skeleton = skeletonize(pred_mask_np * fov_mask_np)
    length = np.sum(skeleton)

    return {'vessel_density': density, 'skeleton_length': int(length)}


def save_comparison_plot(original_img, gt_mask, pred_mask, save_path):
    """
    Save a three-panel comparison plot:
    input image, ground truth mask, and predicted mask.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_img)
    axes[0].set_title('Input Retinal Image')
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def unnormalize_image(tensor):
    """Convert a normalized tensor image back to a displayable numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # reverse normalization
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    # (C, H, W) -> (H, W, C)
    np_img = tensor.numpy().transpose(1, 2, 0)
    np_img = np.clip(np_img, 0, 1)
    return np_img